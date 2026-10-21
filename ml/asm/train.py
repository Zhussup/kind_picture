#!/usr/bin/env python3
"""
train.py — LoRA SFT (supervised fine-tuning) для asm -> C.

Что делает:
  1. читает dataset/train.jsonl (chat-пары «листинг -> C»);
  2. собирает для каждой пары токены полного диалога и маскирует ЛОСС так,
     чтобы он считался только на ответе ассистента (C-коде), а не на
     системном промпте и листинге asm;
  3. навешивает LoRA-адаптер на Qwen3-0.6B: базовые веса заморожены,
     обучаются только маленькие матрицы-добавки;
  4. обучает несколько эпох и сохраняет адаптер в checkpoints/lora-<tag>/.

Зачем masked loss (маска лосса):
  В chat-формате модель видит system + user + assistant. Если считать лосс
  на всех токенах, модель тратит ёмкость на воспроизведение промпта и
  листинга. Мы заменяем метки промпта на -100 — cross_entropy игнорирует
  такие позиции, и градиент идёт только от C-ответа.

Почему LoRA, а не полный fine-tune:
  6 ГБ видеопамяти. Полные градиенты + состояния Adam для 0.6B — это
  несколько гигабайт сверх весов. LoRA обучает ~1% параметров, состояния
  оптимизатора занимают десятки мегабайт; базовые веса остаются bf16.
  Gradient checkpointing дополнительно снижает память активаций (ценой
  повторного forward на backward).

Запуск (дымовой прогон, переобучение на 60 коротких парах):
  python3 train.py --limit 60 --epochs 3 --max-seq-len 1024 \
      --grad-accum 4 --tag smoke

Полный прогон:
  python3 train.py --epochs 2 --max-seq-len 2048 --grad-accum 8 --tag full

Дальше адаптер проверяется тем же eval.py:
  python3 eval.py --model Qwen/Qwen3-0.6B \
      --adapter checkpoints/lora-smoke/final --tag smoke
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "checkpoints")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# ДАННЫЕ
# ---------------------------------------------------------------------------

def load_rows(path, limit=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def ids_of(enc):
    """apply_chat_template(tokenize=True) в разных версиях transformers
    возвращает то список id, то BatchEncoding, то список списков.
    Приводим к плоскому списку int."""
    x = enc["input_ids"] if hasattr(enc, "keys") and "input_ids" in enc else enc
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


def apply_template(tokenizer, msgs, add_generation_prompt):
    """Рендер чата в токены. enable_thinking=False для Qwen3: в датасете
    ответы без блока рассуждений, и на обучении шаблон должен совпадать
    с тем, что делает eval.py."""
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=True,
            add_generation_prompt=add_generation_prompt)


def encode_sample(tokenizer, sample):
    """Полный диалог + позиция, с которой начинается ответ ассистента.

    Токены промпта получают метку -100 (игнорируются лоссом).
    Промпт рендерится тем же шаблоном с add_generation_prompt=True —
    он является точным префиксом полного диалога; на 100 проверенных
    парах префикс совпадал побайтово. На всякий случай есть fallback:
    если префикс не совпал, берём длину общего начала.
    """
    msgs = sample["messages"]
    full = ids_of(apply_template(tokenizer, msgs, False))
    prompt = ids_of(apply_template(tokenizer, msgs[:-1], True))

    n = len(prompt)
    if full[:n] != prompt:
        n = 0
        for a, b in zip(full, prompt):
            if a != b:
                break
            n += 1

    labels = list(full)
    for i in range(n):
        labels[i] = -100
    if all(x == -100 for x in labels):
        return None                      # пустой ответ — не учим

    return {"input_ids": full, "labels": labels, "prompt_len": n}


# ---------------------------------------------------------------------------
# МОДЕЛЬ
# ---------------------------------------------------------------------------

def load_base_model(name):
    kwargs = {"device_map": "auto"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        # transformers < 5
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, **kwargs)
    model.config.use_cache = False        # несовместимо с checkpointing
    return model


def attach_lora(model, args):
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def lr_factor(step, total, warmup, floor=0.05):
    """Линейный разогрев + косинусное затухание до floor*base_lr."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))


# ---------------------------------------------------------------------------
# ОБУЧЕНИЕ
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LoRA SFT asm -> C")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train", default=TRAIN_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0,
                    help="взять первых N ПОДХОДЯЩИХ пар (после фильтра длины)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(args.train):
        print(f"нет файла {args.train}: сначала запусти build_dataset.py",
              file=sys.stderr)
        sys.exit(3)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- фильтр по длине: не влезающие пары пропускаем, а не обрезаем
    # (обрезать листинг слева — значит сломать вход; обрезать ответ —
    # значит учить модель на недописанном C)
    rows = load_rows(args.train)
    data, skipped_long, skipped_empty, lengths = [], 0, 0, []
    for row in rows:
        ex = encode_sample(tokenizer, row)
        if ex is None:
            skipped_empty += 1
            continue
        lengths.append(len(ex["input_ids"]))
        if len(ex["input_ids"]) > args.max_seq_len:
            skipped_long += 1
            continue
        data.append(ex)
        if args.limit and len(data) >= args.limit:
            break

    if not data:
        print("после фильтра не осталось ни одной пары "
              f"(max_seq_len={args.max_seq_len})", file=sys.stderr)
        sys.exit(3)

    lengths.sort()
    print(f"пар в файле: {len(rows)}; взято: {len(data)}; "
          f"длиннее лимита: {skipped_long}; пустой ответ: {skipped_empty}")
    print(f"токены (все): p50={lengths[len(lengths)//2]}, "
          f"p90={lengths[int(len(lengths)*0.9)]}, max={lengths[-1]}")
    print(f"max_seq_len={args.max_seq_len}, LoRA r={args.lora_r}, "
          f"lr={args.lr}, epochs={args.epochs}, "
          f"grad_accum={args.grad_accum}")

    model = load_base_model(args.model)
    model = attach_lora(model, args)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, args.warmup))

    out_root = os.path.join(args.out_dir, f"lora-{args.tag}")
    os.makedirs(out_root, exist_ok=True)
    history = []
    running_loss, running_n, step = 0.0, 0, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(order)
        ep_loss, ep_n, ep_t0 = 0.0, 0, time.time()

        for pos, idx in enumerate(order):
            ex = data[idx]
            ids = torch.tensor([ex["input_ids"]], dtype=torch.long,
                               device=device)
            labels = torch.tensor([ex["labels"]], dtype=torch.long,
                                  device=device)
            attn = torch.ones_like(ids)

            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = out.loss
            (loss / args.grad_accum).backward()

            running_loss += loss.item()
            running_n += 1
            ep_loss += loss.item()
            ep_n += 1

            last = pos == len(order) - 1
            if (pos + 1) % args.grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                hist = {"step": step, "epoch": epoch,
                        "loss": running_loss / running_n,
                        "lr": scheduler.get_last_lr()[0],
                        "sec": round(time.time() - started, 1)}
                history.append(hist)
                running_loss, running_n = 0.0, 0
                if step % args.log_every == 0 or last:
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss={hist['loss']:.4f} lr={hist['lr']:.2e} "
                          f"elapsed={hist['sec']}s", flush=True)

        ep_sec = time.time() - ep_t0
        print(f"--- эпоха {epoch}: loss={ep_loss / max(1, ep_n):.4f}, "
              f"{ep_sec:.0f}s ---", flush=True)
        ckpt = os.path.join(out_root, f"epoch-{epoch}")
        model.save_pretrained(ckpt)
        print(f"адаптер сохранён: {ckpt}", flush=True)

    final = os.path.join(out_root, "final")
    model.save_pretrained(final)
    state = {
        "model": args.model,
        "train": os.path.abspath(args.train),
        "tag": args.tag,
        "n_used": len(data),
        "n_skipped_long": skipped_long,
        "n_skipped_empty": skipped_empty,
        "max_seq_len": args.max_seq_len,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout,
                 "target_modules": TARGET_MODULES},
        "lr": args.lr, "epochs": args.epochs,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "steps": step,
        "elapsed_seconds": round(time.time() - started, 1),
        "history": history,
        "ts": now_iso(),
    }
    with open(os.path.join(out_root, "train_state.json"), "w",
              encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"\nготово: {final}")
    print(f"лог обучения: {os.path.join(out_root, 'train_state.json')}")


if __name__ == "__main__":
    main()
