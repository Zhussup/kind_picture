#!/usr/bin/env python3
"""
train.py — LoRA SFT (supervised fine-tuning) for asm -> C.

What it does:
  1. reads dataset/train.jsonl (chat pairs "listing -> C");
  2. tokenizes each full dialogue and masks the LOSS so that it is computed
     only on the assistant's answer (the C code), not on the system prompt
     and the asm listing;
  3. attaches a LoRA adapter to Qwen3-0.6B: the base weights stay frozen,
     only small additive matrices are trained;
  4. trains for a few epochs and saves the adapter to
     checkpoints/lora-<tag>/.

Why masked loss:
  In chat format the model sees system + user + assistant. If the loss is
  computed on every token, the model wastes capacity reproducing the prompt
  and the listing. We replace the prompt labels with -100 — cross_entropy
  ignores those positions, so the gradient comes only from the C answer.

Why LoRA instead of full fine-tuning:
  6 GB of VRAM. Full gradients plus Adam states for 0.6B take several extra
  gigabytes on top of the weights. LoRA trains ~1% of the parameters and the
  optimizer state is tens of megabytes; the base weights stay in bf16.
  Gradient checkpointing further reduces activation memory at the cost of a
  repeated forward pass during backward.

Run (smoke test, overfit on 60 short pairs):
  python3 train.py --limit 60 --epochs 3 --max-seq-len 1024 \
      --grad-accum 4 --tag smoke

Full run:
  python3 train.py --epochs 2 --max-seq-len 2048 --grad-accum 8 --tag full

The adapter is then checked with the same eval.py:
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
import torch.nn.functional as F
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
# DATA
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
    """apply_chat_template(tokenize=True) returns a plain list of ids, a
    BatchEncoding, or a list of lists depending on the transformers version.
    Normalize it to a flat list of ints."""
    x = enc["input_ids"] if hasattr(enc, "keys") and "input_ids" in enc else enc
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


def apply_template(tokenizer, msgs, add_generation_prompt):
    """Render the chat into tokens. enable_thinking=False for Qwen3: the
    dataset answers contain no reasoning block, and the template at training
    time must match what eval.py does."""
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
    """Full dialogue plus the position where the assistant answer starts.

    Prompt tokens get label -100 (ignored by the loss). The prompt is rendered
    with the same template using add_generation_prompt=True — it is an exact
    prefix of the full dialogue; on 100 checked pairs the prefix matched
    token for token. As a safeguard, if the prefix does not match, the length
    of the common prefix is used instead.
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
        return None                      # empty answer — nothing to learn

    return {"input_ids": full, "labels": labels, "prompt_len": n}


# ---------------------------------------------------------------------------
# MODEL
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
    model.config.use_cache = False        # incompatible with checkpointing
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
    """Linear warmup plus cosine decay down to floor*base_lr."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))


def answer_loss(model, ids, labels, attn, chunk=128):
    """Cross-entropy only on the answer tokens, without full-vocabulary
    logits for the prompt.

    The regular CausalLM.forward computes logits [L, vocab] at EVERY
    position. With L=2048 and vocab=151936 that is ~600 MB in bf16, plus the
    same amount in fp32 inside cross_entropy — it does not fit in 6 GB
    together with the activations (confirmed by the full run: CUDA OOM).

    Here:
      1. run only the transformer (base.model) and take the hidden states;
      2. select the positions where labels != -100 (the assistant answer)
         and take the hidden state one position earlier — a causal model
         predicts the next token;
      3. compute lm_head and CE in chunks of `chunk` positions.

    The result is mathematically the same as the HF masked loss: the same
    positions, the same cross-entropy; only the order of computation differs.
    """
    base = model.base_model.model          # Qwen3ForCausalLM under LoRA
    out = base.model(input_ids=ids, attention_mask=attn, use_cache=False)
    hidden = out.last_hidden_state         # [1, L, H]

    ans = (labels[0] != -100).nonzero(as_tuple=False).flatten()
    ans = ans[ans > 0]                     # the first token has no previous one
    if ans.numel() == 0:
        return None

    total, n = 0.0, 0
    for pos in ans.split(chunk):
        h = hidden[0].index_select(0, pos - 1)
        logits = base.lm_head(h).float()
        target = labels[0].index_select(0, pos)
        total = total + F.cross_entropy(logits, target, reduction="sum")
        n += pos.numel()
    return total / n


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LoRA SFT for asm -> C")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train", default=TRAIN_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0,
                    help="take the first N SUITABLE pairs (after the length "
                         "filter)")
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
    ap.add_argument("--loss-chunk", type=int, default=128,
                    help="how many answer positions to process at once in "
                         "answer_loss")
    args = ap.parse_args()

    if not os.path.exists(args.train):
        print(f"no such file {args.train}: run build_dataset.py first",
              file=sys.stderr)
        sys.exit(3)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- length filter: skip pairs that do not fit instead of truncating
    # (truncating the listing from the left breaks the input; truncating the
    # answer teaches the model on incomplete C)
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
        print("no pairs left after filtering "
              f"(max_seq_len={args.max_seq_len})", file=sys.stderr)
        sys.exit(3)

    lengths.sort()
    print(f"pairs in file: {len(rows)}; used: {len(data)}; "
          f"longer than the limit: {skipped_long}; empty answer: "
          f"{skipped_empty}")
    print(f"tokens (all): p50={lengths[len(lengths)//2]}, "
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

            loss = answer_loss(model, ids, labels, attn, args.loss_chunk)
            if loss is None:
                continue
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
        print(f"--- epoch {epoch}: loss={ep_loss / max(1, ep_n):.4f}, "
              f"{ep_sec:.0f}s ---", flush=True)
        ckpt = os.path.join(out_root, f"epoch-{epoch}")
        model.save_pretrained(ckpt)
        print(f"adapter saved: {ckpt}", flush=True)

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

    print(f"\ndone: {final}")
    print(f"training log: {os.path.join(out_root, 'train_state.json')}")


if __name__ == "__main__":
    main()
