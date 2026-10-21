#!/usr/bin/env python3
"""
eval.py — оценка модели на val-наборе asm -> C.

Что делает:
  1. читает dataset/val.jsonl (пары «листинг -> C», которых не было в обучении);
  2. для каждой пары собирает промпт (system + user, как в датасете) и просит
     модель восстановить C; генерация ЖАДНАЯ (do_sample=False) — ответ
     детерминирован, сравнение не зависит от случайности;
  3. вынимает из ответа блок ```c (тот же экстрактор, что в gen_c.py);
  4. компилирует найденный C и прогоняет difftest против эталонного C;
  5. считает метрики:
       compile_rate — доля ответов, которые вообще собрались;
       pass_rate    — доля пар, где поведение совпало на всех входах.

Эталонный C берётся из assistant-сообщения val-пары, наблюдаемое поведение —
из прогона этого C. Ответ модели сравнивается с ним по ПОВЕДЕНИЮ, а не по
тексту: другой C с тем же поведением — правильный ответ.

Сравнение «до/после обучения»: один и тот же val, те же входы, те же
параметры генерации; отличается только --model/--adapter.

Запуск:
  python3 eval.py --model Qwen/Qwen3-0.6B --limit 5
  python3 eval.py --model Qwen/Qwen3-0.6B --adapter checkpoints/lora --tag lora
  python3 eval.py --model Qwen/Qwen3-0.6B --cases my_inputs.json --out-dir results
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import difftest           # compare(), DEFAULT_INPUTS
from gen_c import extract_c, now_iso

VAL_PATH = os.path.join(BASE_DIR, "dataset", "val.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "eval_results")

C_FENCE_RE = re.compile(r"```c\s*\n(.*?)```", re.S)


def load_rows(path, limit):
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


def reference_c(sample):
    """Достаёт C из assistant-сообщения val-пары (там он в блоке ```c)."""
    text = sample["messages"][-1]["content"]
    m = C_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def build_prompt(tokenizer, sample):
    """System + user из датасета -> строка промпта в чат-шаблоне модели.

    enable_thinking=False для Qwen3: иначе модель сначала «думает» и тратит
    сотни токенов на рассуждения, которых в нашем ответе быть не должно.
    Если шаблон модели такого аргумента не знает — падаем обратно на обычный
    вызов (TypeError)."""
    msgs = [m for m in sample["messages"] if m["role"] in ("system", "user")]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                     # жадная генерация: детерминизм
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    # упёрлись ли в лимит: тогда ответ обрезан и «провал» может быть артефактом
    hit_max = int(new_ids.shape[0]) >= max_new_tokens
    return text, hit_max


ASM_LINE_RE = re.compile(
    r"^\s*(\.(text|section|globl|type|size|LC\d+|string|data|bss)\b"
    r"|push[q|l]?\s|pop[q|l]?\s|mov[q|l|b|w]?\s|lea[q|l]?\s|call\s|ret\b"
    r"|j(mp|e|ne|s|ns|g|ge|l|le|z|nz)\s|add[q|l]?\s|sub[q|l]?\s|"
    r"cmp[q|l|b|w]?\s|test[q|l|b|w]?\s|xor[q|l]?\s|imul[q|l]?\s|"
    r"[A-Za-z_.][\w.]*:)")


def output_kind(text):
    """Грубая классификация ответа: 'c', 'asm' (модель скопировала листинг
    в блок ```c) или 'other'. Нужна для диагностики, а не для метрики."""
    if not text:
        return "other"
    if "#include" in text or re.search(r"\b(int|void|char|long)\s+main\s*\(", text):
        return "c"
    asm_lines = sum(1 for ln in text.splitlines() if ASM_LINE_RE.match(ln))
    return "asm" if asm_lines >= 3 else "other"


def load_model(name, adapter):
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = {"device_map": "auto"}
    try:
        # transformers >= 5: параметр называется dtype
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[-80:]


def main():
    ap = argparse.ArgumentParser(description="Оценка asm->C на val-наборе")
    ap.add_argument("--model", required=True,
                    help="HF id или путь к весам (Qwen/Qwen3-0.6B)")
    ap.add_argument("--adapter", help="путь к LoRA-адаптеру (для дообученной)")
    ap.add_argument("--val", default=VAL_PATH, help="val.jsonl")
    ap.add_argument("--cases", help="JSON со входами для difftest")
    ap.add_argument("--limit", type=int, default=0, help="взять первые N пар")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="", help="метка прогона (имя файлов)")
    args = ap.parse_args()

    if not os.path.exists(args.val):
        print(f"нет файла {args.val}: сначала запусти build_dataset.py",
              file=sys.stderr)
        sys.exit(3)

    inputs = difftest.DEFAULT_INPUTS
    if args.cases:
        try:
            with open(args.cases, "r", encoding="utf-8") as f:
                inputs = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"ошибка в cases: {e}", file=sys.stderr)
            sys.exit(3)

    rows = load_rows(args.val, args.limit)
    if not rows:
        print("val пуст", file=sys.stderr)
        sys.exit(3)

    print(f"модель: {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    print(f"val: {len(rows)} пар, входов difftest: {len(inputs)}")
    model, tokenizer = load_model(args.model, args.adapter)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = "-".join(x for x in (slug(args.tag or args.model), stamp) if x)
    pred_path = os.path.join(args.out_dir, f"predictions_{base}.jsonl")
    sum_path = os.path.join(args.out_dir, f"summary_{base}.json")

    started = time.time()
    stats = {"n": 0, "compiled": 0, "passed": 0, "ref_errors": 0,
             "truncated": 0, "kinds": {}, "by_opt": {}}
    with open(pred_path, "w", encoding="utf-8") as pred_f:
        for i, sample in enumerate(rows, 1):
            ref_c = reference_c(sample)
            prompt = build_prompt(tokenizer, sample)
            t0 = time.time()
            answer, hit_max = generate(model, tokenizer, prompt,
                                       args.max_new_tokens)
            gen_sec = time.time() - t0
            cand_c = extract_c(answer)
            kind = output_kind(answer)
            stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1

            if cand_c is None:
                verdict = {"ref_compiled": True, "cand_compiled": False,
                           "stage": "cand_compile", "n_total": len(inputs),
                           "n_passed": 0, "all_match": False,
                           "error": "в ответе нет блока ```c", "cases": []}
            else:
                verdict = difftest.compare(ref_c, cand_c, inputs)

            if not verdict["ref_compiled"]:
                stats["ref_errors"] += 1
            compiled = bool(verdict["cand_compiled"])
            passed = bool(verdict["all_match"])
            stats["n"] += 1
            stats["compiled"] += compiled
            stats["passed"] += passed
            stats["truncated"] += int(hit_max)
            opt = sample.get("opt", "?")
            bo = stats["by_opt"].setdefault(opt, {"n": 0, "compiled": 0,
                                                 "passed": 0})
            bo["n"] += 1
            bo["compiled"] += compiled
            bo["passed"] += passed

            rec = {
                "id": sample.get("id"), "source_id": sample.get("source_id"),
                "opt": opt,
                "gen_seconds": round(gen_sec, 1),
                "answer_chars": len(answer),
                "hit_max_tokens": hit_max,
                "output_kind": kind,
                "candidate_c": cand_c,
                "extracted": cand_c is not None,
                "cand_compiled": compiled,
                "passed": passed,
                "cases_passed": verdict["n_passed"],
                "cases_total": verdict["n_total"],
                "cases_skipped": verdict.get("n_skipped", 0),
                "error": verdict.get("error"),
                "failing_cases": [c["name"] for c in verdict["cases"]
                                  if c["match"] is False],
            }
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pred_f.flush()
            print(f"[{i}/{len(rows)}] {rec['id']}: "
                  f"{'PASS' if passed else ('compiled' if compiled else 'FAIL')} "
                  f"({gen_sec:.1f}s)")

    elapsed = time.time() - started
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "val": os.path.abspath(args.val),
        "n": stats["n"],
        "compiled": stats["compiled"],
        "passed": stats["passed"],
        "ref_errors": stats["ref_errors"],
        "truncated": stats["truncated"],
        "output_kinds": stats["kinds"],
        "compile_rate": round(stats["compiled"] / stats["n"], 4),
        "pass_rate": round(stats["passed"] / stats["n"], 4),
        "by_opt": {
            opt: {
                "n": v["n"],
                "compile_rate": round(v["compiled"] / v["n"], 4),
                "pass_rate": round(v["passed"] / v["n"], 4),
            } for opt, v in stats["by_opt"].items()
        },
        "decoding": {"do_sample": False,
                     "max_new_tokens": args.max_new_tokens},
        "cases": args.cases or "default",
        "elapsed_seconds": round(elapsed, 1),
        "ts": now_iso(),
    }
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nпредсказания: {pred_path}\nсводка:      {sum_path}")


if __name__ == "__main__":
    main()
