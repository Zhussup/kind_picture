#!/usr/bin/env python3
"""
build_dataset.py — собирает датасет линии A1 для направления asm -> C.

Вход:  ml/asm/raw_c/shard_*.jsonl (то, что пишет gen_c.py).
Выход: ml/asm/dataset/train.jsonl + val.jsonl в chat-формате:
       system  — задача (реконструируй C по листингу gcc);
       user    — asm-листинг, очищенный от утечек;
       assistant — исходный C в блоке ```c.

Важно про утечку исходника:
  gen_c.py компилирует с -fverbose-asm, поэтому в листинге есть строки
  "# prog.c:7: for (...)": это дословные строки исходника. Если оставить
  их, задача asm->C превращается в копирование. Поэтому здесь такие
  комментарии вырезаются (SRC_LINE_RE). Прочие комментарии gcc
  (например "#, f" — имена операндов) остаются: это честная часть asm.

Сплит train/val делается по id ИСХОДНОЙ записи, а не по sample: два
листинга одной программы (-O0 и -O2) обязаны попасть в один и тот же
сплит, иначе валидация видит тот же ответ, что и обучение.

Запуск:
  python3 build_dataset.py                 собрать всё, что есть
  python3 build_dataset.py --val-ratio 0.02 --min-asm-lines 8
  python3 build_dataset.py --dry-run       только статистика, без записи
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_c")
OUT_DIR = os.path.join(BASE_DIR, "dataset")

SYSTEM_PROMPT = (
    "You are an expert reverse engineer. You are given an x86-64 assembly "
    "listing in AT&T syntax produced by gcc (C11) from a single-file C "
    "program. Reconstruct a C program with the same observable behavior: it "
    "reads from stdin and writes its result to stdout. "
    "Reply with exactly one fenced code block that starts with ```c and "
    "nothing else."
)

# "# prog.c:7:     for (int i = 2; i <= n; ++i) f *= i;" — утечка исходника
SRC_LINE_RE = re.compile(r"^\s*#.*?prog\.c:\d+:")


def strip_source_comments(asm_text):
    """Убирает из листинга комментарии-эхо исходных строк (см. docstring)."""
    return "\n".join(
        ln for ln in asm_text.splitlines() if not SRC_LINE_RE.match(ln)
    )


def user_message(asm_text, opt):
    return (
        f"Optimization level: -{opt}\n\n"
        "Assembly listing:\n"
        f"```asm\n{asm_text}\n```"
    )


def assistant_message(c_text):
    return f"```c\n{c_text}\n```"


def iter_records():
    """Читает все шарды по порядку. Битую/недописанную строку пропускает:
    сборщик может писать в последний шард прямо во время чтения."""
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "shard_*.jsonl")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, lineno, json.loads(line)
                except json.JSONDecodeError:
                    yield path, lineno, None


def is_val(source_id, val_ratio):
    """Детерминированный сплит по id исходной записи: перезапуск даёт тот же
    результат, и обе оптимизации одной программы попадают в один сплит."""
    h = int(hashlib.sha1(str(source_id).encode()).hexdigest()[:8], 16)
    return h % 10_000 < int(val_ratio * 10_000)


def build(val_ratio, min_asm_lines):
    stats = {
        "records": 0, "records_broken": 0, "records_no_c": 0,
        "samples": 0, "skipped_dup": 0, "skipped_short": 0,
        "skipped_not_compiled": 0,
        "train": 0, "val": 0,
        "by_opt": {"O0": 0, "O2": 0},
    }
    seen_asm = set()
    train, val = [], []

    for path, lineno, rec in iter_records():
        if rec is None:
            stats["records_broken"] += 1
            continue
        stats["records"] += 1
        if not rec.get("extract_ok") or not rec.get("c_source"):
            stats["records_no_c"] += 1
            continue
        c_text = rec["c_source"].strip()
        source_id = rec.get("id")
        for opt in ("O0", "O2"):
            if not rec.get(f"compile_{opt.lower()}_ok"):
                stats["skipped_not_compiled"] += 1
                continue
            asm = strip_source_comments(rec.get(f"asm_{opt.lower()}") or "")
            if len(asm.splitlines()) < min_asm_lines:
                stats["skipped_short"] += 1
                continue
            key = hashlib.sha1(asm.encode("utf-8")).hexdigest()
            if key in seen_asm:
                stats["skipped_dup"] += 1
                continue
            seen_asm.add(key)
            sample = {
                "id": f"a1-{source_id}-{opt}",
                "source_id": source_id,
                "opt": opt,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(asm, opt)},
                    {"role": "assistant", "content": assistant_message(c_text)},
                ],
            }
            if is_val(source_id, val_ratio):
                val.append(sample)
                stats["val"] += 1
            else:
                train.append(sample)
                stats["train"] += 1
            stats["by_opt"][opt] += 1
            stats["samples"] += 1
    return train, val, stats


def write_jsonl(path, rows):
    """Атомарная запись: сначала .tmp, потом os.replace. Если обучение читает
    файл в момент пересборки, оно увидит либо старую версию, либо новую."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description="Собрать датасет asm->C из ml/asm/raw_c/")
    ap.add_argument("--val-ratio", type=float, default=0.02,
                    help="доля валидации (по id исходной записи), по умолчанию 0.02")
    ap.add_argument("--min-asm-lines", type=int, default=8,
                    help="пропускать листинги короче N строк")
    ap.add_argument("--dry-run", action="store_true",
                    help="только статистика, ничего не писать")
    args = ap.parse_args()

    train, val, stats = build(args.val_ratio, args.min_asm_lines)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not stats["samples"]:
        print("пока нечего собирать: в raw_c/ нет ни одной пригодной пары")
        return

    if args.dry_run:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"записано: {OUT_DIR}/train.jsonl ({stats['train']}), "
          f"val.jsonl ({stats['val']})")


if __name__ == "__main__":
    main()
