#!/usr/bin/env python3
"""
build_dataset.py — превращает накопленный корпус (dset/raw/*.jsonl)
в готовый датасет для файнтюна: dset/dataset.jsonl.

Что делает:
  - читает все шарды по порядку;
  - пропускает битые строки (на всякий случай, хотя чекпоинт их и так чинит);
  - убирает дубликаты по полю "id";
  - складывает всё в один файл dataset.jsonl в формате chat:
        {"messages": [{"role":"system","content":...},
                      {"role":"user","content":...},
                      {"role":"assistant","content":...}], ...}
    — это стандартный формат, который понимают тулзы файнтюна (axolotl,
      unsloth, llama-factory и т.п.), и из него легко получить и plain-text.

Запуск:  python3 build_dataset.py
"""
import argparse
import glob
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUT_PATH = os.path.join(BASE_DIR, "dataset.jsonl")

SYSTEM_PROMPT = (
    "You are an expert programmer. Answer in English. "
    "Be practical: always show working, idiomatic code in fenced code blocks "
    "and explain it clearly."
)


def main():
    parser = argparse.ArgumentParser(description="Собирает dataset.jsonl из raw/")
    parser.add_argument("--include-invalid", action="store_true",
                        help="включить примеры с кодом, не прошедшим проверку синтаксиса")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(RAW_DIR, "shard_*.jsonl")))
    if not shards:
        print("-- raw/ пуст, сначала запусти collector.py")
        sys.exit(1)

    seen = set()           # id уже написанных записей (защита от дублей)
    total, bad, dup, invalid = 0, 0, 0, 0
    out = open(OUT_PATH, "w", encoding="utf-8")
    try:
        for path in shards:
            cat_bad = cat_dup = cat_invalid = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        cat_bad += 1
                        continue
                    # code_valid False == битый python-код; у старых записей поля нет (None) — их не трогаем
                    if rec.get("code_valid") is False and not args.include_invalid:
                        cat_invalid += 1
                        continue
                    sid = rec.get("id")
                    if sid in seen:
                        cat_dup += 1
                        continue
                    seen.add(sid)
                    row = {
                        "id": sid,
                        "category": rec.get("category"),
                        "lang": rec.get("lang"),
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": rec["prompt"]},
                            {"role": "assistant", "content": rec["response"]},
                        ],
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
            if cat_bad or cat_dup or cat_invalid:
                print(f"{os.path.basename(path)}: пропущено битых строк {cat_bad}, "
                      f"дублей {cat_dup}, невалидного кода {cat_invalid}")
            bad += cat_bad
            dup += cat_dup
            invalid += cat_invalid
    finally:
        out.close()

    print(f"Готово: dataset.jsonl, уникальных примеров: {total}, "
          f"битых строк: {bad}, дублей: {dup}, выкинуто синтакс-ошибок: {invalid}")


if __name__ == "__main__":
    main()