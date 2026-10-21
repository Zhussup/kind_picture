#!/usr/bin/env python3
"""
build_dataset.py — turns the accumulated corpus (dset/raw/*.jsonl)
into a ready dataset for fine-tuning: dset/dataset.jsonl.

What it does:
  - reads all shards in order;
  - skips broken lines (just in case, although the checkpoint repairs them anyway);
  - removes duplicates by the "id" field;
  - puts everything into one dataset.jsonl file in chat format:
        {"messages": [{"role":"system","content":...},
                      {"role":"user","content":...},
                      {"role":"assistant","content":...}], ...}
    — this is the standard format understood by fine-tuning tools (axolotl,
      unsloth, llama-factory, etc.), and plain text is easy to derive from it.

Run:  python3 build_dataset.py
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
    parser = argparse.ArgumentParser(description="Builds dataset.jsonl from raw/")
    parser.add_argument("--include-invalid", action="store_true",
                        help="include examples whose code failed syntax validation")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(RAW_DIR, "shard_*.jsonl")))
    if not shards:
        print("-- raw/ is empty, run collector.py first")
        sys.exit(1)

    seen = set()           # ids of records already written (duplicate protection)
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
                    # code_valid False == broken python code; older records lack the field (None) — leave them alone
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
                print(f"{os.path.basename(path)}: skipped broken lines {cat_bad}, "
                      f"duplicates {cat_dup}, invalid code {cat_invalid}")
            bad += cat_bad
            dup += cat_dup
            invalid += cat_invalid
    finally:
        out.close()

    print(f"Done: dataset.jsonl, unique examples: {total}, "
          f"broken lines: {bad}, duplicates: {dup}, syntax errors dropped: {invalid}")


if __name__ == "__main__":
    main()