#!/usr/bin/env python3
"""
build_dataset.py — builds the A1 dataset for the asm -> C direction.

Input:  ml/asm/raw_c/shard_*.jsonl (what gen_c.py writes).
Output: ml/asm/dataset/train.jsonl + val.jsonl in chat format:
        system  — the task (reconstruct C from the gcc listing);
        user    — the asm listing, cleaned of leaks;
        assistant — the original C inside a ```c block.

Important about source leakage:
  gen_c.py compiles with -fverbose-asm, so the listing contains lines like
  "# prog.c:7: for (...)": these are verbatim source lines. If they are kept,
  the asm->C task turns into copying. Therefore such comments are cut here
  (SRC_LINE_RE). Other gcc comments
  (for example "#, f" — operand names) are kept: they are an honest part of asm.

The train/val split is done by the id of the ORIGINAL record, not by sample: the two
listings of one program (-O0 and -O2) must land in the same split,
otherwise validation sees the same answer as training.

Usage:
  python3 build_dataset.py                 build everything that is available
  python3 build_dataset.py --val-ratio 0.02 --min-asm-lines 8
  python3 build_dataset.py --dry-run       statistics only, no writing
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

# "# prog.c:7:     for (int i = 2; i <= n; ++i) f *= i;" — source leak
SRC_LINE_RE = re.compile(r"^\s*#.*?prog\.c:\d+:")


def strip_source_comments(asm_text):
    """Removes source-echo comments from the listing (see the docstring)."""
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
    """Reads all shards in order. Skips broken/partial lines:
    the collector may be writing to the last shard while we are reading it."""
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
    """Deterministic split by the original record's id: a restart gives the same
    result, and both optimizations of one program land in the same split."""
    h = int(hashlib.sha1(str(source_id).encode()).hexdigest()[:8], 16)
    return h % 10_000 < int(val_ratio * 10_000)


def build(val_ratio, min_asm_lines):
    stats = {
        "records": 0, "records_broken": 0, "records_no_c": 0,
        "samples": 0, "skipped_dup": 0, "skipped_short": 0,
        "skipped_not_compiled": 0,
        "train": 0, "val": 0,
        "by_opt": {"O0": 0, "O2": 0},
        "by_model": {},
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
            # The teacher is part of the sample: two models can answer the
            # same task at very different lengths and in different styles,
            # and the run that trains on them must be able to balance or
            # filter by teacher without joining back to raw_c/.
            model = rec.get("model") or "unknown"
            stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
            sample = {
                "id": f"a1-{source_id}-{opt}",
                "source_id": source_id,
                "opt": opt,
                "model": model,
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
    """Atomic write: first .tmp, then os.replace. If training reads the
    file while it is being rebuilt, it sees either the old version or the new one."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description="Build the asm->C dataset from ml/asm/raw_c/")
    ap.add_argument("--val-ratio", type=float, default=0.02,
                    help="validation fraction (by original record id), default 0.02")
    ap.add_argument("--min-asm-lines", type=int, default=8,
                    help="skip listings shorter than N lines")
    ap.add_argument("--dry-run", action="store_true",
                    help="statistics only, write nothing")
    args = ap.parse_args()

    train, val, stats = build(args.val_ratio, args.min_asm_lines)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not stats["samples"]:
        print("nothing to build yet: raw_c/ has no usable pairs")
        return

    if args.dry_run:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"written: {OUT_DIR}/train.jsonl ({stats['train']}), "
          f"val.jsonl ({stats['val']})")


if __name__ == "__main__":
    main()
