#!/usr/bin/env python3
"""
gen_cases.py — per-program differential probes.

Why this exists
---------------
difftest.py runs both programs on a fixed list of 10 inputs for every task.
That list contains inputs no task can react to: for three programs in the
118-pair validation split the reference prints nothing and exits 0 on all ten,
so an empty `int main(void){ return 0; }` "passes" them. Those pairs measure
nothing and inflate pass_rate.

This script fixes that at the source: for each reference program it runs a
pool of candidate inputs, keeps only the inputs the reference actually reacts
to (non-empty stdout, clean exit, no timeout), and drops inputs that produce
the same output twice. Whatever is left is written as a per-program case list
that eval.py can replay.

The selection is deterministic: same program, same cases.

Resume
------
Programs are appended to the output file one by one, flushed after each line
and fsynced every 25 programs. A rerun with the same --out skips the programs
already in the file, so a killed run costs only the program it was working on.

Usage
-----
  python3 gen_cases.py --split val
  python3 gen_cases.py --split all --out dataset/cases.jsonl
  python3 gen_cases.py --split val --dry-run 5

Output record (one JSON line per source program):
  {"source_id": 46, "informative": true, "n_probes": 4,
   "cases": [{"name": "n + array", "stdin": "5\\n3 1 4 1 5\\n"}, ...]}

A program with `"informative": false` has no input the reference reacts to;
eval.py excludes such pairs from pass_rate and reports them separately.
"""

import argparse
import json
import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import difftest
import verifier

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEFAULT_OUT = os.path.join(DATASET_DIR, "cases.jsonl")

# Inputs the model should learn to handle: every common stdin shape of a
# small algorithmic program. Each one is tried on every reference program;
# only the ones the reference actually reacts to survive.
PROBE_POOL = [
    ("empty", ""),
    ("one int", "0\n"),
    ("one int 1", "1\n"),
    ("one int 2", "2\n"),
    ("one int 5", "5\n"),
    ("one int 7", "7\n"),
    ("one int 10", "10\n"),
    ("one int 42", "42\n"),
    ("one int 97", "97\n"),
    ("one int 255", "255\n"),
    ("one int 1000", "1000\n"),
    ("one int 123456", "123456\n"),
    ("negative", "-7\n"),
    ("two ints", "12 18\n"),
    ("two ints small", "3 7\n"),
    ("two ints wide", "12345 -678\n"),
    ("float", "3.14159\n"),
    ("two floats", "1.5 2.5\n"),
    ("n + array", "5\n3 1 4 1 5\n"),
    ("n + sorted", "6\n1 2 3 4 5 6\n"),
    ("n + reversed", "5\n5 4 3 2 1\n"),
    ("n + repeats", "4\n2 2 2 2\n"),
    ("n + ten", "10\n1 2 3 4 5 6 7 8 9 10\n"),
    ("n + negatives", "8\n-1 -2 3 4 -5 6 7 8\n"),
    ("n + pairs", "6\n1 1 2 2 3 3\n"),
    ("n + zeros", "3\n0 0 0\n"),
    ("n + binary", "5\n1 0 1 1 0\n"),
    ("n + extra", "3\n1 2 3 4\n"),
    ("n + array + k", "5\n3 1 4 1 5\n2\n"),
    ("n + array + k big", "7\n9 8 7 6 5 4 3\n3\n"),
    ("matrix", "3 3\n1 2 3\n4 5 6\n7 8 9\n"),
    ("matrix small", "2 2\n1 2\n3 4\n"),
    ("graph edges", "4\n4\n0 1\n1 2\n2 3\n3 0\n"),
    ("text", "hello world\n"),
    ("text long", "the quick brown fox jumps over the lazy dog\n"),
    ("text one word", "hello\n"),
    ("palindrome", "racecar\n"),
    ("two lines", "abc\nxyz\n"),
    ("three lines", "a\nb\nc\n"),
    ("two words", "cat dog\n"),
    ("n + words", "3\ncat dog bird\n"),
    ("n + lines", "3\n10\n20\n30\n"),
    ("count + text", "2\nhello world\nbye\n"),
    ("single char", "x\n"),
    ("digits line", "12345 67890\n"),
    ("blank lines", "\n\n\n"),
]

MAX_PROBES = 6          # kept per program
MAX_OUT_BYTES = 4096    # longer output means the input drove the program wild
MIN_OUT_CHARS = 1       # non-empty stdout: the only thing a no-op cannot fake
SYNC_EVERY = 25         # programs between fsync calls on the output file


def load_done(path):
    """Programs already processed, read back from the output file.

    A line cut in half by a kill is dropped and the file is rewritten without
    it: appending after a broken line would glue two JSON objects together."""
    done = {}
    if not os.path.exists(path):
        return done
    bad = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad = True
                continue
            done[row["source_id"]] = row
    if bad:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in done.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"dropped a broken last line from {path}", flush=True)
    return done


def load_references(split):
    """source_id -> reference C text, from the built dataset (one row per program)."""
    files = {"val": ["val.jsonl"], "train": ["train.jsonl"],
             "all": ["train.jsonl", "val.jsonl"]}[split]
    refs = {}
    for name in files:
        path = os.path.join(DATASET_DIR, name)
        if not os.path.exists(path):
            raise SystemExit(f"no {path}: run build_dataset.py first")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sid = row.get("source_id")
                if sid in refs:
                    continue
                refs[sid] = reference_c(row)
    return {k: v for k, v in refs.items() if v}


def reference_c(sample):
    """The C answer of the pair: the assistant message of the val/train row."""
    import re
    text = sample["messages"][-1]["content"]
    m = re.search(r"```(?:c)?[^\n]*\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def pick_cases(exe, workdir, max_probes):
    """Runs the probe pool against one program; returns the informative inputs."""
    kept = []
    seen_outputs = set()
    for name, stdin in PROBE_POOL:
        r = verifier.run_once(exe, workdir, stdin.encode())
        if r["timed_out"] or r["exec_error"]:
            continue
        if r["exit"] != 0:
            continue
        out = verifier._norm(r["stdout"])
        if len(out) < MIN_OUT_CHARS or len(out) > MAX_OUT_BYTES:
            continue
        if out in seen_outputs:
            continue
        seen_outputs.add(out)
        kept.append((len(out), name, stdin))
    # Prefer the inputs that produce the most output: a ten-element array
    # exercises more of an algorithm than a single digit does.
    kept.sort(key=lambda t: (-t[0], t[1]))
    return [{"name": n, "stdin": s} for _, n, s in kept[:max_probes]]


def main():
    ap = argparse.ArgumentParser(
        description="Build per-program differential probes")
    ap.add_argument("--split", choices=("val", "train", "all"), default="all")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-probes", type=int, default=MAX_PROBES)
    ap.add_argument("--dry-run", type=int, default=0,
                    help="process only the first N programs, write nothing")
    args = ap.parse_args()

    refs = load_references(args.split)
    done = load_done(args.out)
    todo = [sid for sid in sorted(refs) if sid not in done]
    print(f"programs: {len(refs)}, probe pool: {len(PROBE_POOL)}, "
          f"keep up to {args.max_probes}")
    print(f"resume: {len(done)} programs already in {args.out}, "
          f"{len(todo)} to go", flush=True)

    out_f = None
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # Append: a killed run keeps every program it finished. The file is
        # flushed after each line and fsynced every SYNC_EVERY programs.
        out_f = open(args.out, "a", encoding="utf-8")

    n_done = 0
    n_informative = 0
    probe_hist = {}
    try:
        with tempfile.TemporaryDirectory(prefix="gen_cases_") as workdir:
            for sid in todo:
                ok, exe, err = difftest.compile_c(refs[sid], workdir, "ref")
                if not ok:
                    print(f"  source {sid}: reference does not compile, skipped",
                          flush=True)
                    row = {"source_id": sid, "informative": False,
                           "n_probes": 0, "cases": []}
                else:
                    cases = pick_cases(exe, workdir, args.max_probes)
                    os.remove(exe)
                    row = {"source_id": sid, "informative": bool(cases),
                           "n_probes": len(cases), "cases": cases}
                informative = row["informative"]
                n_informative += informative
                probe_hist[len(row["cases"])] = (
                    probe_hist.get(len(row["cases"]), 0) + 1)
                if out_f is not None:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if (n_done + 1) % SYNC_EVERY == 0:
                        os.fsync(out_f.fileno())
                n_done += 1
                if n_done % 100 == 0:
                    print(f"  {n_done}/{len(todo)} new programs "
                          f"({len(done) + n_done}/{len(refs)} total)",
                          flush=True)
                if args.dry_run and n_done >= args.dry_run:
                    break
    finally:
        if out_f is not None:
            out_f.flush()
            os.fsync(out_f.fileno())
            out_f.close()

    total = len(done) + n_done
    print(f"\nthis run: processed {n_done}, informative {n_informative} "
          f"({n_informative / max(1, n_done) * 100:.1f}%)")
    print("probes per program (this run):", dict(sorted(probe_hist.items())))
    print(f"total in {args.out}: {total}/{len(refs)} programs")

    if args.dry_run:
        print("dry run: nothing written")
        return
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
