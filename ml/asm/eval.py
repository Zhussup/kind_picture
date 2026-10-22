#!/usr/bin/env python3
"""
eval.py — evaluates the model on the asm -> C val set.

What it does:
  1. reads dataset/val.jsonl (pairs "listing -> C" that were not in training);
  2. for each pair builds a prompt (system + user, as in the dataset) and asks
     the model to reconstruct the C; generation is GREEDY (do_sample=False) — the answer
     is deterministic, so comparison does not depend on randomness;
  3. extracts the ```c block from the answer (the same extractor as in gen_c.py);
  4. compiles the extracted C and runs difftest against the reference C;
  5. computes metrics:
       compile_rate           — fraction of answers that compiled at all;
       pass_rate              — fraction of pairs whose behavior matched on all inputs;
       pass_rate_informative  — the same, over the pairs whose probes can tell a
                                correct program from a do-nothing one (only with
                                --cases-by-source).

The reference C is taken from the assistant message of the val pair, and the observable
behavior comes from running that C. The model's answer is compared to it by BEHAVIOR, not
by text: different C with the same behavior is a correct answer.

"Before/after training" comparison: the same val, the same inputs, the same
generation parameters; only --model/--adapter differs.

Resume
------
Results are appended to
  eval_results/predictions_<tag>__<val>__<probes>.jsonl
one line per pair, flushed and fsynced, so a kill loses at most the pair being
generated. A rerun with the same arguments skips the pairs already in that
file and continues the old ones. The summary is rebuilt from the file every
ten pairs, so it is correct even after an interruption. The file name contains
a hash of the probe set: changing --cases/--cases-by-source starts a new file
instead of mixing two different measurements.

Usage:
  python3 eval.py --model Qwen/Qwen3-0.6B --limit 5
  python3 eval.py --model Qwen/Qwen3-0.6B --adapter checkpoints/lora --tag lora
  python3 eval.py --model Qwen/Qwen3-0.6B --cases my_inputs.json --out-dir results
"""

import argparse
import hashlib
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
    """Extracts the C from the assistant message of the val pair (there it is inside a ```c block)."""
    text = sample["messages"][-1]["content"]
    m = C_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def build_prompt(tokenizer, sample):
    """System + user from the dataset -> a prompt string in the model's chat template.

    enable_thinking=False for Qwen3: otherwise the model first "thinks" and spends
    hundreds of tokens on reasoning that must not appear in our answer.
    If the model's template does not know this argument, we fall back to a plain
    call (TypeError)."""
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
        do_sample=False,                     # greedy generation: determinism
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    # did we hit the limit: then the answer is truncated and a "failure" may be an artifact
    hit_max = int(new_ids.shape[0]) >= max_new_tokens
    return text, hit_max


ASM_LINE_RE = re.compile(
    r"^\s*(\.(text|section|globl|type|size|LC\d+|string|data|bss)\b"
    r"|push[q|l]?\s|pop[q|l]?\s|mov[q|l|b|w]?\s|lea[q|l]?\s|call\s|ret\b"
    r"|j(mp|e|ne|s|ns|g|ge|l|le|z|nz)\s|add[q|l]?\s|sub[q|l]?\s|"
    r"cmp[q|l|b|w]?\s|test[q|l|b|w]?\s|xor[q|l]?\s|imul[q|l]?\s|"
    r"[A-Za-z_.][\w.]*:)")


def output_kind(text):
    """Rough classification of the answer: 'c', 'asm' (the model copied the listing
    into a ```c block) or 'other'. Needed for diagnostics, not for the metric."""
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
        # transformers >= 5: the parameter is called dtype
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


def file_digest(path):
    """First 8 hex chars of sha1 over the whole file.

    Used as the identity of a probe set: resuming a run with different probes
    would silently mix two different measurements in one file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_done(path, inputs_key):
    """Reads the predictions already on disk.

    Returns (records by id, whether the file had a broken line). A broken line
    is what a kill in the middle of a write leaves behind; such a line is
    dropped, and the caller rewrites the file without it, so the next append
    does not glue two JSON objects into one line."""
    done = {}
    bad = False
    if not os.path.exists(path):
        return done, bad
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad = True
                continue
            if rec.get("inputs_key") != inputs_key:
                raise SystemExit(
                    f"{path} was written with different differential inputs "
                    f"({rec.get('inputs_key')} != {inputs_key}); "
                    f"use another --tag")
            done[rec["id"]] = rec
    return done, bad


def rewrite_clean(path, records):
    """Atomically rewrites the predictions file from the records we trust."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def aggregate(records):
    """Counts for every metric, from a plain list of prediction records.

    Recomputing from the file instead of keeping counters in memory is what
    makes a resumed run add up: the numbers cover the whole file, not just
    the pairs generated in this session."""
    stats = {"n": 0, "compiled": 0, "passed": 0, "ref_errors": 0,
             "truncated": 0, "uninformative": 0, "kinds": {}, "by_opt": {}}
    for rec in records:
        compiled = bool(rec.get("cand_compiled"))
        passed = bool(rec.get("passed"))
        informative = bool(rec.get("informative", True))
        stats["n"] += 1
        stats["compiled"] += compiled
        stats["passed"] += passed
        stats["ref_errors"] += int(rec.get("ref_compiled") is False)
        stats["truncated"] += int(bool(rec.get("hit_max_tokens")))
        stats["uninformative"] += int(not informative)
        kind = rec.get("output_kind", "other")
        stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
        bo = stats["by_opt"].setdefault(
            rec.get("opt", "?"),
            {"n": 0, "compiled": 0, "passed": 0, "uninformative": 0})
        bo["n"] += 1
        bo["compiled"] += compiled
        bo["passed"] += passed
        bo["uninformative"] += int(not informative)
    return stats


def build_summary(records, args, n_pairs, inputs_key, elapsed, n_new,
                  n_done_before):
    stats = aggregate(records)
    n = stats["n"]
    n_inf = n - stats["uninformative"]

    def rate(num, den):
        return round(num / den, 4) if den else 0.0

    return {
        "model": args.model,
        "adapter": args.adapter,
        "val": os.path.abspath(args.val),
        "n_pairs": n_pairs,
        "n": n,
        "compiled": stats["compiled"],
        "passed": stats["passed"],
        "ref_errors": stats["ref_errors"],
        "truncated": stats["truncated"],
        "uninformative": stats["uninformative"],
        "n_informative": n_inf,
        "output_kinds": stats["kinds"],
        "compile_rate": rate(stats["compiled"], n),
        "pass_rate": rate(stats["passed"], n),
        # pass_rate over the pairs whose probes can actually tell a correct
        # program from a do-nothing one
        "pass_rate_informative": rate(stats["passed"], n_inf),
        "by_opt": {
            opt: {
                "n": v["n"],
                "compile_rate": rate(v["compiled"], v["n"]),
                "pass_rate": rate(v["passed"], v["n"]),
                "pass_rate_informative":
                    rate(v["passed"], v["n"] - v["uninformative"]),
            } for opt, v in stats["by_opt"].items()
        },
        "decoding": {"do_sample": False,
                     "max_new_tokens": args.max_new_tokens},
        "cases": args.cases or (
            os.path.abspath(args.cases_by_source) if args.cases_by_source
            else "default"),
        "inputs_key": inputs_key,
        "resumed": n_done_before,
        "new_this_run": n_new,
        "elapsed_seconds": round(elapsed, 1),
        "ts": now_iso(),
    }


def write_summary(path, summary):
    """Atomic: a half-written summary must never be readable as a whole one."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Evaluate asm->C on the val set")
    ap.add_argument("--model", required=True,
                    help="HF id or path to weights (Qwen/Qwen3-0.6B)")
    ap.add_argument("--adapter", help="path to a LoRA adapter (for a fine-tuned model)")
    ap.add_argument("--val", default=VAL_PATH, help="val.jsonl")
    ap.add_argument("--cases", help="JSON with inputs for difftest")
    ap.add_argument("--cases-by-source",
                    help="JSONL from gen_cases.py: per-program probes keyed by "
                         "source_id; pairs without probes are counted as "
                         "uninformative and excluded from pass_rate")
    ap.add_argument("--limit", type=int, default=0, help="take the first N pairs")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="", help="run label (used in file names)")
    args = ap.parse_args()

    if not os.path.exists(args.val):
        print(f"no file {args.val}: run build_dataset.py first",
              file=sys.stderr)
        sys.exit(3)

    inputs = difftest.DEFAULT_INPUTS
    if args.cases:
        try:
            with open(args.cases, "r", encoding="utf-8") as f:
                inputs = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"error in cases: {e}", file=sys.stderr)
            sys.exit(3)

    by_source = None
    if args.cases_by_source:
        by_source = {}
        try:
            with open(args.cases_by_source, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    by_source[row["source_id"]] = row.get("cases") or []
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"error in cases-by-source: {e}", file=sys.stderr)
            sys.exit(3)

    # Identity of the input set. Part of the output file name, so a run with
    # different probes lands in a different file instead of mixing with this one.
    if args.cases_by_source:
        inputs_key = "cases" + file_digest(args.cases_by_source)
    elif args.cases:
        inputs_key = "cases" + file_digest(args.cases)
    else:
        inputs_key = "default"

    rows = load_rows(args.val, args.limit)
    if not rows:
        print("val is empty", file=sys.stderr)
        sys.exit(3)

    # File names are derived only from stable inputs: model, adapter, tag, the
    # val file and the probe set. No timestamp — a rerun must find the same
    # file to continue it instead of starting a second one.
    stem = slug(args.tag) if args.tag else slug("_".join(
        x for x in (os.path.basename(args.model.rstrip("/")),
                    os.path.basename(args.adapter.rstrip("/"))
                    if args.adapter else "base") if x))
    val_stem = slug(os.path.splitext(os.path.basename(args.val))[0])
    base = f"{stem}__{val_stem}__{inputs_key}"
    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, f"predictions_{base}.jsonl")
    sum_path = os.path.join(args.out_dir, f"summary_{base}.json")

    done, bad_tail = load_done(pred_path, inputs_key)
    if bad_tail:
        rewrite_clean(pred_path, done.values())
        print(f"dropped a broken last line from {os.path.basename(pred_path)}")
    todo = [r for r in rows if r.get("id") not in done]
    n_done_before = len(rows) - len(todo)

    print(f"model: {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    print(f"val: {len(rows)} pairs, difftest inputs: "
          f"{'per program' if by_source is not None else len(inputs)}")
    if by_source is not None:
        n_with = sum(1 for r in rows if by_source.get(r.get("source_id")))
        print(f"cases-by-source: {len(by_source)} programs, "
              f"{n_with}/{len(rows)} pairs have probes")
    print(f"resume: {n_done_before} pairs already in "
          f"{os.path.basename(pred_path)}, {len(todo)} to go")

    if not todo:
        # Nothing to generate: do not spend 20 s loading the model.
        summary = build_summary(list(done.values()), args, len(rows),
                                inputs_key, 0.0, 0, n_done_before)
        write_summary(sum_path, summary)
        print("everything is already evaluated")
        print(f"\npredictions: {pred_path}\nsummary:     {sum_path}")
        return

    model, tokenizer = load_model(args.model, args.adapter)

    started = time.time()
    records = list(done.values())      # everything on disk, old and new
    n_new = 0
    # Append: a kill loses at most the pair being generated, never the file.
    with open(pred_path, "a", encoding="utf-8") as pred_f:
        for i, sample in enumerate(todo, 1):
            ref_c = reference_c(sample)
            prompt = build_prompt(tokenizer, sample)
            t0 = time.time()
            answer, hit_max = generate(model, tokenizer, prompt,
                                       args.max_new_tokens)
            gen_sec = time.time() - t0
            cand_c = extract_c(answer)
            kind = output_kind(answer)

            # With per-program probes, a pair whose reference reacts to no
            # input at all measures nothing: exclude it from pass_rate.
            if by_source is not None:
                sample_inputs = by_source.get(sample.get("source_id")) or []
            else:
                sample_inputs = inputs
            informative = bool(sample_inputs)

            if cand_c is None:
                verdict = {"ref_compiled": True, "cand_compiled": False,
                           "stage": "cand_compile",
                           "n_total": len(sample_inputs),
                           "n_passed": 0, "all_match": False,
                           "error": "the response has no ```c block", "cases": []}
            else:
                verdict = difftest.compare(ref_c, cand_c, sample_inputs)

            compiled = bool(verdict["cand_compiled"])
            passed = bool(verdict["all_match"])
            opt = sample.get("opt", "?")

            rec = {
                "id": sample.get("id"), "source_id": sample.get("source_id"),
                "opt": opt,
                "gen_seconds": round(gen_sec, 1),
                "answer_chars": len(answer),
                "hit_max_tokens": hit_max,
                "output_kind": kind,
                "candidate_c": cand_c,
                "extracted": cand_c is not None,
                "informative": informative,
                "ref_compiled": bool(verdict["ref_compiled"]),
                "cand_compiled": compiled,
                "passed": passed,
                "cases_passed": verdict["n_passed"],
                "cases_total": verdict["n_total"],
                "cases_skipped": verdict.get("n_skipped", 0),
                "inputs_key": inputs_key,
                "error": verdict.get("error"),
                "failing_cases": [c["name"] for c in verdict["cases"]
                                  if c["match"] is False],
            }
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pred_f.flush()
            os.fsync(pred_f.fileno())      # survives a kill or a power cut
            records.append(rec)
            n_new += 1

            print(f"[{n_done_before + i}/{len(rows)}] {rec['id']}: "
                  f"{'PASS' if passed else ('compiled' if compiled else 'FAIL')} "
                  f"({gen_sec:.1f}s)")

            # Keep the summary current, so an interrupted run still has numbers
            # for the pairs it did finish.
            if n_new % 10 == 0:
                write_summary(sum_path, build_summary(
                    records, args, len(rows), inputs_key,
                    time.time() - started, n_new, n_done_before))

    summary = build_summary(records, args, len(rows), inputs_key,
                            time.time() - started, n_new, n_done_before)
    write_summary(sum_path, summary)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\npredictions: {pred_path}\nsummary:     {sum_path}")


if __name__ == "__main__":
    main()
