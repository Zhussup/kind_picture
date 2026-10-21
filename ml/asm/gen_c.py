#!/usr/bin/env python3
"""
gen_c.py — builds REAL "C → asm" pairs for the dataset (line A1).

Idea (and why this is more reliable than asking the model to write asm):
  beam writes a C program, and the gcc compiler turns it into asm itself.
  So the asm in the pair is guaranteed correct, a "gold" reference:
  it does not need to be checked with tests; a real compiler produced it.
  The model will later learn exactly this kind of asm (AT&T, x86-64).

What the script does, step by step:
  1. The sample number (0, 1, 2, ...) is deterministically turned into a task
     (task × wording variation; no randomness at all — after a restart
     we continue from exactly the same place).
  2. The prompt goes to the Ollama API (model beam:cloud), and the response is parsed:
     exactly one ```c ... ``` block is required.
  3. The code is compiled TWICE: `gcc -O0 -S` and `gcc -O2 -S` — we get
     two listings of the same program (important: you can see how the optimizer
     changes the code — that is a valuable part of the dataset).
  4. One JSONL line per sample: C source + both listings + flags
     "compiled/failed to compile". Written with flush+fsync (survives a power loss).
  5. The checkpoint state_gen_c.json is updated atomically.

Format A1 — these are "C source → compiler listing" pairs. Tests are
NOT run here: the only quality label is "does it compile". Pairs of format A2
("task → _start program + test run via verifier.py") are deliberately
not mixed in here — that is a separate line B.

Three service items are cut from the listing (for dataset readability):
  - `.file "prog.c"` — the source file name, which we don't need;
  - `.ident "GCC: ..."` — the compiler version;
  - `.section .note...` — empty service sections for the linker.
Everything else (including the .text/.data sections, labels, directives) is kept.
The `-fno-asynchronous-unwind-tables` flag removes the .cfi directive noise at
the compilation stage, and `-fverbose-asm` adds comments with variable names.

Usage:
  python3 gen_c.py --once        one sample (connectivity check)
  python3 gen_c.py --dry-run 5   show 5 prompts without the network
  python3 gen_c.py --status      show the checkpoint
  python3 gen_c.py --self-test   offline check of the gcc part (no network)
  python3 gen_c.py               endless collection (Ctrl+C to stop)
"""

import argparse
import datetime
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import requests

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ml/asm
RAW_DIR = os.path.join(BASE_DIR, "raw_c")               # shards are written here
STATE_PATH = os.path.join(BASE_DIR, "state_gen_c.json") # checkpoint
LOG_PATH = os.path.join(BASE_DIR, "gen_c.log")          # log

MODEL = "beam:cloud"
HOST = "http://localhost:11434"     # local Ollama daemon (it holds the login)

SYSTEM_PROMPT = (
    "You are an expert C programmer. Reply with exactly one fenced code block "
    "that starts with ```c and contains the complete program. "
    "No explanations, no prose outside the block."
)
TEMP_RANGE = (0.6, 0.95)   # random temperature per request — fewer repeats
JITTER = (0.0, 1.5)        # random addition to the pause (anti-hammering)
REQUEST_TIMEOUT = 900      # seconds for one model response
PAUSE = 2.0                # pause between requests, seconds
MAX_SHARD_BYTES = 8 * 1024 * 1024   # new shard after ~8 MB
MAX_BACKOFF = 3600.0       # maximum wait after a series of network errors
STATE_EVERY_SECONDS = 60   # how often to refresh the checkpoint "just in case"

GCC_TIMEOUT = 30           # seconds per gcc run (in practice: milliseconds)
MAX_EXTRACT_FAILS = 3      # try one prompt this many times, then move on

GCC_FLAGS = [
    "gcc", "-std=c11", "-Wall", "-Wextra", "-pedantic",
    "-fno-asynchronous-unwind-tables",   # no .cfi noise in the listing
    "-fverbose-asm",                     # comments with variable names
]

# ---------------------------------------------------------------------------
# TASK SPACE
# ---------------------------------------------------------------------------
# Each line continues the phrase "Write a complete C program that ...".
# The prompt is deterministic: index -> (task, wording variation).
# Once COMBO is exhausted the prompts repeat, but the answers will differ
# (the temperature is random). Tasks are easy to extend — just add a line.

TASKS = [
    # --- numbers ---
    "reads one integer n and prints n factorial (n <= 12) using a loop",
    "reads two integers a and b and prints their greatest common divisor via Euclid's algorithm",
    "reads one integer n and prints the n-th Fibonacci number using an iterative loop",
    "reads one integer n and prints 1 if n is prime, otherwise 0",
    "reads one integer n and prints all primes up to n using a sieve of Eratosthenes on a fixed array",
    "reads one integer n and prints the sum of its decimal digits",
    "reads one integer n and prints n with its decimal digits reversed",
    "reads one integer n and prints its binary representation without leading zeros",
    "reads a non-negative integer n and prints the number of set bits (popcount) computed manually",
    "reads three integers a, b, m and prints (a^b) mod m using fast modular exponentiation",
    "reads one integer n and prints the sum of all its divisors excluding itself",
    "reads one integer n and prints the Collatz sequence length from n down to 1",
    "reads one integer n and prints Euler's totient phi(n) by factorization",
    "reads two integers a and b and prints a^b for small b using a loop",
    "reads one integer n and prints whether it is a perfect square using only integer arithmetic",
    "reads one integer n and prints the largest power of two not exceeding n",
    # --- strings ---
    "reads one line and prints it reversed",
    "reads one line and prints 1 if it is a palindrome ignoring spaces, otherwise 0",
    "reads one line and prints the number of vowels (a, e, i, o, u, case-insensitive)",
    "reads one line and prints the same line with every lowercase letter shifted by one (a->b, ..., z->a)",
    "reads one line and prints the number of words separated by spaces",
    "reads one line and prints the longest word",
    "reads one line and prints the same line with all spaces removed in place using pointer arithmetic",
    "reads one line and prints its run-length encoding (for example aaabbc becomes a3b2c1)",
    "reads one line and prints 1 if all parentheses are balanced, otherwise 0",
    "reads input until EOF and prints the number of lines",
    "reads one line and prints the count of each lowercase letter a-z in 'a:3 b:1' style, skipping zero counts",
    "reads one line and prints its characters sorted in ascending ASCII order without using qsort",
    "reads one line of hexadecimal digits and prints its decimal value",
    "reads one integer in decimal and prints it in lowercase hexadecimal without the 0x prefix",
    "reads one line and converts its words to uppercase without toupper, using ASCII arithmetic",
    "reads one long line and prints 1 if it contains only digits, otherwise 0",
    # --- arrays ---
    "reads n then n integers, sorts them ascending with bubble sort, and prints them space-separated",
    "reads n then n integers, sorts them ascending with insertion sort, and prints them space-separated",
    "reads n then n integers, sorts them ascending with selection sort, and prints them space-separated",
    "reads n then n integers, sorts them ascending with a recursive quicksort, and prints them",
    "reads n then n integers, sorts them ascending with merge sort using a helper array, and prints them",
    "reads n then n integers and prints them using heap sort with an explicit sift-down function",
    "reads n, n sorted integers, and one target x, then prints the index of x via binary search or -1",
    "reads n then n integers and prints the maximum and the second maximum",
    "reads n then n integers and prints their prefix sums space-separated",
    "reads n then n integers and one k, and prints the array rotated right by k positions",
    "reads n then n sorted integers and prints the array with duplicates removed in place",
    "reads n and m, then an n x m matrix, and prints its transpose",
    "reads two 3 x 3 matrices and prints their product",
    "reads n then n integers and prints their median as an integer (lower median for even n)",
    "reads n then n integers and prints the number of inversions using O(n^2) counting",
    "reads n then n integers and prints the distinct values and their counts",
    "reads a 10 x 10 grid of '.' and '#' characters and prints the number of connected '#' regions (4-neighbourhood)",
    "reads n then n integers and prints the longest strictly increasing subsequence length (O(n^2) DP)",
    "reads n then n integers and prints the maximum subarray sum (Kadane's algorithm)",
    "reads n, a capacity C, then n item weights, and prints the maximum number of items that fit",
    # --- pointers, structs, memory ---
    "reads n then n integers into a malloc'd array that grows by doubling, prints the sum and frees everything",
    "defines struct Point { int x; int y; }, reads two points, and prints their squared distance",
    "defines struct Complex { double re; double im; }, reads two complex numbers, and prints their product",
    "builds a singly linked list from n integers read from stdin and prints it reversed using an explicit stack array (no recursion)",
    "builds a singly linked list from n integers, reverses the list in place by relinking pointers, and prints it",
    "reads n then n integers into a binary search tree and prints them in-order using an iterative traversal with an explicit stack",
    "implements a fixed-size int stack on an array of 100 elements driven by commands: p number (push), o (pop and print), q (quit)",
    "implements an int queue with a ring buffer of 100 elements driven by commands: e number (enqueue), d (dequeue and print), q (quit)",
    "reads one line and stores it in a singly linked list of characters, prints it back, and frees every node",
    # --- algorithms ---
    "reads an RPN expression of single-digit numbers and + - * / operators from one line and prints the integer result using an array stack",
    "reads a line of simple additions and subtractions of integers like '3+4-2' and prints the result without using strtol",
    "reads one line of lowercase words and prints them sorted with qsort and a custom comparator",
    "reads a string of roman numerals and prints its decimal value",
    "reads an integer 1..3999 and prints its Roman numeral representation",
    "reads one integer n and prints the multiplication table n x n, rows space-separated",
    "reads one integer n and prints a centred Pascal's triangle with n rows",
    "reads one integer n and prints an n x n spiral matrix filled with 1..n^2",
    "reads one integer n and prints the first n terms of the look-and-say sequence",
    "reads one line and prints the character with the maximum frequency and its count",
    "reads a command character then n integers, applies the matching function pointer (a=abs, s=square, n=negate), and prints the result",
]

VARIATIONS = [
    "Keep it short and direct, without comments.",
    "Add brief comments explaining the tricky parts.",
    "Use clear variable names and keep functions small.",
    "Handle the empty input case gracefully.",
    "Prefer straightforward loops over clever tricks.",
    "Write it for a beginner audience but fully correct.",
    "Include a few defensive checks on input sizes.",
    "Use fixed-size arrays where possible and avoid VLAs.",
    "Make the code self-contained in one file with a main function.",
    "Write it the way you would in a code review: idiomatic and tidy.",
]

COMBO = len(TASKS) * len(VARIATIONS)

DROP_PREFIXES = (".file", ".ident", ".section .note")

FENCE_C = re.compile(r"```[cC]\s*\n(.*?)```", re.S)
FENCE_ANY = re.compile(r"```[^\n]*\n(.*?)```", re.S)


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

def prompt_for(index):
    """index -> (task, variation, prompt text). Deterministic."""
    task_i = index % len(TASKS)
    var_i = (index // len(TASKS)) % len(VARIATIONS)
    task, variation = TASKS[task_i], VARIATIONS[var_i]
    prompt = (
        f"Write a complete C program (one file, C11) that {task}. "
        "Read everything from stdin and print the result to stdout. "
        f"{variation}"
    )
    return task, variation, prompt


def extract_c(response):
    """Extracts C code from the response: first look for a ```c block, then any ```.
    If there are several blocks, take the longest one. No blocks — None."""
    blocks = FENCE_C.findall(response)
    if not blocks:
        blocks = FENCE_ANY.findall(response)
    if not blocks:
        return None
    code = max(blocks, key=len).strip()
    return code or None


# ---------------------------------------------------------------------------
# gcc
# ---------------------------------------------------------------------------

def clean_listing(text):
    """Cuts service lines (see the module docstring) and the compiler header.

    The header consists of the leading comment lines `# GNU C11 ...`, `# options passed ...`:
    that is metadata, not asm; we record the gcc version separately in the "gcc" field.
    The `# prog.c:6: ...` comments INSIDE the code are kept — they are a useful
    link from each instruction to a source line."""
    kept = []
    for ln in text.splitlines():
        # compare ignoring indentation: gcc writes "\t.section\t.note..."
        norm = ln.lstrip().replace("\t", " ")
        if norm.startswith(DROP_PREFIXES):
            continue
        kept.append(ln)
    while kept and (not kept[0].strip() or kept[0].startswith("#")):
        kept.pop(0)
    return "\n".join(kept)


def compile_c(c_text, opt):
    """Compiles C text into an asm listing at optimization level opt ('O0'/'O2').

    Returns a dict: ok (bool), asm (str|None), diag (stderr text, truncated).
    The diagnostics are not used for rejection but as data: you can see that the
    model wrote code with warnings — that is also a quality signal.
    """
    with tempfile.TemporaryDirectory(prefix="genc_") as workdir:
        src = os.path.join(workdir, "prog.c")
        out = os.path.join(workdir, "prog.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(c_text)
        # pass relative names with cwd=workdir: otherwise gcc writes the
        # absolute path /tmp/genc_XXXX/prog.c into comments — random clutter
        cmd = GCC_FLAGS + [f"-{opt}", "-S", "prog.c", "-o", "prog.s"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=GCC_TIMEOUT, cwd=workdir)
        except subprocess.TimeoutExpired:
            return {"ok": False, "asm": None,
                    "diag": f"gcc did not finish within {GCC_TIMEOUT} s"}
        except OSError as e:
            return {"ok": False, "asm": None,
                    "diag": f"{type(e).__name__}: {e}"}
        if p.returncode != 0:
            return {"ok": False, "asm": None,
                    "diag": (p.stderr or p.stdout).strip()[:4000]}
        try:
            with open(out, "r", encoding="utf-8") as f:
                listing = clean_listing(f.read())
        except OSError as e:
            return {"ok": False, "asm": None,
                    "diag": f"could not read the listing: {e}"}
        return {"ok": True, "asm": listing, "diag": p.stderr.strip()[:4000]}


# ---------------------------------------------------------------------------
# CHECKPOINT AND FILES
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def gcc_version():
    """gcc version — for dataset provenance (the listing header is now removed)."""
    try:
        p = subprocess.run(["gcc", "-dumpfullversion"], capture_output=True,
                           text=True, timeout=5)
        return p.stdout.strip() or "?"
    except OSError:
        return "?"


GCC_VERSION = gcc_version()


def default_state():
    return {
        "version": 1,
        "prompt_index": 0,       # number of the next sample
        "shard_num": 1,          # current file raw_c/shard_NNNNN.jsonl
        "samples_ok": 0,
        "samples_no_c": 0,       # response without a C block (after MAX_EXTRACT_FAILS)
        "samples_bad_c": 0,      # C is present, but gcc compiled neither -O0 nor -O2
        "extract_fails": 0,      # failed attempts for the current index
        "consecutive_errors": 0,
        "min_backoff": None,     # how long the server asked to wait (Retry-After)
        "last_error": None,
        "started_at": None,
        "updated_at": None,
    }


def load_state():
    if not os.path.exists(STATE_PATH):
        st = default_state()
        st["started_at"] = now_iso()
        return st
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)
    merged = default_state()
    merged.update(st)
    return merged


def save_state(state):
    """Atomic write: tmp file + os.replace + fsync.
    That way the disk always holds either the old intact version or the new intact one."""
    state["updated_at"] = now_iso()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)
    dir_fd = os.open(os.path.dirname(STATE_PATH), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def shard_path(n):
    return os.path.join(RAW_DIR, f"shard_{n:05d}.jsonl")


def repair_last_line(path):
    """After a crash the last line of a shard may be partially written.
    We truncate the file to the last completed \\n."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n")
        f.truncate(0 if cut < 0 else cut + 1)


def append_sample(shard_num, record):
    """Appends one JSONL line and flushes it to disk. Returns the size."""
    path = shard_path(shard_num)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return os.path.getsize(path)


def log(msg):
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# NETWORK
# ---------------------------------------------------------------------------

def ask_ollama(prompt, temperature):
    """One request to /api/chat. Returns (response text, eval_count).

    Errors are not swallowed — do_one handles them (backoff/Retry-After)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
        "keep_alive": "5m",
    }
    r = requests.post(f"{HOST}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"empty response from the model: keys={list(data.keys())}")
    return content, data.get("eval_count", 0)


# ---------------------------------------------------------------------------
# ONE SAMPLE AND THE LOOP
# ---------------------------------------------------------------------------

def do_one(index, state):
    """Processes one sample.
    Returns a status: "ok" (saved), "skip" (saved as a reject,
    moving on), "net" (network error — we will retry the same index)."""
    task, variation, prompt = prompt_for(index)
    temperature = random.uniform(*TEMP_RANGE)
    log(f"#{index} request (t={temperature:.2f}): {task[:60]}...")
    try:
        response, tokens = ask_ollama(prompt, temperature)
    except requests.HTTPError as e:
        state["consecutive_errors"] += 1
        err = "?"
        if e.response is not None:
            err = f"HTTP {e.response.status_code}"
            ra = e.response.headers.get("retry-after")
            if ra:
                state["min_backoff"] = float(ra)
                err += f", Retry-After={float(ra):.0f}s"
        state["last_error"] = err
        log(f"#{index} ERROR: {err}")
        return "net"
    except Exception as e:
        state["consecutive_errors"] += 1
        state["last_error"] = f"{type(e).__name__}: {e}"
        log(f"#{index} ERROR: {state['last_error']}")
        return "net"

    c_code = extract_c(response)
    if c_code is None:
        state["extract_fails"] += 1
        log(f"#{index} response has no C block "
            f"(attempt {state['extract_fails']}/{MAX_EXTRACT_FAILS})")
        if state["extract_fails"] < MAX_EXTRACT_FAILS:
            return "net"   # retry the same index
        # consistently returns no code — record a reject and move on,
        # otherwise we could get stuck on a "bad" task forever
        record = {
            "id": index, "ts": now_iso(), "model": MODEL, "gcc": GCC_VERSION,
            "task": task, "variation": variation,
            "temperature": round(temperature, 3), "prompt": prompt,
            "extract_ok": False, "raw_response": response[:20000],
            "prompt_tokens": tokens,
            "compile_o0_ok": False, "asm_o0": None, "diag_o0": "no C block",
            "compile_o2_ok": False, "asm_o2": None, "diag_o2": "no C block",
        }
        repair_last_line(shard_path(state["shard_num"]))
        size = append_sample(state["shard_num"], record)
        state["samples_no_c"] += 1
        state["extract_fails"] = 0
        state["prompt_index"] = index + 1
        if size >= MAX_SHARD_BYTES:
            state["shard_num"] += 1
        save_state(state)
        log(f"#{index} reject recorded (no C block), moving on")
        return "skip"

    o0 = compile_c(c_code, "O0")
    o2 = compile_c(c_code, "O2")
    record = {
        "id": index, "ts": now_iso(), "model": MODEL, "gcc": GCC_VERSION,
        "task": task, "variation": variation,
        "temperature": round(temperature, 3), "prompt": prompt,
        "extract_ok": True,
        "c_source": c_code,
        "prompt_tokens": tokens,
        "compile_o0_ok": o0["ok"], "asm_o0": o0["asm"], "diag_o0": o0["diag"],
        "compile_o2_ok": o2["ok"], "asm_o2": o2["asm"], "diag_o2": o2["diag"],
    }
    repair_last_line(shard_path(state["shard_num"]))
    size = append_sample(state["shard_num"], record)
    state["samples_ok"] += 1
    state["extract_fails"] = 0
    state["consecutive_errors"] = 0
    state["last_error"] = None
    state["min_backoff"] = None
    if size >= MAX_SHARD_BYTES:
        state["shard_num"] += 1
    if not (o0["ok"] or o2["ok"]):
        state["samples_bad_c"] += 1
        log(f"#{index} C is present, but gcc failed to compile it: {o0['diag'][:200]}")
    else:
        n0 = len(o0["asm"].splitlines()) if o0["ok"] else 0
        n2 = len(o2["asm"].splitlines()) if o2["ok"] else 0
        log(f"#{index} OK ({tokens} tokens, C lines {len(c_code.splitlines())}, "
            f"asm -O0 {n0} lines, -O2 {n2} lines)")
    state["prompt_index"] = index + 1
    save_state(state)
    return "ok"


def run_forever(state):
    log(f"=== start/resume: continuing from sample #{state['prompt_index']}, "
        f"{COMBO} prompt combinations in total ===")
    backoff = 5.0
    last_state_ts = time.time()
    while True:
        status = do_one(state["prompt_index"], state)
        if status in ("ok", "skip"):
            backoff = 5.0
            wait = PAUSE + random.uniform(*JITTER)
        else:
            # network error: exponential backoff 5s, 10s, 20s ... up to an hour,
            # but not less than Retry-After if the server sent it
            backoff = min(backoff * 2, MAX_BACKOFF)
            wait = max(backoff, state.get("min_backoff") or 0)
            log(f"waiting {wait:.0f}s before retrying")
        if time.time() - last_state_ts > STATE_EVERY_SECONDS:
            save_state(state)
            last_state_ts = time.time()
        time.sleep(wait)


# ---------------------------------------------------------------------------
# SELF-TEST (offline, no network) AND CLI
# ---------------------------------------------------------------------------

SELF_TEST_C = r"""
#include <stdio.h>

static long square(long x) { return x * x; }

int main(void)
{
    int n = 0;
    if (scanf("%d", &n) != 1)
        return 1;
    printf("%ld\n", square(n));
    return 0;
}
"""


def self_test():
    """Checks only the gcc part: compilation, listing and cleaning of service lines.
    No network required. Returns 0 on success."""
    print("1) extracting the C block from the response")
    sample = "Here is the code:\n```c\nint main(void){return 0;}\n```\nDone."
    code = extract_c(sample)
    assert code == "int main(void){return 0;}", repr(code)
    assert extract_c("no block") is None
    print("   ok")

    print("2) gcc -O0/-O2 and listing cleanup")
    o0 = compile_c(SELF_TEST_C, "O0")
    o2 = compile_c(SELF_TEST_C, "O2")
    assert o0["ok"] and o0["asm"], o0["diag"]
    assert o2["ok"] and o2["asm"], o2["diag"]
    for name, res in (("O0", o0), ("O2", o2)):
        asm = res["asm"]
        assert ".file" not in asm, f"{name}: .file still present"
        assert ".ident" not in asm, f"{name}: .ident still present"
        assert ".note" not in asm, f"{name}: .note still present"
        assert "/tmp/" not in asm, f"{name}: temporary path leaked"
        assert not asm.startswith("#"), f"{name}: compiler header still present"
        assert asm.lstrip().startswith(".text"), f"{name}: does not start with .text"
        assert "main:" in asm, f"{name}: no main label"
        assert "square" in asm or "imul" in asm, f"{name}: no square body"
        print(f"   {name}: {len(asm.splitlines())} listing lines — ok")

    print("3) broken C -> compile fail, not an exception")
    bad = compile_c("int main(void) { this is not C }", "O0")
    assert not bad["ok"] and bad["diag"], bad
    print("   ok")

    print("4) prompts are deterministic and cover the space")
    a = prompt_for(0)
    b = prompt_for(0)
    assert a == b
    assert prompt_for(len(TASKS)) != a          # the next variation
    assert prompt_for(COMBO) == a               # the cycle has closed
    print(f"   ok ({len(TASKS)} tasks, {len(VARIATIONS)} variations, "
          f"{COMBO} combinations)")

    print("ALL SELF-TESTS PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Line A1: C -> gcc listing pairs (O0 and O2) via beam")
    ap.add_argument("--once", action="store_true",
                    help="do exactly one sample and exit (connectivity check)")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="show the first N prompts and exit (no network)")
    ap.add_argument("--status", action="store_true", help="show the checkpoint")
    ap.add_argument("--self-test", action="store_true",
                    help="offline check of the gcc part, no network")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    os.makedirs(RAW_DIR, exist_ok=True)
    state = load_state()

    if args.dry_run:
        for i in range(state["prompt_index"],
                       state["prompt_index"] + args.dry_run):
            task, variation, prompt = prompt_for(i)
            print(f"\n#{i}\n{prompt}")
        return

    if args.status:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if args.once:
        status = do_one(state["prompt_index"], state)
        print(f"status: {status}")
        return

    while True:
        try:
            run_forever(state)
        except KeyboardInterrupt:
            save_state(state)
            log("stopped manually (Ctrl+C)")
            return


if __name__ == "__main__":
    main()
