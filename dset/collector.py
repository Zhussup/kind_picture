#!/usr/bin/env python3
"""
collector.py — endless corpus collector for a programming dataset.

How it works (briefly):
  1. The infinite numbering script takes a sample number (0, 1, 2, ...) and
     deterministically turns it into an English programming prompt
     (category × topic × language × variation — no randomness, so after a
     restart it resumes exactly where it left off, and duplicates are impossible).
  2. Sends the prompt to the Ollama API (local server, beam:cloud model
     via Ollama Cloud).
  3. Saves the response line by line to dset/raw/shard_NNNNN.jsonl and
     immediately flushes the file to disk (flush + fsync) so nothing is lost
     on a power failure.
  4. Updates state.json (the checkpoint) with an atomic write.

The program NEVER exits on its own: on any error (no internet, server not
responding, model unavailable) it waits and tries again. The script can be
killed at any moment and simply restarted with run.sh.

Run:      bash run.sh          (background mode, see run.sh)
One-off:  python3 collector.py --once     (one sample, to check connectivity)
          python3 collector.py --dry-run 5  (show 5 prompts without network)
          python3 collector.py --status    (show the checkpoint)
"""

import argparse
import ast
import datetime
import json
import os
import random
import re
import signal
import sys
import time

import requests

# ---------------------------------------------------------------------------
# SETTINGS (can be changed right here)
# ---------------------------------------------------------------------------

MODEL = "beam:cloud"                # Ollama Cloud model
HOST = "http://localhost:11434"     # local Ollama server address
SYSTEM_PROMPT = (
    "You are an expert programmer. Answer in English. "
    "Be practical: always show working, idiomatic code in fenced code blocks "
    "and explain it clearly."
)
TEMP_RANGE = (0.6, 0.95)   # temperature is chosen randomly for each request:
                           # this reduces the chance of "going in circles" (mode collapse)
JITTER = (0.0, 1.5)        # random addition to the pause — traffic is livelier and less
                           # like machine hammering (anti-abuse protection)
REQUEST_TIMEOUT = 900   # seconds to wait for one response (cloud models are slow)
PAUSE = 2.0          # pause between requests, seconds
MAX_SHARD_BYTES = 8 * 1024 * 1024   # split the corpus into files of ~8 MB
MAX_BACKOFF = 3600.0  # after a series of errors, wait no longer than 1 hour
STATE_EVERY_SECONDS = 60  # update checkpoint timestamps no more than once a minute

# Paths — everything inside the dset/ folder, relative to the script itself.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")          # responses land here (in chunks)
STATE_PATH = os.path.join(BASE_DIR, "state.json")  # checkpoint
LOG_PATH = os.path.join(BASE_DIR, "collector.log")

# ---------------------------------------------------------------------------
# PROMPT GENERATION
# ---------------------------------------------------------------------------

# Topics — shared by all programming languages (in English).
TOPICS = [
    "variables and data types", "lists and arrays", "dictionaries and hash maps",
    "sets", "strings and text processing", "regular expressions", "file reading and writing",
    "working with JSON", "working with CSV files", "dates and timestamps",
    "integer and floating point numbers", "bitwise operations", "boolean logic",
    "conditionals and branches", "loops", "functions and parameters",
    "return values", "recursion", "closures", "lambda functions",
    "higher-order functions", "iterators", "generators", "comprehensions",
    "classes and objects", "constructors", "inheritance", "polymorphism",
    "interfaces and abstract classes", "composition over inheritance",
    "static and instance members", "operator overloading", "modules and imports",
    "packages and namespaces", "error handling and exceptions", "custom exceptions",
    "null and optional values", "type conversions", "type checking",
    "static typing", "generics", "enums", "structs",
    "memory management", "pointers and references", "garbage collection",
    "stack and heap", "value vs reference semantics", "immutability",
    "arrays vs linked lists", "stacks and queues", "binary search",
    "hash tables", "binary trees", "graphs and traversals", "dynamic programming",
    "greedy algorithms", "sorting algorithms", "searching algorithms",
    "big O notation", "time and space complexity", "sliding window technique",
    "two-pointer technique", "prefix sums", "matrix operations",
    "random numbers", "statistics and math functions", "unit testing",
    "test-driven development", "mocking and test doubles", "debugging techniques",
    "logging", "environment variables and configuration", "command line arguments",
    "input validation", "defensive programming", "code style and readability",
    "naming conventions", "comments and documentation", "refactoring",
    "design patterns", "singleton pattern", "factory pattern", "observer pattern",
    "model–view–controller structure", "layered architecture",
    "multithreading", "thread safety", "async and await", "event loops",
    "processes and subprocesses", "signals and timers", "synchronization and locks",
    "deadlocks and race conditions", "HTTP requests", "REST API clients and servers",
    "sockets and networking", "caching", "memoization",
    "databases basics", "SQL queries", "database indexes", "transactions",
    "ORM basics", "serialization and deserialization", "cryptography basics",
    "checksums and hashes", "encoding and character sets", "unicode",
    "binary data and bytes", "performance profiling", "micro-optimizations",
    "lazy evaluation", "functional programming style", "pure functions",
    "immutable data structures", "command line tools", "build tooling",
    "version control basics", "code review practice", "reading other people's code",
]
TOPICS = sorted(set(TOPICS))  # remove duplicates and fix the order

LANGS = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++", "Java",
    "C#", "Kotlin", "Swift", "Ruby", "PHP", "Bash (shell script)", "SQL",
]

# Wording variations — add diversity without changing the meaning.
VARIATIONS = [
    "Give a thorough answer.", "Keep it concise but complete.",
    "Focus on best practices.", "Focus on common mistakes to avoid.",
    "Include a real-world example.", "Explain as if teaching a beginner.",
    "Explain for an experienced developer.", "Add short test examples.",
    "Mention edge cases.", "Show how to do it wrong first, then right.",
]

# ---------------------------------------------------------------------------
# Categories: (name, function(topic, language, variation) -> prompt text)
# The order in the lists is fixed — it matters for checkpoint determinism.

def c_explain(topic, lang, var):
    return (f"Explain how {topic} works in {lang}: the idea, the syntax, and "
            f"typical use cases, with short code examples. {var}")

def c_debug(topic, lang, var):
    return (f"Write a small {lang} snippet containing a typical mistake related "
            f"to {topic}. Then show the corrected version and explain what was "
            f"wrong and why. {var}")

def c_pitfalls(topic, lang, var):
    return (f"What are the most common pitfalls when working with {topic} in "
            f"{lang}? List them and show code for each. {var}")

def c_exercise(topic, lang, var):
    return (f"Give a short practical exercise about {topic} in {lang}, then "
            f"solve it step by step with explanations and a final solution. {var}")

def c_interview(topic, lang, var):
    return (f"Answer like in a coding interview: explain {topic} in {lang}, "
            f"then implement the core of it with clean code and comments. {var}")

def c_edge(topic, lang, var):
    return (f"What edge cases must be handled when using {topic} in {lang}? "
            f"Show code examples for each case. {var}")

# The order in the lists is fixed — it matters for checkpoint determinism.
def c_impl(topic, lang, var):
    return (f"Implement {topic} in {lang}. Show a complete, runnable example. {var}")

CATEGORIES = [
    ("implement", c_impl),
    ("explain", c_explain),
    ("debug", c_debug),
    ("pitfalls", c_pitfalls),
    ("exercise", c_exercise),
    ("interview", c_interview),
    ("edge_cases", c_edge),
]
NCAT = len(CATEGORIES)
NTOP = len(TOPICS)
NLANG = len(LANGS)
NVAR = len(VARIATIONS)
COMBO = NCAT * NTOP * NLANG * NVAR   # how many unique prompts exist


def prompt_for(index):
    """Deterministically turns a sample number into (category, language, prompt).

    Division with remainder decomposes index into layers: category changes most
    often, then topic, then language, then wording variation. The same index
    always yields the same prompt — so resuming after a restart works without
    duplicates or gaps.
    """
    cat_i = index % NCAT
    rest = index // NCAT
    top_i = rest % NTOP
    rest //= NTOP
    lang_i = rest % NLANG
    rest //= NLANG
    var_i = rest % NVAR

    name, fn = CATEGORIES[cat_i]
    text = fn(TOPICS[top_i], LANGS[lang_i], VARIATIONS[var_i])
    return name, LANGS[lang_i], text


# ---------------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_state():
    """Reads the checkpoint. If the file is absent (first run) — start from scratch."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": 1,
        "prompt_index": 0,     # next sample to generate
        "shard_num": 1,        # current data file number (raw/shard_N.jsonl)
        "samples_ok": 0,
        "samples_failed": 0,
        "consecutive_errors": 0,
        "last_error": None,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }


def save_state(state):
    """Atomic checkpoint write: write to a temporary file and rename.

    Rename over an existing file on Linux is a single atomic operation,
    so even on a power failure state.json will be either the old one or
    the new one, but never corrupted.
    """
    state["updated_at"] = now_iso()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # force the data out to disk
    os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------------------
# WRITING RESPONSES
# ---------------------------------------------------------------------------

def shard_path(shard_num):
    return os.path.join(RAW_DIR, f"shard_{shard_num:05d}.jsonl")


def repair_last_line(path):
    """If the last chunk of the file does not end with \\n (an interruption at
    power loss between the write and fsync) — truncate it to the last complete
    line. A broken half-line is the only possible kind of JSONL corruption."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        tail = 4096
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - tail))
        data = f.read()
    cut = data.rfind(b"\n")
    if cut == -1:      # there is no newline in the tail at all — the whole file is broken
        keep = 0
    else:              # keep everything up to and including the last newline
        keep = size - (len(data) - (cut + 1))
    with open(path, "r+b") as f:
        f.truncate(keep)


def append_sample(shard_num, record):
    """Appends one record to the current shard. Returns the file size after writing."""
    path = shard_path(shard_num)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())  # force flush to disk: we do not lose data
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# OLLAMA REQUEST
# ---------------------------------------------------------------------------

def validate_python_blocks(response):
    """Checks the syntax of ALL code blocks in the response (```python) via
    ast.parse — the Python module that parses source into a syntax tree: if the
    text does not parse, the code is definitely broken. Returns (all_blocks_ok,
    first_error)."""
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", response, re.S)
    for i, code in enumerate(blocks, 1):
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"block {i}: {e.msg} (line {e.lineno})"
    return True, None


def ask_ollama(prompt, temperature):
    """One request to /api/chat. Returns (response text, token count).

    On a problem — raises an exception; the caller decides what to do.
    """
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
        raise RuntimeError(f"empty response from model: keys={list(data.keys())}")
    return content, data.get("eval_count", 0)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def log(msg):
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def do_one(index, state):
    """Generates and saves one sample. Returns True on success."""
    category, lang, prompt = prompt_for(index)
    temperature = random.uniform(*TEMP_RANGE)
    log(f"#{index} [{category}/{lang}] request (t={temperature:.2f})...")
    try:
        response, tokens = ask_ollama(prompt, temperature)
    except requests.HTTPError as e:
        state["samples_failed"] += 1
        state["consecutive_errors"] += 1
        err = "?"
        if e.response is not None:
            # 429 Too Many Requests / 503: if the server sent Retry-After —
            # how many seconds it asks to wait — we take that into account below
            err = f"HTTP {e.response.status_code}"
            ra = e.response.headers.get("retry-after")
            if ra:
                state["min_backoff"] = float(ra)
                err += f", Retry-After={float(ra):.0f}s"
        state["last_error"] = err
        log(f"#{index} ERROR: {err}")
        return False
    except Exception as e:
        state["samples_failed"] += 1
        state["consecutive_errors"] += 1
        state["last_error"] = f"{type(e).__name__}: {e}"
        log(f"#{index} ERROR: {state['last_error']}")
        return False

    # Syntax check: Python only (no parser at hand for other languages).
    # Do not discard, just mark — the "store/filter" decision is deferred to build_dataset.
    if lang == "Python":
        code_ok, code_err = validate_python_blocks(response)
    else:
        code_ok, code_err = True, None

    record = {
        "id": index,
        "ts": now_iso(),
        "model": MODEL,
        "category": category,
        "lang": lang,
        "temperature": round(temperature, 3),
        "prompt": prompt,
        "response": response,
        "prompt_tokens": tokens,
        "code_valid": code_ok,
        "code_error": code_err,
    }
    repair_last_line(shard_path(state["shard_num"]))
    size = append_sample(state["shard_num"], record)
    saved_shard = state["shard_num"]

    state["prompt_index"] = index + 1
    state["samples_ok"] += 1
    state["consecutive_errors"] = 0
    state["last_error"] = None
    state["min_backoff"] = None    # the server no longer asks us to wait — reset it
    if size >= MAX_SHARD_BYTES:   # the file grew — start the next one
        state["shard_num"] += 1
    save_state(state)
    code_note = "code ok" if code_ok else f"BROKEN CODE ({code_err})"
    log(f"#{index} OK ({tokens} tokens, t={temperature:.2f}, {code_note}, "
        f"saved to shard_{saved_shard:05d}.jsonl)")
    return True


def run_forever(state):
    log(f"=== start/resume: continuing from sample #{state['prompt_index']}, "
        f"total prompt combinations {COMBO} ===")
    backoff = 5.0
    last_state_ts = time.time()
    while True:
        if do_one(state["prompt_index"], state):
            backoff = 5.0
            wait = PAUSE + random.uniform(*JITTER)
        else:
            # exponential backoff: 5s, 10s, 20s ... up to 1 hour;
            # if the server replied with Retry-After — wait at least that long
            backoff = min(backoff * 2, MAX_BACKOFF)
            wait = max(backoff, state.get("min_backoff") or 0)
            log(f"waiting {wait:.0f}s before retry")
        if time.time() - last_state_ts > STATE_EVERY_SECONDS:
            save_state(state)
            last_state_ts = time.time()
        time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Corpus collector via Ollama Cloud")
    parser.add_argument("--once", action="store_true",
                        help="generate exactly one sample and exit (connectivity check)")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="show the first N prompts and exit (no network)")
    parser.add_argument("--status", action="store_true", help="show the checkpoint and exit")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    if args.dry_run:
        for i in range(args.dry_run):
            cat, lang, text = prompt_for(i)
            print(f"[{i}] {cat} / {lang}\n{text}\n")
        return
    if args.status:
        st = load_state()
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return

    state = load_state()

    # Graceful shutdown on Ctrl+C and on `kill` (systemd etc.): the checkpoint is
    # already saved after every sample, so here it is enough to simply exit.
    def on_term(_sig, _frm):
        log("stop signal received — exiting")
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)

    try:
        if args.once:
            ok = do_one(state["prompt_index"], state)
            save_state(state)
            sys.exit(0 if ok else 1)
        run_forever(state)
    except KeyboardInterrupt:
        log("stopped (Ctrl+C) — progress saved")


if __name__ == "__main__":
    main()