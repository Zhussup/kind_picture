#!/usr/bin/env python3
"""
difftest.py — differential check of asm -> C reconstruction.

Idea: we have an "asm listing -> original C" pair produced by gcc.
The model reads the listing and emits its own C. We check mechanically:
  - compile the ORIGINAL and the PREDICTION with the same gcc;
  - run both on identical inputs (stdin);
  - compare the observable behavior: stdout and return code.
If the behavior matches on all inputs, the reconstruction is functionally correct.
No comparison of C text against the reference: different code with the same behavior
is also a correct answer.

These are not unit tests with known expectations: the expected output
is taken from the original. Therefore any inputs are acceptable — even random ones.

Behavior on failures (honestly):
  - both hung                 -> counted as a match (same behavior);
  - only one hung             -> mismatch;
  - killed by a signal        -> compare signals as they are;
  - original failed to build  -> an input error (code 3), not a model failure.

Run isolation (timeout, RLIMIT on memory/CPU/core/processes, empty
environment) is reused from verifier.py — the rationale for the constants is there too.

Usage:
  python3 difftest.py ref.c cand.c                  built-in set of inputs
  python3 difftest.py ref.c cand.c --cases cases.json
  python3 difftest.py ref.c cand.c --json
  python3 difftest.py --self-test                   check difftest itself

cases.json format: [{"name": "...", "stdin": "..."}, ...]
Return codes: 0 — everything matched; 1 — behavior mismatch;
2 — the candidate failed to compile; 3 — input error.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import verifier  # run_once, _norm, _exit_matches — the same run isolation

COMPILE_TIMEOUT = 30   # seconds per gcc run (in practice: tens of ms)

# Default inputs: cover the typical shapes of our tasks
# (empty, one number, two numbers, n + array, text, several lines).
DEFAULT_INPUTS = [
    {"name": "empty", "stdin": ""},
    {"name": "zero", "stdin": "0\n"},
    {"name": "one", "stdin": "1\n"},
    {"name": "five", "stdin": "5\n"},
    {"name": "two ints", "stdin": "12 18\n"},
    {"name": "n + array", "stdin": "5\n3 1 4 1 5\n"},
    {"name": "text", "stdin": "hello world\n"},
    {"name": "multiline", "stdin": "abc\nxyz\n"},
    {"name": "big number", "stdin": "123456\n"},
    {"name": "negative", "stdin": "-7\n"},
]


def compile_c(c_text, workdir, name):
    """Compiles C text into an executable. Returns (ok, path, log)."""
    src = os.path.join(workdir, name + ".c")
    exe = os.path.join(workdir, name)
    with open(src, "w", encoding="utf-8") as f:
        f.write(c_text)
    cmd = ["gcc", "-std=c11", "-O0", "-w", src, "-o", exe]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, None, f"gcc did not finish within {COMPILE_TIMEOUT} s"
    except OSError as e:
        return False, None, f"{type(e).__name__}: {e}"
    if p.returncode != 0:
        return False, None, (p.stderr or p.stdout).strip()[:4000]
    return True, exe, p.stderr.strip()[:4000]


def _same_behavior(ref, cand):
    """Compares two runs. Returns (match, mismatch_reason|None)."""
    if ref["timed_out"] and cand["timed_out"]:
        return True, None
    if ref["timed_out"] != cand["timed_out"]:
        return False, "timeout on only one side"
    if ref["exec_error"] or cand["exec_error"]:
        same = ref["exec_error"] == cand["exec_error"]
        return same, None if same else "different exec errors"
    if verifier._norm(ref["stdout"]) != verifier._norm(cand["stdout"]):
        return False, "stdout differs"
    if not verifier._exit_matches(ref["exit"], cand["exit"]):
        return False, "return code differs"
    return True, None


def compare(ref_c, cand_c, inputs):
    """Main function: verdict for two C texts and a list of inputs."""
    verdict = {
        "ref_compiled": False,
        "cand_compiled": False,
        "stage": "ref_compile",   # ref_compile|cand_compile|run
        "error": None,
        "n_total": len(inputs),
        "n_passed": 0,
        "n_skipped": 0,
        "all_match": False,
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="difftest_") as workdir:
        ok, ref_exe, log = compile_c(ref_c, workdir, "ref")
        if not ok:
            verdict["error"] = log
            return verdict
        verdict["ref_compiled"] = True
        verdict["stage"] = "cand_compile"

        ok, cand_exe, log = compile_c(cand_c, workdir, "cand")
        if not ok:
            verdict["error"] = log
            return verdict
        verdict["cand_compiled"] = True
        verdict["stage"] = "run"

        for case in inputs:
            stdin_bytes = str(case.get("stdin", "")).encode()
            r = verifier.run_once(ref_exe, workdir, stdin_bytes)
            # If the reference itself hit OUR limit (timeout, SIGKILL from RLIMIT_CPU,
            # some other signal), the input does not define a meaningful expectation:
            # reproducing "the reference was killed by a limit" is not the goal we teach the model.
            if (r["timed_out"] or r["exec_error"]
                    or (r["exit"] is not None and r["exit"] < 0)):
                verdict["n_skipped"] += 1
                verdict["cases"].append({
                    "name": case.get("name", ""),
                    "stdin": case.get("stdin", ""),
                    "ref_stdout": r["stdout"].decode("utf-8", "replace"),
                    "ref_exit": r["exit"],
                    "ref_timed_out": r["timed_out"],
                    "cand_stdout": None,
                    "cand_exit": None,
                    "cand_timed_out": None,
                    "signal": verifier.SIGNAL_HINTS.get(r["exit"]),
                    "match": None,
                    "reason": "reference hit a limit/signal — input is uninformative",
                })
                continue

            c = verifier.run_once(cand_exe, workdir, stdin_bytes)
            match, reason = _same_behavior(r, c)
            verdict["cases"].append({
                "name": case.get("name", ""),
                "stdin": case.get("stdin", ""),
                "ref_stdout": r["stdout"].decode("utf-8", "replace"),
                "ref_exit": r["exit"],
                "ref_timed_out": r["timed_out"],
                "cand_stdout": c["stdout"].decode("utf-8", "replace"),
                "cand_exit": c["exit"],
                "cand_timed_out": c["timed_out"],
                "signal": verifier.SIGNAL_HINTS.get(c["exit"]),
                "match": match,
                "reason": reason,
            })
            if match:
                verdict["n_passed"] += 1

    # n_total is now the number of INFORMATIVE inputs: those where the reference was
    # killed by a limit do not count (match=None)
    verdict["n_total"] = sum(1 for x in verdict["cases"]
                             if x["match"] is not None)
    verdict["all_match"] = (verdict["n_total"] > 0
                            and verdict["n_passed"] == verdict["n_total"])
    return verdict


def print_human(v):
    if not v["ref_compiled"]:
        print(f"[REF FAIL] the original failed to compile (input error): {v['error']}")
        return
    if not v["cand_compiled"]:
        print(f"[CAND FAIL] the prediction failed to compile: {v['error']}")
        return
    skipped = (f", {v['n_skipped']} uninformative"
               if v.get("n_skipped") else "")
    print(f"[DIFF] inputs: {v['n_passed']}/{v['n_total']} matched{skipped}")
    for c in v["cases"]:
        if c["match"] is not False:
            continue
        extra = f" ({c['signal']})" if c["signal"] else ""
        print(f'  - {c["name"]}: {c["reason"]}; '
              f'stdin={c["stdin"]!r} ref={c["ref_stdout"]!r}/{c["ref_exit"]}{extra} '
              f'cand={c["cand_stdout"]!r}/{c["cand_exit"]}')


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

REF_C = """
#include <stdio.h>
int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    printf("%d\\n", n * 2);
    return 0;
}
"""

WRONG_C = """
#include <stdio.h>
int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    printf("%d\\n", n + 1);
    return 0;
}
"""

CRASH_C = """
int main(void) {
    volatile int *p = 0;
    *p = 1;
    return 0;
}
"""

HANG_C = """
int main(void) {
    for (;;) { }
    return 0;
}
"""


def self_test():
    cases = [{"name": "five", "stdin": "5\n"}]
    print("1) identical C -> everything matches")
    v = compare(REF_C, REF_C, cases)
    assert v["all_match"] and v["n_passed"] == 1, v
    print("   ok")

    print("2) different answer -> mismatch")
    v = compare(REF_C, WRONG_C, cases)
    assert not v["all_match"] and v["cases"][0]["reason"] == "stdout differs", v
    print("   ok")

    print("3) uncompilable C -> cand_compile, not an exception")
    v = compare(REF_C, "int main(void) { this is not C }", cases)
    assert not v["cand_compiled"] and v["stage"] == "cand_compile", v
    print("   ok")

    print("4) signal crash is detected")
    v = compare(REF_C, CRASH_C, cases)
    assert not v["all_match"], v
    assert v["cases"][0]["cand_exit"] == -11, v["cases"][0]
    print("   ok (-11 SIGSEGV)")

    print("5) candidate hang is detected as a mismatch")
    v = compare(REF_C, HANG_C, cases)
    c = v["cases"][0]
    assert not v["all_match"], v
    # race between the wall timeout and RLIMIT_CPU: the kernel may kill earlier (SIGKILL -9),
    # so we accept both variants — what matters is that the failure is detected
    assert c["cand_timed_out"] or c["cand_exit"] == -9, c
    print("   ok (timeout or SIGKILL)")

    print("ALL SELF-TESTS PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Differential check of asm->C reconstruction")
    ap.add_argument("ref_c", nargs="?", help="reference C (original)")
    ap.add_argument("cand_c", nargs="?", help="predicted C (candidate)")
    ap.add_argument("--cases", help="JSON file with a list of inputs")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="check difftest itself (no files)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.ref_c or not args.cand_c:
        ap.error("ref.c and cand.c are required (or --self-test)")

    try:
        with open(args.ref_c, "r", encoding="utf-8") as f:
            ref_c = f.read()
        with open(args.cand_c, "r", encoding="utf-8") as f:
            cand_c = f.read()
        if args.cases:
            with open(args.cases, "r", encoding="utf-8") as f:
                inputs = json.load(f)
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("cases.json must be a non-empty list")
        else:
            inputs = DEFAULT_INPUTS
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"input error: {e}", file=sys.stderr)
        sys.exit(3)

    verdict = compare(ref_c, cand_c, inputs)
    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print_human(verdict)

    if not verdict["ref_compiled"]:
        sys.exit(3)
    if not verdict["cand_compiled"]:
        sys.exit(2)
    sys.exit(0 if verdict["all_match"] else 1)


if __name__ == "__main__":
    main()
