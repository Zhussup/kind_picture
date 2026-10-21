#!/usr/bin/env python3
"""
verifier.py — verifies asm programs for the C→ASM dataset.

Program contract (mandatory for all dataset tasks):
  - x86-64 architecture, AT&T syntax (as emitted by `gcc -S`);
  - entry point `_start`, termination via the `exit` syscall;
  - no libc (printf/scanf and the like): input/output only through the
    read(0) / write(1) syscalls.

Pipeline:
  1. the source into a temporary directory;
  2. `as --64 prog.s -o prog.o`;
  3. `ld prog.o -o prog`;
  4. for each test, a run: (stdin → expected stdout, expected exit code),
     with isolation — timeout, RLIMIT (memory/CPU/core/processes), empty environment;
  5. verdict (JSON or human-readable).

CLI return codes:
  0 — all tests passed; 1 — it built, but some tests failed;
  2 — failed to assemble/link; 3 — input error (missing file, broken JSON,
  empty test list, tests are not a dict/list).

About the constants: timeouts do NOT constrain correct programs (those run
in microseconds) — they set the price of handling broken/hanging programs.

Usage: python3 verifier.py prog.s --tests prog.test.json [--json]
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile

# --- constants (rationale in the README): only the price of a broken trick ---
ASM_TIMEOUT = 5                  # seconds for as/ld (in practice: tens of ms)
RUN_TIMEOUT = 2                  # seconds per test — a tax on hanging programs
MEM_LIMIT = 256 * 1024 * 1024    # bytes of address space
CPU_LIMIT = 2                    # seconds of CPU time
CORE_LIMIT = 0                   # do not write core dumps (otherwise crashes would fill the disk)
NPROC_LIMIT = 64                 # against fork bombs

# Human-readable explanation of negative codes (signals); useful for the dataset
SIGNAL_HINTS = {
    -11: "SIGSEGV — common cause: returning from _start without the exit syscall, or a bad dereference",
    -9:  "SIGKILL — killed by the kernel: almost always our CPU limit or RAM limit (race with the wall timeout)",
    -4:  "SIGILL — illegal instruction",
    -8:  "SIGFPE — division by zero / overflow",
    -7:  "SIGBUS",
}


def _limits():
    """preexec_fn for subprocess: rlimits are set BEFORE the child code runs.

    preexec_fn runs in the child between fork and exec; that is the only window
    where we can limit code we do not trust yet."""
    for res, val in (
        (resource.RLIMIT_AS, MEM_LIMIT),
        (resource.RLIMIT_CPU, CPU_LIMIT),
        (resource.RLIMIT_CORE, CORE_LIMIT),
        (resource.RLIMIT_NPROC, NPROC_LIMIT),
    ):
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass  # if the kernel does not allow limiting something, work without that limit


def _norm(b: bytes) -> bytes:
    """stdout normalization: ignore differences only in trailing \\n."""
    return b.rstrip(b"\n")


def _exit_matches(expected, got) -> bool:
    """expected exit: None = do not check; <0 = signal (compare exactly);
    >=0 — compare mod 256 (the syscall exit code is truncated to a byte)."""
    if expected is None:
        return True
    if expected < 0 or got < 0:
        return expected == got
    return (got & 0xFF) == (expected & 0xFF)


def _tool(cmd, timeout):
    """Runs as/ld, returns (success, error/warning text)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env={})
    except subprocess.TimeoutExpired:
        return False, f"did not finish within {timeout} s"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    return p.returncode == 0, (p.stderr or p.stdout).strip()


def run_once(prog, workdir, stdin_bytes):
    """One run of the built binary. Never depends on the program's contents:
    stdin is provided, output is captured, limits and timeout are mandatory."""
    try:
        p = subprocess.run(
            [prog],
            input=stdin_bytes,
            capture_output=True,     # capture stdout+stderr
            timeout=RUN_TIMEOUT,
            preexec_fn=_limits,
            env={},                  # empty environment: determinism
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return {"stdout": b"", "stderr": b"", "exit": None,
                "timed_out": True, "exec_error": None}
    except OSError as e:
        return {"stdout": b"", "stderr": b"", "exit": None,
                "timed_out": False, "exec_error": f"{type(e).__name__}: {e}"}
    return {"stdout": p.stdout, "stderr": p.stderr, "exit": p.returncode,
            "timed_out": False, "exec_error": None}


def verify(asm_text, tests):
    """Main function: verdict for the asm text and a list of tests."""
    verdict = {
        "assembled": False,     # passed as
        "linked": False,        # passed ld
        "stage": "asm",         # where we stopped: asm|link|run|no_tests
        "error": None,          # first error text (truncated)
        "tests": [],
        "n_total": len(tests),
        "n_passed": 0,
        "all_passed": False,
    }
    if not isinstance(tests, list) or not tests:
        verdict["stage"] = "no_tests"
        verdict["error"] = "test list is empty or malformed"
        return verdict

    with tempfile.TemporaryDirectory(prefix="asmverify_") as workdir:
        src = os.path.join(workdir, "prog.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(asm_text)

        ok, log = _tool(["as", "--64", src, "-o", os.path.join(workdir, "prog.o")],
                        ASM_TIMEOUT)
        if not ok:
            verdict["error"] = log[:2000]
            return verdict
        verdict["assembled"] = True
        verdict["stage"] = "link"

        ok, log = _tool(["ld", os.path.join(workdir, "prog.o"),
                         "-o", os.path.join(workdir, "prog")], ASM_TIMEOUT)
        if not ok:
            verdict["error"] = log[:2000]
            return verdict
        verdict["linked"] = True
        verdict["stage"] = "run"

        prog = os.path.join(workdir, "prog")
        for t in tests:
            r = run_once(prog, workdir, str(t.get("stdin", "")).encode())
            rec = {
                "name": t.get("name", ""),
                "stdin": t.get("stdin", ""),
                "expected_stdout": t.get("stdout", ""),
                "got_stdout": r["stdout"].decode("utf-8", "replace"),
                "expected_exit": t.get("exit"),
                "got_exit": r["exit"],
                "signal": SIGNAL_HINTS.get(r["exit"]),
                "timed_out": r["timed_out"],
                "exec_error": r["exec_error"],
                "stderr_tail": r["stderr"].decode("utf-8", "replace")[-300:],
            }
            if r["exec_error"] or r["timed_out"]:
                rec["passed"] = False
            else:
                ok_out = _norm(r["stdout"]) == _norm(str(t.get("stdout", "")).encode())
                ok_exit = _exit_matches(t.get("exit"), r["exit"])
                rec["passed"] = bool(ok_out and ok_exit)
            if rec["passed"]:
                verdict["n_passed"] += 1
            verdict["tests"].append(rec)

    verdict["all_passed"] = verdict["n_passed"] == verdict["n_total"]
    return verdict


def print_human(v):
    """Short human-readable report."""
    if not v["assembled"]:
        print(f"[ASM FAIL] {v['error']}")
        return
    if not v["linked"]:
        print(f"[LINK FAIL] {v['error']}")
        return
    print(f"[RUN] tests: {v['n_passed']}/{v['n_total']}")
    for t in v["tests"]:
        if t["passed"]:
            print(f'  + {t["name"] or t["stdin"]!r}')
            continue
        if t["timed_out"]:
            print(f'  - {t["name"]}: timeout {RUN_TIMEOUT}s (hanging?)')
        elif t["exec_error"]:
            print(f'  - {t["name"]}: exec error {t["exec_error"]}')
        else:
            mark = t["signal"] or ""
            print(f'  - {t["name"]}: exit got/exp. {t["got_exit"]}/{t["expected_exit"]} {mark}, '
                  f'stdout got/exp. {t["got_stdout"]!r}/{t["expected_stdout"]!r}')


def main():
    ap = argparse.ArgumentParser(description="Verifier for asm programs (x86-64, AT&T, _start, syscalls only)")
    ap.add_argument("prog_s", help="source prog.s")
    ap.add_argument("--tests", required=True, help="JSON file with a list of tests")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON")
    args = ap.parse_args()

    try:
        with open(args.prog_s, "r", encoding="utf-8") as f:
            src = f.read()
        with open(args.tests, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"input error: {e}", file=sys.stderr)
        sys.exit(3)

    verdict = verify(src, tests)
    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print_human(verdict)

    if not verdict.get("assembled"):
        sys.exit(2)
    sys.exit(0 if verdict["all_passed"] else 1)


if __name__ == "__main__":
    main()