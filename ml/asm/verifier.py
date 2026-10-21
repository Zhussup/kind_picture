#!/usr/bin/env python3
"""
verifier.py — проверяет asm-программы для датасета C→ASM.

Контракт программы (обязателен для всех задач датасета):
  - архитектура x86-64, синтаксис AT&T (как выдаёт `gcc -S`);
  - точка входа `_start`, завершение — syscall `exit`;
  - никакого libc (printf/scanf и т.п.): ввод/вывод только syscalls
    read(0) / write(1).

Пайплайн:
  1. исходник во временный каталог;
  2. `as --64 prog.s -o prog.o`;
  3. `ld prog.o -o prog`;
  4. для каждого теста прогон: (stdin → ожидаемый stdout, ожидаемый exit code),
     с изоляцией — timeout, RLIMIT (память/CPU/core/процессы), пустое окружение;
  5. вердикт (JSON или человекочитаемо).

Коды возврата CLI:
  0 — все тесты прошли; 1 — собрался, но есть проваленные тесты;
  2 — не собрался/не слинковался; 3 — ошибка входа (нет файла, битый JSON,
  список тестов пуст, тесты не словарь/не список).

Про константы: таймауты НЕ ограничивают правильные программы (те работают
за микросекунды) — они задают цену обработки битых/висящих программ.

Запуск: python3 verifier.py prog.s --tests prog.test.json [--json]
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile

# --- константы (обоснование в README): только цены на битый трюк ---
ASM_TIMEOUT = 5                  # сек на as/ld (реальность: десятки мс)
RUN_TIMEOUT = 2                  # сек на один тест — налог на висящие программы
MEM_LIMIT = 256 * 1024 * 1024    # байт адресного пространства
CPU_LIMIT = 2                    # сек процессорного времени
CORE_LIMIT = 0                   # core dump не писать (иначе падёжки завалят диск)
NPROC_LIMIT = 64                 # от fork-бомб

# Человекочитаемая расшифровка отрицательных кодов (сигналов), для датасета полезно
SIGNAL_HINTS = {
    -11: "SIGSEGV — частая причина: возврат из _start без syscall exit, либо битое разыменование",
    -9:  "SIGKILL — убит ядром: почти всегда наш CPU-лимит или RAM-лимит (гонка с wall-таймаутом)",
    -4:  "SIGILL — недопустимая инструкция",
    -8:  "SIGFPE — деление на ноль / переполнение",
    -7:  "SIGBUS",
}


def _limits():
    """preexec_fn для subprocess: rlimits ставятся ДО запуска дочернего кода.

    preexec_fn исполняется в потомке между fork и exec; это единственное окно,
    где можно ограничить код, которому мы ещё не доверяем."""
    for res, val in (
        (resource.RLIMIT_AS, MEM_LIMIT),
        (resource.RLIMIT_CPU, CPU_LIMIT),
        (resource.RLIMIT_CORE, CORE_LIMIT),
        (resource.RLIMIT_NPROC, NPROC_LIMIT),
    ):
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass  # если ядро что-то не даёт ограничить — работаем без этого лимита


def _norm(b: bytes) -> bytes:
    """Нормализация stdout: игнорируем разницу только в хвостовых \\n."""
    return b.rstrip(b"\n")


def _exit_matches(expected, got) -> bool:
    """exit expected: None = не сверяем; <0 = сигнал (сравниваем честно);
    >=0 — сравниваем mod 256 (syscalls exit код обрезается до байта)."""
    if expected is None:
        return True
    if expected < 0 or got < 0:
        return expected == got
    return (got & 0xFF) == (expected & 0xFF)


def _tool(cmd, timeout):
    """Запускает as/ld, возвращает (успех, текст ошибок/предупреждений)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env={})
    except subprocess.TimeoutExpired:
        return False, f"не завершился за {timeout} с"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    return p.returncode == 0, (p.stderr or p.stdout).strip()


def run_once(prog, workdir, stdin_bytes):
    """Один прогон собранного бинаря. Никогда не зависит от содержимого программы:
    stdin подан, вывод пойман, лимиты и таймаут обязательны."""
    try:
        p = subprocess.run(
            [prog],
            input=stdin_bytes,
            capture_output=True,     # перехватываем stdout+stderr
            timeout=RUN_TIMEOUT,
            preexec_fn=_limits,
            env={},                  # пустое окружение: детерминизм
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
    """Главная функция: вердикт по тексту asm и списку тестов."""
    verdict = {
        "assembled": False,     # прошёл as
        "linked": False,        # прошёл ld
        "stage": "asm",         # где остановились: asm|link|run|no_tests
        "error": None,          # текст первой ошибки (усечённый)
        "tests": [],
        "n_total": len(tests),
        "n_passed": 0,
        "all_passed": False,
    }
    if not isinstance(tests, list) or not tests:
        verdict["stage"] = "no_tests"
        verdict["error"] = "список тестов пуст или битый"
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
    """Короткий человекочитаемый отчёт."""
    if not v["assembled"]:
        print(f"[ASM FAIL] {v['error']}")
        return
    if not v["linked"]:
        print(f"[LINK FAIL] {v['error']}")
        return
    print(f"[RUN] тесты: {v['n_passed']}/{v['n_total']}")
    for t in v["tests"]:
        if t["passed"]:
            print(f'  + {t["name"] or t["stdin"]!r}')
            continue
        if t["timed_out"]:
            print(f'  - {t["name"]}: таймаут {RUN_TIMEOUT}с (висит?)')
        elif t["exec_error"]:
            print(f'  - {t["name"]}: exec-ошибка {t["exec_error"]}')
        else:
            mark = t["signal"] or ""
            print(f'  - {t["name"]}: exit got/охот.{t["got_exit"]}/{t["expected_exit"]} {mark}, '
                  f'stdout got/охот. {t["got_stdout"]!r}/{t["expected_stdout"]!r}')


def main():
    ap = argparse.ArgumentParser(description="Верификатор asm-программ (x86-64, AT&T, _start, syscalls only)")
    ap.add_argument("prog_s", help="исходник prog.s")
    ap.add_argument("--tests", required=True, help="JSON-файл со списком тестов")
    ap.add_argument("--json", action="store_true", help="вывести вердикт JSON'ом")
    args = ap.parse_args()

    try:
        with open(args.prog_s, "r", encoding="utf-8") as f:
            src = f.read()
        with open(args.tests, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ошибка входа: {e}", file=sys.stderr)
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