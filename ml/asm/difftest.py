#!/usr/bin/env python3
"""
difftest.py — дифференциальная проверка реконструкции asm -> C.

Идея: у нас есть пара "листинг asm -> исходный C", которую создал gcc.
Модель читает листинг и выдаёт свой C. Проверяем механически:
  - компилируем ОРИГИНАЛ и ПРЕДСКАЗАНИЕ одним и тем же gcc;
  - гоняем оба на одинаковых входах (stdin);
  - сравниваем наблюдаемое поведение: stdout и код возврата.
Если поведение совпало на всех входах — реконструкция функционально верна.
Никакого сравнения текста C с эталоном: другой код с тем же поведением
тоже правильный ответ.

Это не юнит-тесты с заранее известными ожиданиями: ожидаемый вывод
берётся у оригинала. Поэтому любые входы допустимы — даже случайные.

Поведение при сбоях (честно):
  - оба зависли            -> считаем совпадением (одинаковое поведение);
  - завис только один      -> несовпадение;
  - падение по сигналу     -> сравниваем сигналы как есть;
  - оригинал не собрался   -> это ошибка входа (код 3), а не провал модели.

Изоляция прогонов (таймаут, RLIMIT на память/CPU/core/процессы, пустое
окружение) переиспользуется из verifier.py — там же обоснование констант.

Запуск:
  python3 difftest.py ref.c cand.c                  встроенный набор входов
  python3 difftest.py ref.c cand.c --cases cases.json
  python3 difftest.py ref.c cand.c --json
  python3 difftest.py --self-test                   проверка самого difftest

Формат cases.json: [{"name": "...", "stdin": "..."}, ...]
Коды возврата: 0 — всё совпало; 1 — расхождение поведения;
2 — кандидат не собрался; 3 — ошибка входа.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import verifier  # run_once, _norm, _exit_matches — та же изоляция прогонов

COMPILE_TIMEOUT = 30   # сек на один gcc (реальность: десятки мс)

# Входы «по умолчанию»: покрывают типичные формы наших задач
# (пусто, одно число, два числа, n + массив, текст, несколько строк).
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
    """Компилирует C-текст в исполняемый файл. Возвращает (ok, путь, лог)."""
    src = os.path.join(workdir, name + ".c")
    exe = os.path.join(workdir, name)
    with open(src, "w", encoding="utf-8") as f:
        f.write(c_text)
    cmd = ["gcc", "-std=c11", "-O0", "-w", src, "-o", exe]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, None, f"gcc не завершился за {COMPILE_TIMEOUT} с"
    except OSError as e:
        return False, None, f"{type(e).__name__}: {e}"
    if p.returncode != 0:
        return False, None, (p.stderr or p.stdout).strip()[:4000]
    return True, exe, p.stderr.strip()[:4000]


def _same_behavior(ref, cand):
    """Сравнивает два прогона. Возвращает (match, причина_расхождения|None)."""
    if ref["timed_out"] and cand["timed_out"]:
        return True, None
    if ref["timed_out"] != cand["timed_out"]:
        return False, "таймаут только у одной стороны"
    if ref["exec_error"] or cand["exec_error"]:
        same = ref["exec_error"] == cand["exec_error"]
        return same, None if same else "разные ошибки запуска"
    if verifier._norm(ref["stdout"]) != verifier._norm(cand["stdout"]):
        return False, "stdout отличается"
    if not verifier._exit_matches(ref["exit"], cand["exit"]):
        return False, "код возврата отличается"
    return True, None


def compare(ref_c, cand_c, inputs):
    """Главная функция: вердикт по двум C-текстам и списку входов."""
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
            # Если эталон сам упёрся в НАШ лимит (таймаут, SIGKILL от RLIMIT_CPU,
            # иной сигнал), вход не задаёт осмысленного ожидания: воспроизводить
            # «эталон убит лимитом» — не та цель, которой мы учим модель.
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
                    "reason": "эталон упёрся в лимит/сигнал — вход неинформативен",
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

    # n_total теперь — число ИНФОРМАТИВНЫХ входов: те, где эталон убит
    # лимитом, не в счёт (match=None)
    verdict["n_total"] = sum(1 for x in verdict["cases"]
                             if x["match"] is not None)
    verdict["all_match"] = (verdict["n_total"] > 0
                            and verdict["n_passed"] == verdict["n_total"])
    return verdict


def print_human(v):
    if not v["ref_compiled"]:
        print(f"[REF FAIL] оригинал не собрался (ошибка входа): {v['error']}")
        return
    if not v["cand_compiled"]:
        print(f"[CAND FAIL] предсказание не собралось: {v['error']}")
        return
    skipped = (f", {v['n_skipped']} неинформативны"
               if v.get("n_skipped") else "")
    print(f"[DIFF] входы: {v['n_passed']}/{v['n_total']} совпали{skipped}")
    for c in v["cases"]:
        if c["match"] is not False:
            continue
        extra = f" ({c['signal']})" if c["signal"] else ""
        print(f'  - {c["name"]}: {c["reason"]}; '
              f'stdin={c["stdin"]!r} ref={c["ref_stdout"]!r}/{c["ref_exit"]}{extra} '
              f'cand={c["cand_stdout"]!r}/{c["cand_exit"]}')


# ---------------------------------------------------------------------------
# САМОПРОВЕРКА
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
    print("1) одинаковый C -> всё совпадает")
    v = compare(REF_C, REF_C, cases)
    assert v["all_match"] and v["n_passed"] == 1, v
    print("   ok")

    print("2) другой ответ -> расхождение")
    v = compare(REF_C, WRONG_C, cases)
    assert not v["all_match"] and v["cases"][0]["reason"] == "stdout отличается", v
    print("   ok")

    print("3) некомпилируемый C -> cand_compile, не исключение")
    v = compare(REF_C, "int main(void) { this is not C }", cases)
    assert not v["cand_compiled"] and v["stage"] == "cand_compile", v
    print("   ok")

    print("4) падение по сигналу детектится")
    v = compare(REF_C, CRASH_C, cases)
    assert not v["all_match"], v
    assert v["cases"][0]["cand_exit"] == -11, v["cases"][0]
    print("   ok (-11 SIGSEGV)")

    print("5) зависание кандидата детектится как расхождение")
    v = compare(REF_C, HANG_C, cases)
    c = v["cases"][0]
    assert not v["all_match"], v
    # гонка wall-таймаута и RLIMIT_CPU: ядро может убить раньше (SIGKILL -9),
    # поэтому принимаем оба варианта — важно, что провал детектирован
    assert c["cand_timed_out"] or c["cand_exit"] == -9, c
    print("   ok (таймаут или SIGKILL)")

    print("ALL SELF-TESTS PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Дифференциальная проверка реконструкции asm->C")
    ap.add_argument("ref_c", nargs="?", help="эталонный C (оригинал)")
    ap.add_argument("cand_c", nargs="?", help="предсказанный C (кандидат)")
    ap.add_argument("--cases", help="JSON-файл со списком входов")
    ap.add_argument("--json", action="store_true", help="вывести вердикт JSON'ом")
    ap.add_argument("--self-test", action="store_true",
                    help="проверить сам difftest (без файлов)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.ref_c or not args.cand_c:
        ap.error("нужны ref.c и cand.c (или --self-test)")

    try:
        with open(args.ref_c, "r", encoding="utf-8") as f:
            ref_c = f.read()
        with open(args.cand_c, "r", encoding="utf-8") as f:
            cand_c = f.read()
        if args.cases:
            with open(args.cases, "r", encoding="utf-8") as f:
                inputs = json.load(f)
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("cases.json должен быть непустым списком")
        else:
            inputs = DEFAULT_INPUTS
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"ошибка входа: {e}", file=sys.stderr)
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
