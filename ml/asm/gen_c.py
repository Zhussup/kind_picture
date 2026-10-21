#!/usr/bin/env python3
"""
gen_c.py — сборщик НАСТОЯЩИХ пар "C → asm" для датасета (линия A1).

Идея (и почему это надёжнее, чем просить модель писать asm):
  beam пишет C-программу, а компилятор gcc сам превращает её в asm.
  Значит asm в паре — гарантированно корректный, "золотой" эталон:
  его не надо проверять тестами, его создал настоящий компилятор.
  Модель позже будет учиться именно такому asm (AT&T, x86-64).

Что делает скрипт, по шагам:
  1. Номер sample (0, 1, 2, ...) детерминированно превращается в задачу
     (задача × вариант формулировки; никакого random — после перезапуска
     продолжаем ровно с того же места).
  2. Промпт уходит в Ollama API (модель beam:cloud), ответ парсится:
     нужен ровно один блок ```c ... ```.
  3. Код компилируется ДВА раза: `gcc -O0 -S` и `gcc -O2 -S` — получаем
     два листинга одной и той же программы (важно: видно, как оптимизатор
     меняет код — это ценная часть датасета).
  4. Одна строка JSONL на sample: C-исходник + оба листинга + пометки
     "собралось/не собралось". Пишется с flush+fsync (переживёт отключение).
  5. Чекпоинт state_gen_c.json обновляется атомарно.

Формат A1 — это пары "C-источник → листинг компилятора". Тесты здесь
НЕ прогоняются: метка качества — только "компилируется ли". Пары формата A2
("задача → программа _start + прогон тестов через verifier.py") сюда
специально не смешиваются — это отдельная линия B.

Из листинга вырезаются три служебных вещи (для читаемости датасета):
  - `.file "prog.c"` — имя исходника, нам не нужно;
  - `.ident "GCC: ..."` — версия компилятора;
  - `.section .note...` — пустые служебные секции для линкера.
Всё остальное (включая секции .text/.data, метки, директивы) сохраняется.
Флаг `-fno-asynchronous-unwind-tables` убирает шум .cfi-директив ещё на
этапе компиляции, `-fverbose-asm` добавляет комментарии с именами переменных.

Запуск:
  python3 gen_c.py --once        один sample (проверка связи)
  python3 gen_c.py --dry-run 5   показать 5 промптов без сети
  python3 gen_c.py --status      показать чекпоинт
  python3 gen_c.py --self-test   офлайн-проверка gcc-части (без сети)
  python3 gen_c.py               бесконечный сбор (Ctrl+C — остановка)
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
# НАСТРОЙКИ
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ml/asm
RAW_DIR = os.path.join(BASE_DIR, "raw_c")               # сюда пишутся шарды
STATE_PATH = os.path.join(BASE_DIR, "state_gen_c.json") # чекпоинт
LOG_PATH = os.path.join(BASE_DIR, "gen_c.log")          # лог

MODEL = "beam:cloud"
HOST = "http://localhost:11434"     # локальный демон Ollama (он держит логин)

SYSTEM_PROMPT = (
    "You are an expert C programmer. Reply with exactly one fenced code block "
    "that starts with ```c and contains the complete program. "
    "No explanations, no prose outside the block."
)
TEMP_RANGE = (0.6, 0.95)   # случайная температура на запрос — меньше повторов
JITTER = (0.0, 1.5)        # случайная добавка к паузе (анти-долбёжка)
REQUEST_TIMEOUT = 900      # сек на один ответ модели
PAUSE = 2.0                # пауза между запросами, сек
MAX_SHARD_BYTES = 8 * 1024 * 1024   # новый шард после ~8 МБ
MAX_BACKOFF = 3600.0       # максимум ожидания после серии сетевых ошибок
STATE_EVERY_SECONDS = 60   # как часто обновлять чекпоинт "на всякий случай"

GCC_TIMEOUT = 30           # сек на один запуск gcc (реальность: миллисекунды)
MAX_EXTRACT_FAILS = 3      # столько раз пробуем один промпт, потом идём дальше

GCC_FLAGS = [
    "gcc", "-std=c11", "-Wall", "-Wextra", "-pedantic",
    "-fno-asynchronous-unwind-tables",   # без .cfi-шума в листинге
    "-fverbose-asm",                     # комментарии с именами переменных
]

# ---------------------------------------------------------------------------
# ПРОСТРАНСТВО ЗАДАЧ
# ---------------------------------------------------------------------------
# Каждая строка — продолжение фразы "Write a complete C program that ...".
# Промпт детерминированный: index -> (задача, вариант формулировки).
# После исчерпания COMBO промпты повторяются, но ответы будут другими
# (температура случайна). Задачи легко дописывать — просто добавь строку.

TASKS = [
    # --- числа ---
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
    # --- строки ---
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
    # --- массивы ---
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
    # --- указатели, структуры, память ---
    "reads n then n integers into a malloc'd array that grows by doubling, prints the sum and frees everything",
    "defines struct Point { int x; int y; }, reads two points, and prints their squared distance",
    "defines struct Complex { double re; double im; }, reads two complex numbers, and prints their product",
    "builds a singly linked list from n integers read from stdin and prints it reversed using an explicit stack array (no recursion)",
    "builds a singly linked list from n integers, reverses the list in place by relinking pointers, and prints it",
    "reads n then n integers into a binary search tree and prints them in-order using an iterative traversal with an explicit stack",
    "implements a fixed-size int stack on an array of 100 elements driven by commands: p number (push), o (pop and print), q (quit)",
    "implements an int queue with a ring buffer of 100 elements driven by commands: e number (enqueue), d (dequeue and print), q (quit)",
    "reads one line and stores it in a singly linked list of characters, prints it back, and frees every node",
    # --- алгоритмы ---
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
# ПРОМПТЫ
# ---------------------------------------------------------------------------

def prompt_for(index):
    """index -> (задача, вариант, текст промпта). Детерминированно."""
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
    """Достаёт C-код из ответа: сначала ищем блок ```c, потом любой ```.
    Если блоков несколько — берём самый длинный. Нет блоков — None."""
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
    """Вырезает служебные строки (см. docstring модуля) и шапку компилятора.

    Шапка — ведущие строки-комментарии `# GNU C11 ...`, `# options passed ...`:
    это не asm, а метаданные; версию gcc пишем отдельно в поле "gcc".
    Комментарии `# prog.c:6: ...` ВНУТРИ кода остаются — это полезная
    привязка каждой инструкции к строке исходника."""
    kept = []
    for ln in text.splitlines():
        # сравнение без учёта отступов: gcc пишет "\t.section\t.note..."
        norm = ln.lstrip().replace("\t", " ")
        if norm.startswith(DROP_PREFIXES):
            continue
        kept.append(ln)
    while kept and (not kept[0].strip() or kept[0].startswith("#")):
        kept.pop(0)
    return "\n".join(kept)


def compile_c(c_text, opt):
    """Компилирует C-текст в листинг asm с уровнем оптимизации opt ('O0'/'O2').

    Возвращает dict: ok (bool), asm (str|None), diag (текст stderr, усечён).
    Диагностика нужна не для отбраковки, а как данные: видно, что модель
    написала код с предупреждениями — это тоже сигнал качества.
    """
    with tempfile.TemporaryDirectory(prefix="genc_") as workdir:
        src = os.path.join(workdir, "prog.c")
        out = os.path.join(workdir, "prog.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(c_text)
        # имена передаём относительные, cwd=workdir: иначе gcc вписывает в
        # комментарии абсолютный путь /tmp/genc_XXXX/prog.c — случайный мусор
        cmd = GCC_FLAGS + [f"-{opt}", "-S", "prog.c", "-o", "prog.s"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=GCC_TIMEOUT, cwd=workdir)
        except subprocess.TimeoutExpired:
            return {"ok": False, "asm": None,
                    "diag": f"gcc не завершился за {GCC_TIMEOUT} с"}
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
                    "diag": f"не прочитал листинг: {e}"}
        return {"ok": True, "asm": listing, "diag": p.stderr.strip()[:4000]}


# ---------------------------------------------------------------------------
# ЧЕКПОИНТ И ФАЙЛЫ
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def gcc_version():
    """Версия gcc — для провенанса датасета (шапка листинга теперь удаляется)."""
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
        "prompt_index": 0,       # номер следующего sample
        "shard_num": 1,          # текущий файл raw_c/shard_NNNNN.jsonl
        "samples_ok": 0,
        "samples_no_c": 0,       # ответ без C-блока (после MAX_EXTRACT_FAILS)
        "samples_bad_c": 0,      # C есть, но gcc не собрал ни -O0, ни -O2
        "extract_fails": 0,      # неудачных попыток для текущего index
        "consecutive_errors": 0,
        "min_backoff": None,     # сколько сервер просил подождать (Retry-After)
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
    """Атомарная запись: tmp-файл + os.replace + fsync.
    Так на диске всегда либо старая целая версия, либо новая целая."""
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
    """После аварии последняя строка шарда может быть недописана.
    Обрезаем файл до последнего завершённого \\n."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n")
        f.truncate(0 if cut < 0 else cut + 1)


def append_sample(shard_num, record):
    """Дописывает одну строку JSONL и сбрасывает её на диск. Возвращает размер."""
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
# СЕТЬ
# ---------------------------------------------------------------------------

def ask_ollama(prompt, temperature):
    """Один запрос к /api/chat. Возвращает (текст ответа, eval_count).

    Ошибки не глотаем — их обрабатывает do_one (backoff/Retry-After)."""
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
        raise RuntimeError(f"пустой ответ от модели: keys={list(data.keys())}")
    return content, data.get("eval_count", 0)


# ---------------------------------------------------------------------------
# ОДИН SAMPLE И ЦИКЛ
# ---------------------------------------------------------------------------

def do_one(index, state):
    """Делает один sample.
    Возвращаем статус: "ok" (сохранено), "skip" (сохранено как брак,
    идём дальше), "net" (сетевая ошибка — этот же index повторим)."""
    task, variation, prompt = prompt_for(index)
    temperature = random.uniform(*TEMP_RANGE)
    log(f"#{index} запрос (t={temperature:.2f}): {task[:60]}...")
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
                err += f", Retry-After={float(ra):.0f}с"
        state["last_error"] = err
        log(f"#{index} ОШИБКА: {err}")
        return "net"
    except Exception as e:
        state["consecutive_errors"] += 1
        state["last_error"] = f"{type(e).__name__}: {e}"
        log(f"#{index} ОШИБКА: {state['last_error']}")
        return "net"

    c_code = extract_c(response)
    if c_code is None:
        state["extract_fails"] += 1
        log(f"#{index} в ответе нет C-блока "
            f"(попытка {state['extract_fails']}/{MAX_EXTRACT_FAILS})")
        if state["extract_fails"] < MAX_EXTRACT_FAILS:
            return "net"   # повторим этот же index
        # стабильно не даёт код — фиксируем брак и двигаемся дальше,
        # иначе на «плохой» задаче можно застрять навсегда
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
        log(f"#{index} брак записан (нет C-блока), идём дальше")
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
        log(f"#{index} C есть, но gcc не собрал: {o0['diag'][:200]}")
    else:
        n0 = len(o0["asm"].splitlines()) if o0["ok"] else 0
        n2 = len(o2["asm"].splitlines()) if o2["ok"] else 0
        log(f"#{index} OK ({tokens} токенов, C-строк {len(c_code.splitlines())}, "
            f"asm -O0 {n0} строк, -O2 {n2} строк)")
    state["prompt_index"] = index + 1
    save_state(state)
    return "ok"


def run_forever(state):
    log(f"=== старт/возобновление: продолжаю с sample #{state['prompt_index']}, "
        f"комбинаций промптов всего {COMBO} ===")
    backoff = 5.0
    last_state_ts = time.time()
    while True:
        status = do_one(state["prompt_index"], state)
        if status in ("ok", "skip"):
            backoff = 5.0
            wait = PAUSE + random.uniform(*JITTER)
        else:
            # сетевая ошибка: экспоненциальный backoff 5с, 10с, 20с ... до часа,
            # но не меньше Retry-After, если сервер его прислал
            backoff = min(backoff * 2, MAX_BACKOFF)
            wait = max(backoff, state.get("min_backoff") or 0)
            log(f"жду {wait:.0f}с перед повтором")
        if time.time() - last_state_ts > STATE_EVERY_SECONDS:
            save_state(state)
            last_state_ts = time.time()
        time.sleep(wait)


# ---------------------------------------------------------------------------
# САМОПРОВЕРКА (офлайн, без сети) И CLI
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
    """Проверяет только gcc-часть: компиляцию, листинг и чистку служебных строк.
    Сеть не нужна. Возвращает 0 при успехе."""
    print("1) извлечение C-блока из ответа")
    sample = "Вот код:\n```c\nint main(void){return 0;}\n```\nГотово."
    code = extract_c(sample)
    assert code == "int main(void){return 0;}", repr(code)
    assert extract_c("нет блока") is None
    print("   ok")

    print("2) gcc -O0/-O2 и чистка листинга")
    o0 = compile_c(SELF_TEST_C, "O0")
    o2 = compile_c(SELF_TEST_C, "O2")
    assert o0["ok"] and o0["asm"], o0["diag"]
    assert o2["ok"] and o2["asm"], o2["diag"]
    for name, res in (("O0", o0), ("O2", o2)):
        asm = res["asm"]
        assert ".file" not in asm, f"{name}: .file остался"
        assert ".ident" not in asm, f"{name}: .ident остался"
        assert ".note" not in asm, f"{name}: .note остался"
        assert "/tmp/" not in asm, f"{name}: утёк временный путь"
        assert not asm.startswith("#"), f"{name}: шапка компилятора осталась"
        assert asm.lstrip().startswith(".text"), f"{name}: начало не .text"
        assert "main:" in asm, f"{name}: нет метки main"
        assert "square" in asm or "imul" in asm, f"{name}: нет тела square"
        print(f"   {name}: {len(asm.splitlines())} строк листинга — ok")

    print("3) битый C -> compile fail, не исключение")
    bad = compile_c("int main(void) { this is not C }", "O0")
    assert not bad["ok"] and bad["diag"], bad
    print("   ok")

    print("4) промпты детерминированы и покрывают пространство")
    a = prompt_for(0)
    b = prompt_for(0)
    assert a == b
    assert prompt_for(len(TASKS)) != a          # следующий вариант
    assert prompt_for(COMBO) == a               # цикл замкнулся
    print(f"   ok (задач {len(TASKS)}, вариантов {len(VARIATIONS)}, "
          f"комбинаций {COMBO})")

    print("ALL SELF-TESTS PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Линия A1: пары C -> gcc-листинг (O0 и O2) через beam")
    ap.add_argument("--once", action="store_true",
                    help="сделать ровно один sample и выйти (проверка связи)")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="показать первые N промптов и выйти (без сети)")
    ap.add_argument("--status", action="store_true", help="показать чекпоинт")
    ap.add_argument("--self-test", action="store_true",
                    help="офлайн-проверка gcc-части, без сети")
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
        print(f"статус: {status}")
        return

    while True:
        try:
            run_forever(state)
        except KeyboardInterrupt:
            save_state(state)
            log("остановлено вручную (Ctrl+C)")
            return


if __name__ == "__main__":
    main()
