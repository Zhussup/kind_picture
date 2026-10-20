#!/usr/bin/env python3
"""
collector.py — бесконечный сборщик корпуса для датасета по программированию.

Как это работает (кратко):
  1. Скрипт бесконечной нумерации берет номер sample (0, 1, 2, ...) и
     детерминированно превращает его в английский промпт про программирование
     (категория × тема × язык × вариант — никакого random, поэтому после
     перезапуска продолжается ровно с того же места, а повторы невозможны).
  2. Отправляет промпт в Ollama API (локальный сервер, модель beam:cloud
     через Ollama Cloud).
  3. Сохраняет ответ построчно в dset/raw/shard_NNNNN.jsonl и сразу
     "заливает" файл на диск (flush + fsync), чтобы при отключении
     питания не потерять.
  4. Обновляет state.json (чекпоинт) атомарной записью.

Программа НИКОГДА не выходит сама: при любой ошибке (нет интернета,
сервер не отвечает, модель недоступна) она ждет и пробует снова.
Скрипт можно убивать в любой момент и просто запустить заново командой run.sh.

Запуск:  bash run.sh          (фоновый режим, см. run.sh)
Разово:  python3 collector.py --once     (один sample, для проверки связи)
         python3 collector.py --dry-run 5  (показать 5 промптов без сети)
         python3 collector.py --status    (показать чекпоинт)
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
# НАСТРОЙКИ (можно менять прямо здесь)
# ---------------------------------------------------------------------------

MODEL = "beam:cloud"                # модель Ollama Cloud
HOST = "http://localhost:11434"     # адрес локального сервера Ollama
SYSTEM_PROMPT = (
    "You are an expert programmer. Answer in English. "
    "Be practical: always show working, idiomatic code in fenced code blocks "
    "and explain it clearly."
)
TEMP_RANGE = (0.6, 0.95)   # температура выбирается случайно на каждый запрос:
                           # так меньше шанс "ходить по кругу" (mode collapse)
JITTER = (0.0, 1.5)        # случайная добавка к паузе — трафик живее и меньше
                           # похож на машинную долбёжку (анти-абьюз защита)
REQUEST_TIMEOUT = 900   # секунд ожидания одного ответа (облачные модели медленные)
PAUSE = 2.0          # пауза между запросами, секунд
MAX_SHARD_BYTES = 8 * 1024 * 1024   # разрезаем корпус на файлы по ~8 МБ
MAX_BACKOFF = 3600.0  # после серии ошибок ждем не дольше 1 часа
STATE_EVERY_SECONDS = 60  # обновлять метки времени в чекпоинте не чаще чем раз в минуту

# Пути — всё внутри папки dset/, относительно самого скрипта.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")          # сюда падают ответы (по кускам)
STATE_PATH = os.path.join(BASE_DIR, "state.json")  # чекпоинт
LOG_PATH = os.path.join(BASE_DIR, "collector.log")

# ---------------------------------------------------------------------------
# ГЕНЕРАЦИЯ ПРОМПТОВ
# ---------------------------------------------------------------------------

# Темы — общие для всех языков программирования (на английском).
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
TOPICS = sorted(set(TOPICS))  # убираем дубли и фиксируем порядок

LANGS = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++", "Java",
    "C#", "Kotlin", "Swift", "Ruby", "PHP", "Bash (shell script)", "SQL",
]

# Варианты формулировки — добавляют разнообразие, не меняя смысл.
VARIATIONS = [
    "Give a thorough answer.", "Keep it concise but complete.",
    "Focus on best practices.", "Focus on common mistakes to avoid.",
    "Include a real-world example.", "Explain as if teaching a beginner.",
    "Explain for an experienced developer.", "Add short test examples.",
    "Mention edge cases.", "Show how to do it wrong first, then right.",
]

# ---------------------------------------------------------------------------
# Категории: (имя, функция(тема, язык, вариант) -> текст промпта)
# Порядок в списках фиксированный — он важен для детерминизма чекпоинта.

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

# Порядок в списках фиксированный — он важен для детерминизма чекпоинта.
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
COMBO = NCAT * NTOP * NLANG * NVAR   # сколько уникальных промптов существует


def prompt_for(index):
    """Детерминированно превращает номер sample в (категория, язык, промпт).

    Деление с остатком раскладывает index по слоям: категория меняется чаще
    всего, затем тема, затем язык, затем вариант формулировки. Один и тот же
    index всегда даёт один и тот же промпт — поэтому возобновление после
    перезапуска работает без повторов и пропусков.
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
# ЧЕКПОИНТ
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_state():
    """Читает чекпоинт. Если файла нет (первый запуск) — стартуем с нуля."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": 1,
        "prompt_index": 0,     # следующий sample, который надо сгенерировать
        "shard_num": 1,        # номер текущего файла с данными (raw/shard_N.jsonl)
        "samples_ok": 0,
        "samples_failed": 0,
        "consecutive_errors": 0,
        "last_error": None,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }


def save_state(state):
    """Атомарная запись чекпоинта: пишем во временный файл и переименовываем.

    Rename поверх существующего файла на Linux — одна атомарная операция,
    поэтому даже при отключении питания state.json окажется либо старым,
    либо новым, но никогда не битым.
    """
    state["updated_at"] = now_iso()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # принудительно сбрасываем данные на диск
    os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------------------
# ЗАПИСЬ ОТВЕТОВ
# ---------------------------------------------------------------------------

def shard_path(shard_num):
    return os.path.join(RAW_DIR, f"shard_{shard_num:05d}.jsonl")


def repair_last_line(path):
    """Если последний кусок файла не заканчивается \\n (обрыв при выключении
    питания между записью и fsync) — обрезаем его до последней целой строки.
    Битая половина строки — единственно возможное повреждение JSONL."""
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
    if cut == -1:      # в хвосте вообще нет перевода строки — файл целиком битый
        keep = 0
    else:              # оставляем всё до последнего перевода строки включительно
        keep = size - (len(data) - (cut + 1))
    with open(path, "r+b") as f:
        f.truncate(keep)


def append_sample(shard_num, record):
    """Добавляет одну запись в текущий шард. Возвращает размер файла после записи."""
    path = shard_path(shard_num)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())  # принудительный сброс на диск: данных не теряем
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# ЗАПРОС К OLLAMA
# ---------------------------------------------------------------------------

def validate_python_blocks(response):
    """Проверяет синтаксис ВСЕХ code-блоков ответа (```python) через ast.parse —
    модуль Python, который разбирает исходник в синтаксическое дерево: если текст
    не разбирается, код заведомо битый. Возвращает (все_ли_блоки_ок, первая_ошибка)."""
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", response, re.S)
    for i, code in enumerate(blocks, 1):
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"блок {i}: {e.msg} (строка {e.lineno})"
    return True, None


def ask_ollama(prompt, temperature):
    """Один запрос к /api/chat. Возвращает (текст ответа, количество токенов).

    При проблеме — кидает исключение, вызывающий код решает, что делать.
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
        raise RuntimeError(f"пустой ответ от модели: keys={list(data.keys())}")
    return content, data.get("eval_count", 0)


# ---------------------------------------------------------------------------
# ОСНОВНОЙ ЦИКЛ
# ---------------------------------------------------------------------------

def log(msg):
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def do_one(index, state):
    """Генерирует и сохраняет один sample. Возвращает True при успехе."""
    category, lang, prompt = prompt_for(index)
    temperature = random.uniform(*TEMP_RANGE)
    log(f"#{index} [{category}/{lang}] запрос (t={temperature:.2f})...")
    try:
        response, tokens = ask_ollama(prompt, temperature)
    except requests.HTTPError as e:
        state["samples_failed"] += 1
        state["consecutive_errors"] += 1
        err = "?"
        if e.response is not None:
            # 429 Too Many Requests / 503: если сервер прислала Retry-After —
            # сколько секунд он просит подождать, учтём это в цикле ниже
            err = f"HTTP {e.response.status_code}"
            ra = e.response.headers.get("retry-after")
            if ra:
                state["min_backoff"] = float(ra)
                err += f", Retry-After={float(ra):.0f}с"
        state["last_error"] = err
        log(f"#{index} ОШИБКА: {err}")
        return False
    except Exception as e:
        state["samples_failed"] += 1
        state["consecutive_errors"] += 1
        state["last_error"] = f"{type(e).__name__}: {e}"
        log(f"#{index} ОШИБКА: {state['last_error']}")
        return False

    # Проверка синтаксиса: только для Python (для других языков парсера под рукой нет).
    # Не выбрасываем, а помечаем — решение "хранить/фильтровать" отложено в build_dataset.
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
    state["min_backoff"] = None    # сервер больше не просит ждать — сбрасываем
    if size >= MAX_SHARD_BYTES:   # файл вырос — начинаем следующий
        state["shard_num"] += 1
    save_state(state)
    code_note = "код ok" if code_ok else f"КОД БИТЫЙ ({code_err})"
    log(f"#{index} OK ({tokens} токенов, t={temperature:.2f}, {code_note}, "
        f"сохранено в shard_{saved_shard:05d}.jsonl)")
    return True


def run_forever(state):
    log(f"=== старт/возобновление: продолжаю с sample #{state['prompt_index']}, "
        f"комбинаций промптов всего {COMBO} ===")
    backoff = 5.0
    last_state_ts = time.time()
    while True:
        if do_one(state["prompt_index"], state):
            backoff = 5.0
            wait = PAUSE + random.uniform(*JITTER)
        else:
            # экспоненциальный backoff: 5с, 10с, 20с ... до 1 часа;
            # если сервер ответил Retry-After — ждём не меньше его
            backoff = min(backoff * 2, MAX_BACKOFF)
            wait = max(backoff, state.get("min_backoff") or 0)
            log(f"жду {wait:.0f}с перед повтором")
        if time.time() - last_state_ts > STATE_EVERY_SECONDS:
            save_state(state)
            last_state_ts = time.time()
        time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Сборщик корпуса через Ollama Cloud")
    parser.add_argument("--once", action="store_true",
                        help="сделать ровно один sample и выйти (проверка связи)")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="показать первые N промптов и выйти (без сети)")
    parser.add_argument("--status", action="store_true", help="показать чекпоинт и выйти")
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

    # Аккуратное завершение по Ctrl+C и по `kill` (systemd и т.п.): чекпоинт и так
    # сохраняется после каждого sample, поэтому тут достаточно просто выйти.
    def on_term(_sig, _frm):
        log("получен сигнал остановки — выхожу")
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)

    try:
        if args.once:
            ok = do_one(state["prompt_index"], state)
            save_state(state)
            sys.exit(0 if ok else 1)
        run_forever(state)
    except KeyboardInterrupt:
        log("остановлено (Ctrl+C) — прогресс сохранён")


if __name__ == "__main__":
    main()