#!/usr/bin/env bash
# Остановка сборщика и его цикла автоперезапуска.
# Данные и чекпоинт целы: после каждого sample уже всё сохранено на диск.

cd "$(dirname "$0")" || exit 1

# Остановить цикл авторазапуска (run.sh)
if [ -f collector.pid ]; then
    kill "$(cat collector.pid)" 2>/dev/null && echo "Цикл run.sh остановлен."
    rm -f collector.pid
fi

# Остановить сам collector, если он ещё работал
pkill -f "python3 collector.py" && echo "collector.py остановлен." || echo "collector.py не был запущен."