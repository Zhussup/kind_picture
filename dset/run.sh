#!/usr/bin/env bash
# Запуск сборщика в фоне (24/7) с автоперезапуском, если процесс всё же упал.
# Всё живёт только внутри папки dset/.
#
# Запуск:      bash run.sh
# Остановка:   bash stop.sh
# Смотреть лог: tail -f collector.log

cd "$(dirname "$0")" || exit 1

# Уже запущено?
if [ -f collector.pid ] && kill -0 "$(cat collector.pid)" 2>/dev/null; then
    echo "Сборщик уже работает (PID $(cat collector.pid)). Сначала bash stop.sh"
    exit 1
fi

nohup bash -c '
    while true; do
        echo "[run.sh] запуск collector.py"
        python3 collector.py
        code=$?
        echo "[run.sh] collector вышел с кодом $code — перезапуск через 15с"
        # Ctrl+C / signal stop внутри collector.py выходит только по этим кодам?
        # Нет: collector никогда не выходит сам, значит код != 0 — это падение.
        sleep 15
    done
' >> collector_console.log 2>&1 &

echo $! > collector.pid
echo "Сборщик запущен (PID $(cat collector.pid))."
echo "Лог: dset/collector.log (детали) и dset/collector_console.log (всё подряд)."