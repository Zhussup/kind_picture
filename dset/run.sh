#!/usr/bin/env bash
# Start the collector in the background (24/7) with auto-restart if the process does crash.
# Everything lives only inside the dset/ folder.
#
# Start:       bash run.sh
# Stop:        bash stop.sh
# View log:    tail -f collector.log

cd "$(dirname "$0")" || exit 1

# Already running?
if [ -f collector.pid ] && kill -0 "$(cat collector.pid)" 2>/dev/null; then
    echo "The collector is already running (PID $(cat collector.pid)). Run bash stop.sh first"
    exit 1
fi

nohup bash -c '
    while true; do
        echo "[run.sh] starting collector.py"
        python3 collector.py
        code=$?
        echo "[run.sh] collector exited with code $code — restarting in 15s"
        # Does Ctrl+C / a stop signal inside collector.py exit only with these codes?
        # No: the collector never exits on its own, so code != 0 means a crash.
        sleep 15
    done
' >> collector_console.log 2>&1 &

echo $! > collector.pid
echo "Collector started (PID $(cat collector.pid))."
echo "Log: dset/collector.log (details) and dset/collector_console.log (everything)."