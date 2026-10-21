#!/usr/bin/env bash
# Stop the collector and its auto-restart loop.
# Data and checkpoint are intact: everything is already saved to disk after each sample.

cd "$(dirname "$0")" || exit 1

# Stop the auto-restart loop (run.sh)
if [ -f collector.pid ]; then
    kill "$(cat collector.pid)" 2>/dev/null && echo "run.sh loop stopped."
    rm -f collector.pid
fi

# Stop the collector itself if it was still running
pkill -f "python3 collector.py" && echo "collector.py stopped." || echo "collector.py was not running."