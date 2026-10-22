#!/bin/bash
# Supervisor for the line-A1 collector (gen_c.py).
#
# Restarts gen_c.py whenever it exits, for any reason: an uncaught exception,
# a killed process, or Ollama going away when the machine shuts down. A
# multi-hour collection must not depend on one Python process staying alive.
#
# A flock guard keeps a second copy from starting while this one is up, so
# running the script twice by accident does nothing.
#
# It lives in the repository rather than /tmp because /tmp is wiped on reboot,
# and losing the supervisor means the collector never comes back on its own.
#
# Usage:
#   ./collector.sh &                      # in the foreground of a terminal
#   nohup setsid ./collector.sh >/dev/null 2>&1 &   # detached, survives the shell

set -u
cd "$(dirname "$0")" || exit 1

exec 9>"${TMPDIR:-/tmp}/kind_picture_gen_c.lock"
flock -n 9 || exit 0

echo $$ > gen_c.pid
echo "[$(date -Iseconds)] supervisor started (pid $$)" >> gen_c.supervisor.log

while true; do
    python3 -u gen_c.py
    code=$?
    echo "[$(date -Iseconds)] collector exited with code $code, restarting in 15s" \
        >> gen_c.supervisor.log
    sleep 15
done
