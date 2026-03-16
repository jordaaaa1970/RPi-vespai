#!/usr/bin/env bash
# Stop the running VespAI web process (Python vespai.py --web)

set -euo pipefail

# Find the PID(s) of the running vespai.py web process
PIDS=$(pgrep -f "vespai.py --web" || true)

if [ -z "$PIDS" ]; then
    echo "No running VespAI web process found."
    exit 0
fi

echo "Stopping VespAI web process(es): $PIDS"
kill $PIDS

# Optionally, wait for processes to terminate
timeout=10
while pgrep -f "vespai.py --web" > /dev/null && [ $timeout -gt 0 ]; do
    sleep 1
    timeout=$((timeout-1))
done

if pgrep -f "vespai.py --web" > /dev/null; then
    echo "Process did not stop gracefully, sending SIGKILL."
    pkill -9 -f "vespai.py --web"
else
    echo "VespAI web process stopped."
fi
