#!/bin/bash
# Monitor QA generation log in real time

PROJECT_ROOT="/hpc2hdd/home/yjiang909/Veritas/VeritasCarbon_ACL"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/qa_generation.pid"

LATEST_LOG=$(ls -t ${LOG_DIR}/qa_generation_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log file found"
    exit 1
fi

echo "=" | tee
echo "QA generation monitor" | tee
echo "=" | tee
echo "  Log: $LATEST_LOG" | tee

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "  Status: running (PID: $PID)" | tee
    else
        echo "  Status: process ended (PID: $PID)" | tee
    fi
else
    echo "  Status: no PID file" | tee
fi

echo "  Press Ctrl+C to exit" | tee
echo "=" | tee
echo ""

tail -f "$LATEST_LOG"
