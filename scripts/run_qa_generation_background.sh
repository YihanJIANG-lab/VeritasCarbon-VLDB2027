#!/bin/bash
# Run QA generation notebook in background via papermill

set -e

PROJECT_ROOT="/hpc2hdd/home/yjiang909/Veritas/VeritasCarbon_ACL"
NOTEBOOK_DIR="${PROJECT_ROOT}/notebooks"
LOG_DIR="${PROJECT_ROOT}/logs"

INPUT_NOTEBOOK="${NOTEBOOK_DIR}/02_InstructionGeneration_v3_next_30k.ipynb"
OUTPUT_NOTEBOOK="${NOTEBOOK_DIR}/02_InstructionGeneration_v3_next_30k_output.ipynb"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/qa_generation_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/qa_generation.pid"

if [ ! -f "$INPUT_NOTEBOOK" ]; then
    echo "Notebook not found: $INPUT_NOTEBOOK"
    exit 1
fi

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Another process already running (PID: $OLD_PID). Stop it first: kill $OLD_PID"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate VeritasCarbon

if ! command -v papermill &> /dev/null; then
    echo "papermill not found, installing..."
    pip install papermill -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 5
fi

echo "=" | tee -a "$LOG_FILE"
echo "QA generation started in background" | tee -a "$LOG_FILE"
echo "=" | tee -a "$LOG_FILE"
echo "  Input: $INPUT_NOTEBOOK" | tee -a "$LOG_FILE"
echo "  Output: $OUTPUT_NOTEBOOK" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "  $(date)" | tee -a "$LOG_FILE"
echo "=" | tee -a "$LOG_FILE"
echo ""

nohup papermill \
    "$INPUT_NOTEBOOK" \
    "$OUTPUT_NOTEBOOK" \
    --log-output \
    --log-level INFO \
    --progress-bar \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

echo "Task started in background. PID: $PID  Log: $LOG_FILE"
echo "  tail -f $LOG_FILE   ps -p $PID   kill $PID   nvidia-smi"
echo ""
