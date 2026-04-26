#!/bin/bash
# Batch QA generation - process remaining chunks via papermill in background

set -e

PROJECT_ROOT="/hpc2hdd/home/yjiang909/Veritas/VeritasCarbon_ACL"
NOTEBOOK_DIR="${PROJECT_ROOT}/notebooks"
INPUT_NOTEBOOK="${NOTEBOOK_DIR}/02_InstructionGeneration_v3_continue.ipynb"
OUTPUT_NOTEBOOK="${NOTEBOOK_DIR}/02_InstructionGeneration_v3_continue_output.ipynb"
LOG_FILE="${PROJECT_ROOT}/logs/qa_generation_continue.log"

mkdir -p "${PROJECT_ROOT}/logs"

echo "=========================================="
echo "Batch QA generation - continue remaining chunks"
echo "=========================================="
echo ""
echo "Input notebook: ${INPUT_NOTEBOOK}"
echo "Output notebook: ${OUTPUT_NOTEBOOK}"
echo "Log file: ${LOG_FILE}"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate VeritasCarbon

echo "Environment: VeritasCarbon"
echo ""

if ! command -v papermill &> /dev/null; then
    echo "papermill not found, installing..."
    pip install papermill -q
    echo "papermill installed"
fi

if [ ! -f "${INPUT_NOTEBOOK}" ]; then
    echo "Input notebook not found: ${INPUT_NOTEBOOK}"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/data/processed_corpus/chunks_unprocessed.jsonl" ]; then
    echo "Unprocessed chunks file not found. Run: python scripts/continue_qa_generation.py"
    exit 1
fi

echo "=========================================="
echo "Starting notebook (background)"
echo "=========================================="
echo ""
echo "Process runs in background. Use tail -f ${LOG_FILE} for progress, kill <PID> to stop."
echo "Estimated total time: ~2900h. Recommend batching 10k-50k chunks per run."
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

nohup papermill \
    "${INPUT_NOTEBOOK}" \
    "${OUTPUT_NOTEBOOK}" \
    --log-output \
    --progress-bar \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Notebook started in background. PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "  tail -f ${LOG_FILE}"
echo "  kill ${PID}"
echo ""
