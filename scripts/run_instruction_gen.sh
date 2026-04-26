#!/bin/bash

NOTEBOOK_IN="/root/autodl-fs/Veritas/VeritasCarbon_ACL/notebooks/02_InstructionGeneration_v2.ipynb"
NOTEBOOK_OUT="/root/autodl-fs/Veritas/VeritasCarbon_ACL/notebooks/02_InstructionGeneration_v2_output.ipynb"
LOG_FILE="/root/autodl-fs/Veritas/VeritasCarbon_ACL/notebooks/papermill_run.log"

echo "Running notebook: $NOTEBOOK_IN"
echo "Output: $NOTEBOOK_OUT"
echo "Log: $LOG_FILE"

papermill "$NOTEBOOK_IN" "$NOTEBOOK_OUT" \
    --kernel qwen_unsloth \
    --log-output \
    2>&1 | tee "$LOG_FILE"

echo "Done."
