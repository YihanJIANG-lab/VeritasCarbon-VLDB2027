#!/bin/bash
# QA generation script (Qwen2-72B-Instruct or config model)
set -e

MODEL_PATH="/root/autodl-fs/models/Qwen2-72B-Instruct"
CONFIG_PATH="configs/config.yaml"
INPUT_FILE="data/processed/chunks.json"
OUTPUT_FILE="results/qa_pairs/qa_pairs_qwen72b.json"
DEVICE="auto"

echo "=========================================="
echo "QA generation"
echo "=========================================="
echo ""
echo "Config:"
echo "  Model: $MODEL_PATH"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Device: $DEVICE"
echo ""

source /root/miniconda3/bin/activate qwen_unsloth
cd /root/autodl-fs/Veritas/VeritasCarbon_ACL

python -m src.instruction_generation.qa_generator_02_10 \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --use_coe \
    --device "$DEVICE"

echo ""
echo "QA generation done."
echo ""
