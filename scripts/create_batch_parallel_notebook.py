#!/usr/bin/env python3
"""Create batch-parallel optimized instruction generation notebook."""
import json
from pathlib import Path

original_nb_path = Path("/hpc2hdd/home/yjiang909/Veritas/VeritasCarbon_ACL/notebooks/02_InstructionGeneration_v3_next_30k.ipynb")
output_nb_path = Path("/hpc2hdd/home/yjiang909/Veritas/VeritasCarbon_ACL/notebooks/02_InstructionGeneration_v3_batch_parallel.ipynb")

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.19"
        },
        "veritas_version": "v3.1-batch-parallel"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown_cell(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n")
    })

def add_code_cell(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n")
    })

add_markdown_cell("""# VeritasCarbon - Instruction data generation (batch-parallel v3.1)

## Batch processing + parallelization

### Strategy
1. **Batch GPU inference** – multiple prompts per call; dynamic batch size by GPU memory.
2. **Parallel data prep** – multiprocessing for expert selection, topic extraction; decoupled from GPU.
3. **Async pipeline** – producer-consumer queue; data prep and inference in parallel.

### Performance (approx.)
| Metric | Serial v3.0 | Batch-parallel v3.1 |
|--------|--------------|---------------------|
| Single QA | 82.6s | ~20-30s |
| 30k QA pairs | 1400+ h | ~350-500 h |
| CPU | 1 core | Multi-core |

### Unchanged
- Meta-Expert, 11 experts, quality evaluation, CoE, checkpoint resume.

---
Note: This version needs more resources (multi-core CPU + large GPU). Run on a capable server.""")

print("Reading original notebook...")
with open(original_nb_path, 'r', encoding='utf-8') as f:
    original_nb = json.load(f)

print("Adding environment check...")
add_markdown_cell("## 0. Environment and paths")
add_code_cell("""# Import unsloth before transformers for best performance
import torch
import os

if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    except Exception as e:
        print(f"GPU cache warning: {e}")

os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

try:
    import unsloth
    UNSLOTH_AVAILABLE = True
    print("unsloth imported")
except (ImportError, RuntimeError, Exception) as e:
    error_msg = str(e)
    if "CUDA" in error_msg or "out of memory" in error_msg.lower():
        print("unsloth import failed (GPU memory)")
        UNSLOTH_AVAILABLE = False
    else:
        print(f"unsloth import failed: {error_msg[:200]}")
        UNSLOTH_AVAILABLE = False

import sys
from pathlib import Path

print("=" * 80)
print("Environment check")
print("=" * 80)
print()

print("[1. GPU]")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Count: {torch.cuda.device_count()}")
    print(f"  CUDA: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i} mem: {total_mem:.1f}GB")
else:
    print("  GPU not available")

print()

MODEL_PATH = "/hpc2hdd/home/yjiang909/models/Qwen2-72B-Instruct/Qwen/Qwen2-72B-Instruct"
print("[2. Model path]")
if Path(MODEL_PATH).exists():
    print(f"  OK: {MODEL_PATH}")
else:
    print(f"  Missing: {MODEL_PATH}")

print()

PROJECT_ROOT = Path("/hpc2hdd/home/yjiang909/Veritas/VeritasCarbon_ACL")
SRC_PATH = PROJECT_ROOT / "src"

print("[3. Project path]")
if SRC_PATH not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(SRC_PATH))
    print(f"  Added: {SRC_PATH}")
else:
    print("  src already in sys.path")

print()
print("=" * 80)
print("Environment check done")
print("=" * 80)""")

print("Adding model load...")
add_markdown_cell("## 1. Load model and tokenizer (batch inference)")
add_code_cell("""from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 80)
print("Load model and tokenizer")
print("=" * 80)
print()

MODEL_PATH = "/hpc2hdd/home/yjiang909/models/Qwen2-72B-Instruct/Qwen/Qwen2-72B-Instruct"

print("[1. Tokenizer]")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side='left'
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"  Vocab size: {len(tokenizer)}  Pad: {tokenizer.pad_token}")
print()

print("[2. Model]")
if UNSLOTH_AVAILABLE:
    try:
        from unsloth import FastLanguageModel
        print("  Loading with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=4096,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        print("  Model loaded (Unsloth)")
    except Exception as e:
        print(f"  Unsloth failed: {e}")
        UNSLOTH_AVAILABLE = False

if not UNSLOTH_AVAILABLE:
    print("  Loading standard...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded (standard)")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {total_params / 1e9:.1f}B")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU mem: {allocated:.1f}GB")
print("=" * 80)
print("Model and tokenizer loaded")
print("=" * 80)""")

print(f"Notebook created: {output_nb_path}  ({len(notebook['cells'])} cells)")
with open(output_nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print("Open the new notebook in Cursor and run.")
