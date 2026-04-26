"""
Quick check of chunk token-length distribution.
Use to decide if data needs re-processing.
"""

import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Please install transformers: pip install transformers")
    sys.exit(1)


def check_chunk_tokens(chunks_file: str, sample_size: int = 1000, model_name: str = "Qwen/Qwen1.5-7B-Chat"):
    """Check token-length distribution of chunks (sample_size=0 for all)."""
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print(f"File not found: {chunks_path}")
        return
    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer load failed: {e}")
        print("Tip: set HF_ENDPOINT for mirror if needed.")
        return
    print(f"Reading chunks: {chunks_path}")
    token_lengths = []
    max_tokens = 0
    over_2048_count = 0
    over_1024_count = 0
    total_chunks = 0
    problem_chunks = []
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            total_chunks += 1
            if sample_size > 0 and total_chunks > sample_size:
                break
            
            try:
                chunk = json.loads(line)
                text = chunk.get("text", "")
                
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
                
                token_lengths.append(token_count)
                max_tokens = max(max_tokens, token_count)
                
                if token_count > 2048:
                    over_2048_count += 1
                    problem_chunks.append({
                        "chunk_id": chunk.get("chunk_id", "unknown"),
                        "tokens": token_count,
                        "chars": len(text)
                    })
                elif token_count > 1024:
                    over_1024_count += 1
                    
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
    if not token_lengths:
        print("No valid chunks found.")
        return
    print("\n" + "="*60)
    print("Token length distribution")
    print("="*60)
    print(f"Chunks checked: {len(token_lengths):,}")
    if sample_size > 0 and total_chunks > sample_size:
        print(f"Sampled first {sample_size:,} (total {total_chunks:,})")
    print(f"\nStats: mean={np.mean(token_lengths):.1f} median={np.median(token_lengths):.1f} min={np.min(token_lengths)} max={max_tokens} std={np.std(token_lengths):.1f}")
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        print(f"  {p}%: {value:.1f} tokens")
    print(f"\nRanges:")
    ranges = [
        (0, 256, "0-256"),
        (256, 512, "256-512"),
        (512, 1024, "512-1024"),
        (1024, 2048, "1024-2048"),
        (2048, float('inf'), ">2048")
    ]
    for min_val, max_val, label in ranges:
        count = sum(1 for x in token_lengths if min_val <= x < max_val)
        percentage = count / len(token_lengths) * 100
        print(f"  {label:12s}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\nOver limit: >1024: {over_1024_count:,} ({over_1024_count/len(token_lengths)*100:.1f}%)  >2048: {over_2048_count:,} ({over_2048_count/len(token_lengths)*100:.1f}%)")
    if over_2048_count > 0:
        print(f"\n{over_2048_count} chunks exceed 2048 tokens. First 10:")
        for i, pc in enumerate(problem_chunks[:10], 1):
            print(f"  {i}. {pc['chunk_id']}: {pc['tokens']} tokens ({pc['chars']} chars)")
    else:
        print("\nAll chunks within 2048 tokens.")
    print("\n" + "="*60)
    print("Recommendation")
    print("="*60)
    if over_2048_count == 0:
        print("No re-processing needed. All within limit; use dynamic batch in training.")
    elif over_2048_count < 10:
        print("Few over limit; consider truncation in training or fix those chunks.")
    else:
        print("Many over limit; check chunking logic or re-process.")
    if np.mean(token_lengths) < 400:
        print("Low mean length; consider larger chunk_size; current data is usable.")
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check chunk token-length distribution")
    parser.add_argument("--chunks-file", type=str, default="data/processed_corpus/chunks_clean.jsonl", help="Chunks file path")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size (0=all)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat", help="Model name for tokenizer")
    args = parser.parse_args()
    chunks_file = args.chunks_file
    if not Path(chunks_file).is_absolute():
        chunks_file = project_root / chunks_file
    
    check_chunk_tokens(str(chunks_file), args.sample_size, args.model)

