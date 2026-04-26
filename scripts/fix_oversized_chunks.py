"""
Fix chunks that exceed 2048 tokens.

Strategy:
1. Use tokenizer to find truncation point at sentence boundary
2. Truncate oversized chunks to within 2048 tokens
3. Optionally create new chunk from remainder if long enough
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("transformers not installed; using character-based estimate")


def find_sentence_boundary(text: str, max_chars: int) -> int:
    """Find sentence boundary near max_chars (supports Chinese/English punctuation)."""
    sentence_endings = ['。', '！', '？', '.', '!', '?', '\n']
    for i in range(max_chars, max(0, max_chars - 200), -1):
        if i < len(text) and text[i] in sentence_endings:
            return i + 1
    for i in range(max_chars, min(len(text), max_chars + 200)):
        if i < len(text) and text[i] in sentence_endings:
            return i + 1
    return max_chars


def truncate_chunk_by_tokens(text: str, tokenizer, max_tokens: int = 2048) -> str:
    """Truncate text to at most max_tokens using tokenizer, at sentence boundary when possible."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    estimated_chars = int(max_tokens * 1.3)
    if estimated_chars < len(text):
        boundary = find_sentence_boundary(text, estimated_chars)
        truncated_text = text[:boundary]
        truncated_tokens_check = tokenizer.encode(truncated_text, add_special_tokens=False)
        if len(truncated_tokens_check) > max_tokens:
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text


def fix_oversized_chunks(
    input_file: str,
    output_file: str,
    max_tokens: int = 2048,
    model_name: str = "Qwen/Qwen1.5-7B-Chat"
):
    """Fix chunks exceeding max_tokens; write to output_file."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return
    tokenizer = None
    if TOKENIZER_AVAILABLE:
        try:
            print(f"Loading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print("Tokenizer loaded.")
        except Exception as e:
            print(f"Tokenizer load failed: {e}")
            print("Using character-based estimate.")
    else:
        print("Using character-based estimate (transformers not installed).")
    total_chunks = 0
    fixed_chunks = 0
    oversized_chunks = []
    print(f"\nReading chunks: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, total=total_lines, desc="Processing chunks"):
            try:
                chunk = json.loads(line)
                text = chunk.get("text", "")
                total_chunks += 1
                if tokenizer:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_count = len(tokens)
                else:
                    token_count = int(len(text) * 1.5)
                if token_count > max_tokens:
                    oversized_chunks.append({
                        "chunk_id": chunk.get("chunk_id", "unknown"),
                        "original_tokens": token_count,
                        "original_chars": len(text)
                    })
                    if tokenizer:
                        fixed_text = truncate_chunk_by_tokens(text, tokenizer, max_tokens)
                    else:
                        max_chars = int(max_tokens / 1.5)
                        boundary = find_sentence_boundary(text, max_chars)
                        fixed_text = text[:boundary]
                    chunk["text"] = fixed_text
                    fixed_chunks += 1
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
    print("\n" + "="*60)
    print("Fix summary")
    print("="*60)
    print(f"Total chunks: {total_chunks:,}")
    print(f"Fixed: {fixed_chunks:,}")
    print(f"Ratio: {fixed_chunks/total_chunks*100:.2f}%")
    if oversized_chunks:
        print(f"\nFirst 10 fixed chunks:")
        for i, oc in enumerate(oversized_chunks[:10], 1):
            print(f"  {i}. {oc['chunk_id'][:80]}...")
            print(f"     Original: {oc['original_tokens']} tokens ({oc['original_chars']} chars)")
    print(f"\nDone. Input: {input_path}  Output: {output_path}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix chunks exceeding 2048 tokens")
    parser.add_argument("--input", type=str, default="data/processed_corpus/chunks_clean.jsonl", help="Input chunks file")
    parser.add_argument("--output", type=str, default="data/processed_corpus/chunks_clean_fixed.jsonl", help="Output chunks file")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max token limit")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat", help="Model name for tokenizer")
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    if not Path(input_file).is_absolute():
        input_file = project_root / input_file
    if not Path(output_file).is_absolute():
        output_file = project_root / output_file
    
    fix_oversized_chunks(str(input_file), str(output_file), args.max_tokens, args.model)

