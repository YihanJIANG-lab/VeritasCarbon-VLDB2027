#!/usr/bin/env python3
"""
Continue QA generation: process remaining chunks.
Resume from checkpoint; only chunks without QA pairs are processed.
"""

import json
import jsonlines
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def load_processed_chunk_ids(qa_file_path, checkpoint_file_path):
    """Load set of already processed chunk_ids from QA file and optional checkpoint."""
    processed_ids = set()
    if Path(qa_file_path).exists():
        print(f"Reading existing QA pairs: {qa_file_path}")
        with jsonlines.open(qa_file_path) as reader:
            for obj in reader:
                if 'chunk_id' in obj:
                    processed_ids.add(obj['chunk_id'])
                elif 'metadata' in obj and 'chunk_id' in obj['metadata']:
                    processed_ids.add(obj['metadata']['chunk_id'])
    if Path(checkpoint_file_path).exists():
        print(f"Reading checkpoint: {checkpoint_file_path}")
        with open(checkpoint_file_path, 'r') as f:
            checkpoint = json.load(f)
            if 'processed_chunk_ids' in checkpoint:
                processed_ids.update(checkpoint['processed_chunk_ids'])
    print(f"Processed chunks: {len(processed_ids)}")
    return processed_ids

def filter_unprocessed_chunks(chunks_file_path, processed_ids):
    """Return chunks that have no QA pair yet."""
    unprocessed_chunks = []
    total_chunks = 0
    print(f"Reading all chunks: {chunks_file_path}")
    with jsonlines.open(chunks_file_path) as reader:
        for chunk in tqdm(reader, desc="Scanning chunks"):
            total_chunks += 1
            chunk_id = chunk.get('chunk_id', '')
            if chunk_id and chunk_id not in processed_ids:
                unprocessed_chunks.append(chunk)
    print(f"Total chunks: {total_chunks}")
    print(f"Unprocessed: {len(unprocessed_chunks)}")
    print(f"Progress: {len(processed_ids)}/{total_chunks} ({len(processed_ids)/total_chunks*100:.2f}%)")
    return unprocessed_chunks

def save_unprocessed_chunks(unprocessed_chunks, output_path):
    """Save unprocessed chunks to file."""
    print(f"Saving unprocessed chunks to: {output_path}")
    with jsonlines.open(output_path, mode='w') as writer:
        for chunk in unprocessed_chunks:
            writer.write(chunk)
    print(f"Saved {len(unprocessed_chunks)} unprocessed chunks.")

def main():
    chunks_file = PROJECT_ROOT / "data/processed_corpus/chunks_clean_fixed.jsonl"
    qa_file = PROJECT_ROOT / "data/instructions/qa_pairs_complete_v3.jsonl"
    checkpoint_file = PROJECT_ROOT / "data/instructions/.checkpoint_complete_v3.json"
    unprocessed_file = PROJECT_ROOT / "data/processed_corpus/chunks_unprocessed.jsonl"
    
    print("=" * 80)
    print("Continue QA generation – identify unprocessed chunks")
    print("=" * 80)
    print()
    processed_ids = load_processed_chunk_ids(qa_file, checkpoint_file)
    print()
    unprocessed_chunks = filter_unprocessed_chunks(chunks_file, processed_ids)
    print()
    if unprocessed_chunks:
        save_unprocessed_chunks(unprocessed_chunks, unprocessed_file)
        print()
        print("=" * 80)
        print("Unprocessed chunks saved.")
        print("=" * 80)
        print(f"File: {unprocessed_file}")
        print(f"Count: {len(unprocessed_chunks)}")
        print(f"Estimated time: {len(unprocessed_chunks) * 20 / 3600:.1f} h")
        print()
        print("Next: Use chunks_unprocessed.jsonl as input in the notebook and run; output appends to qa_pairs_complete_v3.jsonl")
    else:
        print("=" * 80)
        print("All chunks already processed.")
        print("=" * 80)

if __name__ == "__main__":
    main()
