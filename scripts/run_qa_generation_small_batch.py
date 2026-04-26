#!/usr/bin/env python3
"""
Small-batch QA generation: process a fixed number of chunks per batch.
Use for incremental runs instead of processing all chunks at once.
"""

import json
import jsonlines
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent

def create_batch_chunks(batch_size, batch_num, skip_processed=True):
    """Create a chunk file for the given batch."""
    unprocessed_file = PROJECT_ROOT / "data/processed_corpus/chunks_unprocessed.jsonl"
    batch_file = PROJECT_ROOT / f"data/processed_corpus/chunks_batch_{batch_num}.jsonl"
    if not unprocessed_file.exists():
        print(f"Unprocessed chunks file not found: {unprocessed_file}")
        print("Run first: python scripts/continue_qa_generation.py")
        return None
    processed_ids = set()
    if skip_processed:
        qa_file = PROJECT_ROOT / "data/instructions/qa_pairs_complete_v3.jsonl"
        if qa_file.exists():
            with jsonlines.open(qa_file) as reader:
                for obj in reader:
                    if 'chunk_id' in obj:
                        processed_ids.add(obj['chunk_id'])
                    elif 'metadata' in obj and 'chunk_id' in obj['metadata']:
                        processed_ids.add(obj['metadata']['chunk_id'])
    
    print(f"Reading unprocessed chunks: {unprocessed_file}")
    print(f"Already processed: {len(processed_ids)} chunks")
    start_idx = (batch_num - 1) * batch_size
    end_idx = start_idx + batch_size
    print(f"Creating batch {batch_num}: chunks [{start_idx}:{end_idx}]")
    batch_chunks = []
    skipped = 0
    with jsonlines.open(unprocessed_file) as reader:
        for i, chunk in enumerate(tqdm(reader, desc="Scanning chunks")):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            chunk_id = chunk.get('chunk_id', '')
            if skip_processed and chunk_id in processed_ids:
                skipped += 1
                continue
            
            batch_chunks.append(chunk)
    
    if not batch_chunks:
        print(f"Batch {batch_num} has no unprocessed chunks.")
        return None
    with jsonlines.open(batch_file, mode='w') as writer:
        for chunk in batch_chunks:
            writer.write(chunk)
    print(f"Batch file created: {batch_file}")
    print(f"Batch size: {len(batch_chunks)} chunks")
    if skipped > 0:
        print(f"Skipped (already processed): {skipped} chunks")
    print(f"Estimated time: {len(batch_chunks) * 20 / 3600:.1f} h")
    
    return batch_file

def update_notebook_for_batch(batch_file, batch_num):
    """Update notebook to use the batch file."""
    notebook_file = PROJECT_ROOT / "notebooks/02_InstructionGeneration_v3_continue.ipynb"
    output_file = PROJECT_ROOT / f"notebooks/02_InstructionGeneration_v3_batch_{batch_num}.ipynb"
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'INPUT_FILE = Path' in source:
                new_source = []
                for line in cell['source']:
                    if 'INPUT_FILE = Path' in line:
                        new_line = f'INPUT_FILE = Path("{batch_file}")\n'
                        new_source.append(new_line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                break
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Batch notebook created: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Small-batch QA generation')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Chunks per batch (default: 10000)')
    parser.add_argument('--batch-num', type=int, default=1,
                        help='Batch number (default: 1)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Do not skip already processed chunks')
    parser.add_argument('--create-only', action='store_true',
                        help='Only create batch file, do not update notebook')
    args = parser.parse_args()
    print("=" * 80)
    print("Small-batch QA generation")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Batch number: {args.batch_num}")
    print(f"Skip processed: {not args.no_skip}")
    print()
    batch_file = create_batch_chunks(
        batch_size=args.batch_size,
        batch_num=args.batch_num,
        skip_processed=not args.no_skip
    )
    if not batch_file:
        print("Batch creation failed.")
        return
    print()
    if not args.create_only:
        notebook_file = update_notebook_for_batch(batch_file, args.batch_num)
        print()
        print("=" * 80)
        print("Batch ready.")
        print("=" * 80)
        print(f"Batch file: {batch_file}")
        print(f"Notebook: {notebook_file}")
        print()
        print("Next: Open the notebook in Jupyter and run all cells, or run with papermill.")
    else:
        print()
        print("=" * 80)
        print("Batch file created.")
        print("=" * 80)
        print(f"Batch file: {batch_file}")

if __name__ == "__main__":
    main()
