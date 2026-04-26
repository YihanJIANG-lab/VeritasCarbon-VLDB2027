#!/usr/bin/env python3
"""
Select chunks by year order: prefer 2024, then 2023, 2022...
Focus on layer-2 corpus (CSR reports).
"""

import jsonlines
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).parent.parent

def extract_year_from_chunk_id(chunk_id):
    """Extract year from chunk_id."""
    patterns = [
        r'_(\d{4})_',
        r'(\d{4})年',
        r'\((\d{4})\)',
        r'_(\d{4})\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, chunk_id)
        if match:
            year = int(match.group(1))
            if 2000 <= year <= 2024:
                return year
    return None

def load_processed_chunk_ids(qa_file):
    """Load set of already processed chunk_ids from QA file."""
    processed_ids = set()
    if qa_file.exists():
        with jsonlines.open(qa_file) as reader:
            for obj in reader:
                if 'chunk_id' in obj:
                    processed_ids.add(obj['chunk_id'])
                elif 'metadata' in obj and 'chunk_id' in obj['metadata']:
                    processed_ids.add(obj['metadata']['chunk_id'])
    return processed_ids

def select_chunks_by_year(chunks_file, processed_ids, target_count=30000, start_year=2024):
    """Select chunks in year order (newest first)."""
    print(f"Reading chunks: {chunks_file}")
    chunks_by_year = defaultdict(list)
    layer2_total = 0
    with jsonlines.open(chunks_file) as reader:
        for chunk in tqdm(reader, desc="Scanning chunks"):
            chunk_id = chunk.get('chunk_id', '')
            if chunk_id in processed_ids:
                continue
            if not chunk_id.startswith('Layer2'):
                continue
            layer2_total += 1
            year = extract_year_from_chunk_id(chunk_id)
            if year:
                chunks_by_year[year].append(chunk)
    print(f"\nLayer-2 unprocessed total: {layer2_total:,}")
    print(f"With recognizable year: {sum(len(v) for v in chunks_by_year.values()):,}")
    print("\nYear distribution:")
    print("-" * 60)
    for year in sorted(chunks_by_year.keys(), reverse=True):
        count = len(chunks_by_year[year])
        print(f"  {year}: {count:8,} chunks")
    selected_chunks = []
    year_selection = defaultdict(int)
    print(f"\nSelecting {target_count:,} chunks by year:")
    print("-" * 60)
    for year in sorted(chunks_by_year.keys(), reverse=True):
        available = chunks_by_year[year]
        needed = target_count - len(selected_chunks)
        if needed <= 0:
            break
        to_select = min(len(available), needed)
        selected_chunks.extend(available[:to_select])
        year_selection[year] = to_select
        print(f"  {year}: selected {to_select:8,} (available: {len(available):8,})")
    print("-" * 60)
    print(f"Total selected: {len(selected_chunks):,} chunks")
    return selected_chunks, year_selection

def main():
    chunks_file = PROJECT_ROOT / "data/processed_corpus/chunks_clean_fixed.jsonl"
    qa_file = PROJECT_ROOT / "data/instructions/qa_pairs_complete_v3.jsonl"
    output_file = PROJECT_ROOT / "data/processed_corpus/chunks_next_30k_by_year.jsonl"
    print("=" * 80)
    print("Select chunks by year (layer 2, 2024 -> 2023 -> 2022 ...)")
    print("=" * 80)
    print()
    print("[1. Load processed chunks]")
    processed_ids = load_processed_chunk_ids(qa_file)
    print(f"Processed: {len(processed_ids):,} chunks")
    print()
    print("[2. Select chunks by year]")
    selected_chunks, year_selection = select_chunks_by_year(
        chunks_file, processed_ids, target_count=30000, start_year=2024
    )
    print()
    if selected_chunks:
        print("[3. Save selected chunks]")
        with jsonlines.open(output_file, mode='w') as writer:
            for chunk in selected_chunks:
                writer.write(chunk)
        print(f"Saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        print("=" * 80)
        print("Selection done.")
        print("=" * 80)
        print(f"Output: {output_file}")
        print(f"Count: {len(selected_chunks):,} chunks")
        print(f"Estimated time: {len(selected_chunks) * 20 / 3600:.1f} h")
        print()
        print("Next: Create batch notebook or use this file as input.")
        print()
        print("Year selection detail:")
        print("-" * 60)
        for year in sorted(year_selection.keys(), reverse=True):
            count = year_selection[year]
            percentage = count / len(selected_chunks) * 100
            print(f"  {year}: {count:8,} chunks ({percentage:5.2f}%)")
        print("=" * 80)
    else:
        print("No chunks selected.")

if __name__ == "__main__":
    main()
