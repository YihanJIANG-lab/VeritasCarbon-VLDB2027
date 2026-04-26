"""
Evaluation benchmark builder helper.

Helps annotators build ESG evaluation benchmarks quickly.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class BenchmarkBuilder:
    """Evaluation benchmark builder."""
    def __init__(self, chunks_file: str, output_dir: str):
        self.chunks_file = Path(chunks_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    def prepare_qa_candidates(self, num_samples: int = 5000) -> List[Dict]:
        """Prepare QA benchmark candidates from chunks (length + ESG keyword filter)."""
        print(f"Reading chunks: {self.chunks_file}")
        chunks = []
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading chunks"):
                try:
                    chunk = json.loads(line)
                    text = chunk.get("text", "")
                    if 200 <= len(text) <= 800:
                        esg_keywords = ["环境", "社会", "治理", "ESG", "CSR", "可持续发展",
                                       "碳排放", "员工", "培训", "供应链", "合规"]
                        if any(kw in text for kw in esg_keywords):
                            chunks.append(chunk)
                except Exception:
                    continue
        print(f"Candidates: {len(chunks)}")
        if len(chunks) > num_samples:
            chunks = random.sample(chunks, num_samples)
        candidates_file = self.output_dir / "qa_candidates.jsonl"
        with open(candidates_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"Saved to: {candidates_file}. Next: annotate 2-3 questions per chunk.")
        return chunks
    def prepare_ie_candidates(self, num_samples: int = 500) -> List[Dict]:
        """Prepare IE benchmark candidates (chunks with numbers and structure)."""
        print(f"Reading chunks: {self.chunks_file}")
        chunks = []
        import re
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading chunks"):
                try:
                    chunk = json.loads(line)
                    text = chunk.get("text", "")
                    has_numbers = bool(re.search(r'\d+', text))
                    has_structure = any(marker in text for marker in ["：", "、", "；", "1.", "2."])
                    
                    if has_numbers and has_structure:
                        chunks.append(chunk)
                except Exception:
                    continue
        
        print(f"IE candidates: {len(chunks)}")
        if len(chunks) > num_samples:
            chunks = random.sample(chunks, num_samples)
        candidates_file = self.output_dir / "ie_candidates.jsonl"
        with open(candidates_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"Saved to: {candidates_file}. Next: annotate entities and relations.")
        return chunks
    def prepare_summary_candidates(self, num_samples: int = 100) -> List[str]:
        """Prepare summary benchmark candidates (unique doc_ids)."""
        print(f"Reading chunks: {self.chunks_file}")
        doc_ids = set()
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Collecting doc_id"):
                try:
                    chunk = json.loads(line)
                    doc_id = chunk.get("doc_id", "")
                    if doc_id:
                        doc_ids.add(doc_id)
                except Exception:
                    continue
        
        doc_ids = list(doc_ids)
        print(f"Unique docs: {len(doc_ids)}")
        if len(doc_ids) > num_samples:
            doc_ids = random.sample(doc_ids, num_samples)
        candidates_file = self.output_dir / "summary_candidates.json"
        with open(candidates_file, "w", encoding="utf-8") as f:
            json.dump({"doc_ids": doc_ids}, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {candidates_file}. Next: write summary per doc.")
        return doc_ids
    
    def prepare_classification_candidates(self, num_samples: int = 2000) -> List[Dict]:
        """Prepare classification benchmark candidates (chunks with empty label fields)."""
        print(f"Reading chunks: {self.chunks_file}")
        chunks = []
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading chunks"):
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except Exception:
                    continue
        print(f"Chunks: {len(chunks)}")
        if len(chunks) > num_samples:
            chunks = random.sample(chunks, num_samples)
        candidates_file = self.output_dir / "classification_candidates.jsonl"
        with open(candidates_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                chunk["esg_dimension"] = ""
                chunk["sub_category"] = ""
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"Saved to: {candidates_file}. Next: annotate ESG dimension and sub_category.")
        return chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build ESG evaluation benchmark")
    parser.add_argument("--chunks-file", type=str, required=True, help="Chunks file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--benchmark-type", type=str, choices=["qa", "ie", "summary", "classification", "all"], default="all", help="Benchmark type")
    args = parser.parse_args()
    builder = BenchmarkBuilder(args.chunks_file, args.output_dir)
    if args.benchmark_type in ("qa", "all"):
        print("\n" + "="*60 + "\nPreparing QA candidates\n" + "="*60)
        builder.prepare_qa_candidates()
    if args.benchmark_type in ("ie", "all"):
        print("\n" + "="*60 + "\nPreparing IE candidates\n" + "="*60)
        builder.prepare_ie_candidates()
    if args.benchmark_type in ("summary", "all"):
        print("\n" + "="*60 + "\nPreparing Summary candidates\n" + "="*60)
        builder.prepare_summary_candidates()
    if args.benchmark_type in ("classification", "all"):
        print("\n" + "="*60 + "\nPreparing Classification candidates\n" + "="*60)
        builder.prepare_classification_candidates()
    print("\n" + "="*60 + "\nDone. See output dir for candidate files and start annotation.")

