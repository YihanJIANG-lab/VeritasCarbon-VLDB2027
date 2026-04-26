"""
03-03 Intrinsic evaluation of generated instruction data.

Computes quality metrics across CoDE output AND baselines:
  - ROUGE-L, BLEU-4 (self-consistency with source chunk)
  - Distinct-n diversity
  - Domain relevance (ESG keyword density)
  - Average length (instruction / response)
  - Structural completeness (has instruction + has response)
  - FactCheck (number/entity overlap with source)
  - Quality score distribution

Produces a comparison table (JSON + CSV) ready for the paper.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Metric computations
# =========================================================================

class IntrinsicEvaluator:
    """Evaluate instruction-data quality WITHOUT reference answers."""

    ESG_KEYWORDS = [
        "环境", "社会", "治理", "ESG", "CSR", "可持续发展",
        "碳排放", "碳中和", "碳达峰", "环保", "社会责任", "公司治理",
        "员工", "培训", "供应链", "合规", "风险", "创新", "质量", "安全",
        "能源", "水资源", "废弃物", "温室气体", "GRI", "TCFD", "SASB",
    ]

    def __init__(self):
        self.rouge_scorer = None
        self._init_rouge()

    def _init_rouge(self):
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            logger.info("ROUGE scorer loaded")
        except ImportError:
            logger.warning("rouge_score not installed; ROUGE-L will be skipped")

    # --- single-record metrics ---

    def rouge_l(self, reference: str, candidate: str) -> float:
        if not self.rouge_scorer:
            return 0.0
        try:
            return self.rouge_scorer.score(reference, candidate)["rougeL"].fmeasure
        except Exception:
            return 0.0

    @staticmethod
    def bleu4(reference: str, candidate: str) -> float:
        """Character-level BLEU-4 (suitable for Chinese)."""
        ref_chars = list(reference)
        cand_chars = list(candidate)
        if len(cand_chars) < 4 or len(ref_chars) < 4:
            return 0.0
        from collections import Counter

        def ngram_counts(tokens, n):
            return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        score = 1.0
        for n in range(1, 5):
            ref_ng = ngram_counts(ref_chars, n)
            cand_ng = ngram_counts(cand_chars, n)
            clipped = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
            total = max(sum(cand_ng.values()), 1)
            precision = clipped / total
            if precision == 0:
                return 0.0
            score *= precision

        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_chars) / max(len(cand_chars), 1)))
        return bp * (score ** 0.25)

    @staticmethod
    def distinct_n(texts: List[str], n: int = 2) -> float:
        """Distinct-n diversity across a collection of texts."""
        all_ngrams = []
        for text in texts:
            chars = list(text)
            for i in range(len(chars) - n + 1):
                all_ngrams.append(tuple(chars[i : i + n]))
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    @staticmethod
    def domain_relevance(text: str, keywords: Optional[List[str]] = None) -> float:
        if keywords is None:
            keywords = IntrinsicEvaluator.ESG_KEYWORDS
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw in text_lower)
        return min(matched / len(keywords) * 2, 1.0)

    @staticmethod
    def factcheck_overlap(candidate: str, source: str) -> float:
        """Fraction of numbers/entities in candidate that also appear in source."""
        cand_nums = set(re.findall(r"\d+", candidate))
        src_nums = set(re.findall(r"\d+", source))
        if not cand_nums:
            return 1.0  # no numbers to check
        return len(cand_nums & src_nums) / len(cand_nums)

    @staticmethod
    def structural_completeness(instruction: str, response: str) -> float:
        """1.0 if both instruction & response are non-trivial, else partial."""
        score = 0.0
        if instruction and len(instruction.strip()) >= 10:
            score += 0.5
        if response and len(response.strip()) >= 20:
            score += 0.5
        return score

    # --- batch evaluation ---

    def evaluate_dataset(self, records: List[Dict], label: str = "") -> Dict:
        """Evaluate a list of QA records and return aggregate metrics.

        Each record should have keys:
            instruction / question
            output / answer / response
            input / chunk (source text)
        """
        instructions = []
        responses = []
        sources = []
        quality_scores = []

        for r in records:
            instr = r.get("instruction") or r.get("question") or ""
            resp = r.get("output") or r.get("answer") or r.get("response") or ""
            src = r.get("input") or r.get("chunk") or ""
            instructions.append(instr)
            responses.append(resp)
            sources.append(src)

            qs = 0
            meta = r.get("metadata", {})
            if isinstance(meta, dict):
                qs = meta.get("quality_score", 0) or 0
            quality_scores.append(qs)

        n = len(records)
        if n == 0:
            return {"label": label, "n": 0}

        # Per-record metrics
        rouge_scores = []
        bleu_scores = []
        domain_scores = []
        factcheck_scores = []
        completeness_scores = []

        for instr, resp, src in zip(instructions, responses, sources):
            if src:
                rouge_scores.append(self.rouge_l(src, resp))
                bleu_scores.append(self.bleu4(src, resp))
                factcheck_scores.append(self.factcheck_overlap(resp, src))
            domain_scores.append(self.domain_relevance(resp))
            completeness_scores.append(self.structural_completeness(instr, resp))

        # Collection-level metrics
        distinct1 = self.distinct_n(responses, 1)
        distinct2 = self.distinct_n(responses, 2)
        distinct3 = self.distinct_n(responses, 3)

        avg_instr_len = np.mean([len(i) for i in instructions])
        avg_resp_len = np.mean([len(r) for r in responses])

        result = {
            "label": label,
            "n": n,
            "avg_instruction_length": round(float(avg_instr_len), 1),
            "avg_response_length": round(float(avg_resp_len), 1),
            "rouge_l": round(float(np.mean(rouge_scores)), 4) if rouge_scores else None,
            "bleu4": round(float(np.mean(bleu_scores)), 4) if bleu_scores else None,
            "distinct_1": round(float(distinct1), 4),
            "distinct_2": round(float(distinct2), 4),
            "distinct_3": round(float(distinct3), 4),
            "domain_relevance": round(float(np.mean(domain_scores)), 4),
            "factcheck": round(float(np.mean(factcheck_scores)), 4) if factcheck_scores else None,
            "structural_completeness": round(float(np.mean(completeness_scores)), 4),
            "avg_quality_score": round(float(np.mean(quality_scores)), 4) if any(q > 0 for q in quality_scores) else None,
        }

        return result


# =========================================================================
# Comparison runner
# =========================================================================

def load_jsonl(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_comparison(
    coe_files: List[str],
    baseline_dir: str = "results/baselines",
    ablation_dir: str = "results/ablation",
    output_path: str = "results/outputs/intrinsic_comparison.json",
    max_per_dataset: Optional[int] = None,
) -> Dict:
    """Evaluate CoDE data + all baselines + ablation outputs and produce a comparison table.

    Args:
        coe_files: list of CoDE QA JSONL paths
        baseline_dir: directory containing baseline result JSONL files
        ablation_dir: directory containing ablation result JSONL files
        output_path: where to save the comparison JSON
        max_per_dataset: cap records per dataset for speed

    Returns:
        dict of {label: metrics_dict}
    """
    evaluator = IntrinsicEvaluator()
    comparison = {}

    # --- CoDE data ---
    coe_records = []
    for f in coe_files:
        if Path(f).exists():
            coe_records.extend(load_jsonl(f))
    if max_per_dataset and len(coe_records) > max_per_dataset:
        import random
        coe_records = random.sample(coe_records, max_per_dataset)
    if coe_records:
        comparison["CoDE (ours)"] = evaluator.evaluate_dataset(coe_records, label="CoDE (ours)")
        logger.info(f"CoDE: {len(coe_records)} records evaluated")

    # --- Baselines ---
    baseline_path = Path(baseline_dir)
    if baseline_path.exists():
        for jsonl_file in sorted(baseline_path.glob("*_results.jsonl")):
            method = jsonl_file.stem.replace("_results", "")
            records = load_jsonl(str(jsonl_file))
            if max_per_dataset and len(records) > max_per_dataset:
                import random
                records = random.sample(records, max_per_dataset)
            if records:
                comparison[method] = evaluator.evaluate_dataset(records, label=method)
                logger.info(f"Baseline {method}: {len(records)} records evaluated")

    # --- Ablation outputs ---
    ablation_path = Path(ablation_dir)
    if ablation_path.exists():
        for jsonl_file in sorted(ablation_path.rglob("*.jsonl")):
            label = f"ablation/{jsonl_file.parent.name}/{jsonl_file.stem}"
            records = load_jsonl(str(jsonl_file))
            if max_per_dataset and len(records) > max_per_dataset:
                import random
                records = random.sample(records, max_per_dataset)
            if records:
                comparison[label] = evaluator.evaluate_dataset(records, label=label)

    # --- Save ---
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"Comparison saved: {out}")

    # Also save CSV
    csv_path = out.with_suffix(".csv")
    _save_csv(comparison, csv_path)
    logger.info(f"CSV saved: {csv_path}")

    return comparison


def _save_csv(comparison: Dict, csv_path: Path):
    """Save comparison dict as CSV table."""
    if not comparison:
        return
    all_keys = set()
    for v in comparison.values():
        all_keys.update(v.keys())
    all_keys.discard("label")
    columns = sorted(all_keys)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method," + ",".join(columns) + "\n")
        for label, metrics in comparison.items():
            row = [label]
            for col in columns:
                val = metrics.get(col, "")
                row.append(str(val) if val is not None else "")
            f.write(",".join(row) + "\n")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Intrinsic evaluation of instruction data")
    parser.add_argument("--coe_files", nargs="+",
                        default=["data/instructions/qa_pairs_complete_v3_1.5w.jsonl",
                                 "data/instructions/qa_pairs_complete_v3_2w.jsonl"])
    parser.add_argument("--baseline_dir", default="results/baselines")
    parser.add_argument("--ablation_dir", default="results/ablation")
    parser.add_argument("--output", default="results/outputs/intrinsic_comparison.json")
    parser.add_argument("--max_per_dataset", type=int, default=None)
    args = parser.parse_args()

    run_comparison(
        coe_files=args.coe_files,
        baseline_dir=args.baseline_dir,
        ablation_dir=args.ablation_dir,
        output_path=args.output,
        max_per_dataset=args.max_per_dataset,
    )
