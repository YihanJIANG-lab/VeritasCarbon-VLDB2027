"""
Quality Scoring Function Q(o, c) — Reference Implementation

Implements the composite quality scoring function defined in Section 2.3 of:
  "VeritasCarbon: Scalable and Traceable Instruction Data Generation for ESG Domains"

Q(o, c) = w_f * Fidelity(o,c) + w_r * Relevance(o,c) + w_g * Grounding(o,c)
         + w_s * Structure(o) + w_d * Diversity(o)

Default weights: w_f=0.25, w_r=0.25, w_g=0.20, w_s=0.15, w_d=0.15
Calibrated via grid search on a held-out validation set of 500 chunks
to maximize rank correlation with expert judgments.

Usage:
    scorer = QualityScorer()
    score = scorer.compute(output_text, source_chunk, esg_vocab)
"""

import re
from typing import List, Optional, Dict


# ── ESG domain vocabulary (default) ────────────────────────────────────
DEFAULT_ESG_VOCAB = [
    # Environment
    "碳排放", "碳中和", "碳达峰", "温室气体", "环境", "能源", "可再生",
    "排放量", "减排", "废弃物", "水资源", "生态", "气候", "污染",
    "scope 1", "scope 2", "scope 3", "carbon", "emission", "renewable",
    "GHG", "CO2", "energy", "waste", "water", "biodiversity",
    # Social
    "员工", "培训", "社会", "社区", "供应链", "安全", "健康", "多样性",
    "人权", "劳工", "社会责任", "CSR", "employee", "safety", "diversity",
    # Governance
    "治理", "合规", "风险", "董事会", "审计", "反腐", "透明度",
    "governance", "compliance", "risk", "board", "audit",
    # Standards & Frameworks
    "ESG", "GRI", "TCFD", "SASB", "CDP", "ISO 14001", "SDG",
    "可持续发展", "信息披露", "disclosure",
]


class QualityScorer:
    """Composite quality scorer for instruction-response pairs."""

    def __init__(
        self,
        w_fidelity: float = 0.25,
        w_relevance: float = 0.25,
        w_grounding: float = 0.20,
        w_structure: float = 0.15,
        w_diversity: float = 0.15,
        l_min: int = 50,
        l_trivial: int = 20,
        esg_vocab: Optional[List[str]] = None,
    ):
        """
        Args:
            w_fidelity:  Weight for Source Fidelity (ROUGE-L).
            w_relevance: Weight for Domain Relevance (ESG vocab coverage).
            w_grounding: Weight for Factual Grounding (numerical verifiability).
            w_structure: Weight for Structural Completeness.
            w_diversity: Weight for Lexical Diversity (Distinct-2/3).
            l_min:       Minimum output length for full structure score.
            l_trivial:   Minimum output length to be considered non-trivial.
            esg_vocab:   ESG domain vocabulary list; uses DEFAULT_ESG_VOCAB if None.
        """
        self.weights = {
            "fidelity": w_fidelity,
            "relevance": w_relevance,
            "grounding": w_grounding,
            "structure": w_structure,
            "diversity": w_diversity,
        }
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {sum(self.weights.values())}"
        self.l_min = l_min
        self.l_trivial = l_trivial
        self.esg_vocab = esg_vocab or DEFAULT_ESG_VOCAB

    # ── Sub-score implementations ──────────────────────────────────────

    def fidelity(self, output: str, source: str) -> float:
        """
        Source Fidelity: ROUGE-L(output, source).
        Measures recall-oriented n-gram overlap with the source chunk.
        """
        return self._rouge_l(output, source)

    def relevance(self, output: str, source: str) -> float:
        """
        Domain Relevance: |V_ESG(output) ∩ V_ESG(source)| / |V_ESG(source)|.
        Measures ESG vocabulary coverage preservation.
        """
        vocab_in_source = set()
        vocab_in_output = set()
        source_lower = source.lower()
        output_lower = output.lower()
        for term in self.esg_vocab:
            t = term.lower()
            if t in source_lower:
                vocab_in_source.add(t)
            if t in output_lower:
                vocab_in_output.add(t)
        if not vocab_in_source:
            return 1.0  # no domain terms in source → vacuously satisfied
        return len(vocab_in_output & vocab_in_source) / len(vocab_in_source)

    def grounding(self, output: str, source: str) -> float:
        """
        Factual Grounding: proportion of numerical claims in output
        that are verifiable in source.
        Extracts percentages, dates, monetary values, emission figures.
        """
        output_nums = self._extract_numbers(output)
        source_nums = self._extract_numbers(source)
        if not output_nums:
            # No numerical claims in output: return neutral score rather than
            # vacuous 1.0, penalizing outputs that avoid concrete data.
            # Scaled by structural completeness to avoid rewarding empty outputs.
            return 0.5 * self.structure(output)
        verified = sum(1 for n in output_nums if n in source_nums)
        return verified / len(output_nums)

    def structure(self, output: str) -> float:
        """
        Structural Completeness: min(1, |output|/L_min) * 𝟙[|output| > L_trivial].
        Filters trivially short or degenerate outputs.
        """
        length = len(output.strip())
        if length <= self.l_trivial:
            return 0.0
        return min(1.0, length / self.l_min)

    def diversity(self, output: str) -> float:
        """
        Lexical Diversity: (Distinct-2 + Distinct-3) / 2.
        Ratio of unique n-grams to total n-grams.
        """
        tokens = self._tokenize(output)
        d2 = self._distinct_n(tokens, 2)
        d3 = self._distinct_n(tokens, 3)
        return (d2 + d3) / 2.0

    # ── Composite score ────────────────────────────────────────────────

    def compute(self, output: str, source: str) -> float:
        """
        Compute the composite quality score Q(o, c).

        Args:
            output: Generated instruction-response text (concatenated).
            source: Source chunk text.

        Returns:
            Composite quality score in [0, 1].
        """
        scores = self.compute_breakdown(output, source)
        return scores["composite"]

    def compute_breakdown(self, output: str, source: str) -> Dict[str, float]:
        """
        Compute all sub-scores and the composite.

        Returns:
            Dict with keys: fidelity, relevance, grounding, structure,
            diversity, composite.
        """
        s = {
            "fidelity": self.fidelity(output, source),
            "relevance": self.relevance(output, source),
            "grounding": self.grounding(output, source),
            "structure": self.structure(output),
            "diversity": self.diversity(output),
        }
        s["composite"] = sum(self.weights[k] * s[k] for k in self.weights)
        return s

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple character/word tokenizer for Chinese + English mixed text."""
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)  # Chinese character as token
            elif char.isalnum():
                if tokens and tokens[-1].isascii() and tokens[-1].isalnum():
                    tokens[-1] += char
                else:
                    tokens.append(char)
            # skip whitespace and punctuation
        return tokens

    @staticmethod
    def _distinct_n(tokens: List[str], n: int) -> float:
        """Distinct-n: unique n-grams / total n-grams."""
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """Extract numerical tokens: integers, decimals, percentages."""
        return re.findall(r'\d+\.?\d*%?', text)

    @staticmethod
    def _rouge_l(candidate: str, reference: str) -> float:
        """
        ROUGE-L F1 score (LCS-based).
        Operates at token level using the simple tokenizer.
        """
        cand_tokens = QualityScorer._tokenize(candidate)
        ref_tokens = QualityScorer._tokenize(reference)
        if not cand_tokens or not ref_tokens:
            return 0.0
        lcs_len = QualityScorer._lcs_length(ref_tokens, cand_tokens)
        precision = lcs_len / len(cand_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Longest Common Subsequence length (DP)."""
        m, n = len(x), len(y)
        # Space-optimized: two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]


# ── Convenience: threshold-based filter ────────────────────────────────

def filter_by_quality(
    pairs: List[Dict],
    source_chunks: List[str],
    threshold: float = 0.5,
    scorer: Optional[QualityScorer] = None,
) -> List[Dict]:
    """
    Filter instruction-response pairs by quality threshold.

    Args:
        pairs: List of dicts with keys 'instruction', 'response'.
        source_chunks: Corresponding source chunks.
        threshold: Minimum Q(o,c) score (default τ=0.5).
        scorer: QualityScorer instance; uses defaults if None.

    Returns:
        List of pairs with Q >= threshold, each augmented with 'quality_score'.
    """
    if scorer is None:
        scorer = QualityScorer()
    accepted = []
    for pair, chunk in zip(pairs, source_chunks):
        output = pair.get("instruction", "") + " " + pair.get("response", "")
        breakdown = scorer.compute_breakdown(output, chunk)
        pair["quality_score"] = breakdown["composite"]
        pair["quality_breakdown"] = breakdown
        if breakdown["composite"] >= threshold:
            accepted.append(pair)
    return accepted


# ── CLI demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = QualityScorer()

    source = ("本公司2023年度碳排放总量为125万吨CO2当量，较2022年减少8.3%。"
              "其中范围一排放45万吨，范围二排放80万吨。公司已承诺在2030年前实现碳达峰，"
              "并计划投资5亿元用于可再生能源项目。")

    output = ("问题：该公司2023年碳排放表现如何？其减排承诺是否可信？\n"
              "回答：该公司2023年碳排放总量为125万吨CO2当量，同比下降8.3%。"
              "范围一排放45万吨，范围二排放80万吨。公司承诺2030年碳达峰，"
              "拟投资5亿元发展可再生能源。减排趋势积极，但需关注范围三排放数据缺失。")

    breakdown = scorer.compute_breakdown(output, source)
    print("Quality Score Breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\n  Threshold τ=0.5 → {'PASS' if breakdown['composite'] >= 0.5 else 'FAIL'}")
