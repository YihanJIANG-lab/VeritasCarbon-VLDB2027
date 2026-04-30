"""
Feature Vector Extraction for Layered Expert Selection — Reference Implementation

Implements the rule-based feature extractor used in Section 3.3 of:
  "VeritasCarbon: Scalable and Traceable Instruction Data Generation for ESG Domains"

Given a chunk c_i, VeritasCarbon computes a feature vector describing its content
(numerical evidence, temporal references, standards language, etc.). Expert
activation proceeds layer-by-layer:

  Layer 1 (Base):         QA, Summary, Extraction, Classification, Analysis
  Layer 2 (Analysis):     Temporal Analysis, Benchmark Comparison, Greenwashing Detection
  Layer 3 (Verification): Consistency Verification, Standard Alignment
  Layer 4 (Graph):        Knowledge Graph

Each expert defines a scoring function s_i(chunk) ∈ [0,1]. Experts with
s_i >= threshold (default 0.5) are selected. If none meets the threshold,
the fallback QAExpert is assigned.

Usage:
    extractor = FeatureExtractor()
    features = extractor.extract(chunk_text)
    selected, scores = extractor.select_experts(chunk_text)
"""

import re
from typing import Dict, List, Tuple


class FeatureExtractor:
    """Extract content features from ESG text chunks for expert selection."""

    # ── Feature dimensions ─────────────────────────────────────────────

    # Carbon/environment keywords
    CARBON_KEYWORDS = [
        "碳", "排放", "emission", "carbon", "CO2", "温室气体", "GHG",
        "能源", "energy", "可再生", "renewable", "减排", "中和",
    ]

    # Social keywords
    SOCIAL_KEYWORDS = [
        "员工", "employee", "培训", "training", "社区", "community",
        "安全", "safety", "健康", "health", "多样性", "diversity",
    ]

    # Governance keywords
    GOVERNANCE_KEYWORDS = [
        "治理", "governance", "合规", "compliance", "董事会", "board",
        "审计", "audit", "风险", "risk", "反腐", "corruption",
    ]

    # Standards/frameworks
    STANDARD_KEYWORDS = [
        "GRI", "TCFD", "SASB", "ISO", "CDP", "SDG",
        "标准", "指引", "框架", "规范", "合规",
    ]

    # Vague commitment patterns
    VAGUE_COMMITMENT_KEYWORDS = [
        "致力于", "计划", "将", "预计", "目标", "承诺", "努力", "争取",
        "committed to", "plan to", "aim to", "intend to",
    ]

    # Comparative language
    COMPARATIVE_KEYWORDS = [
        "对比", "比较", "差异", "优于", "高于", "低于", "相比",
        "higher than", "lower than", "compared to", "ranked",
    ]

    # Temporal markers
    TEMPORAL_KEYWORDS = [
        "年", "月", "季度", "年度", "去年", "今年", "明年",
        "趋势", "变化", "增长", "下降", "since", "by 20",
    ]

    # Industry/benchmark
    INDUSTRY_KEYWORDS = [
        "行业", "同行", "同类", "平均水平", "基准", "标准", "领先", "排名",
        "industry", "benchmark", "average", "peer",
    ]

    def extract(self, chunk_text: str) -> Dict[str, float]:
        """
        Extract feature vector from chunk text.

        Returns:
            Dict mapping feature name to score in [0, 1].
        """
        text = chunk_text.lower()
        return {
            "length": min(len(chunk_text) / 2000.0, 1.0),
            "numerical_density": self._numerical_density(chunk_text),
            "temporal_markers": self._keyword_score(text, self.TEMPORAL_KEYWORDS, 5),
            "year_mentions": self._year_score(chunk_text),
            "standards_language": self._keyword_score(text, self.STANDARD_KEYWORDS, 5),
            "carbon_keywords": self._keyword_score(text, self.CARBON_KEYWORDS, 6),
            "social_keywords": self._keyword_score(text, self.SOCIAL_KEYWORDS, 4),
            "governance_keywords": self._keyword_score(text, self.GOVERNANCE_KEYWORDS, 4),
            "vague_commitments": self._keyword_score(text, self.VAGUE_COMMITMENT_KEYWORDS, 4),
            "comparative_language": self._keyword_score(text, self.COMPARATIVE_KEYWORDS, 3),
            "industry_keywords": self._keyword_score(text, self.INDUSTRY_KEYWORDS, 4),
            "structured_data": self._structured_score(chunk_text),
            "entity_density": self._entity_density(chunk_text),
            "verifiable_claims": self._verifiable_claims_score(chunk_text),
        }

    # ── Expert scoring functions s_i(chunk) ────────────────────────────

    EXPERT_SCORING = {
        # Layer 1: Base
        "qa_expert": lambda f: f["carbon_keywords"] * 0.5 + f["numerical_density"] * 0.3 + f["length"] * 0.2,
        "summary_expert": lambda f: f["length"] * 0.5 + f["structured_data"] * 0.3 + f["carbon_keywords"] * 0.2,
        "extraction_expert": lambda f: f["structured_data"] * 0.4 + f["numerical_density"] * 0.3 + f["entity_density"] * 0.3,
        "classification_expert": lambda f: (f["carbon_keywords"] + f["social_keywords"] + f["governance_keywords"]) / 3 * 0.7 + f["entity_density"] * 0.3,
        "analysis_expert": lambda f: f["comparative_language"] * 0.6 + f["entity_density"] * 0.4,
        # Layer 2: Analysis
        "temporal_analysis_expert": lambda f: f["temporal_markers"] * 0.5 + f["year_mentions"] * 0.5,
        "benchmark_comparison_expert": lambda f: f["industry_keywords"] * 0.6 + f["comparative_language"] * 0.4,
        "greenwashing_detection_expert": lambda f: f["vague_commitments"] * 0.6 + f["verifiable_claims"] * 0.4,
        # Layer 3: Verification
        "consistency_verification_expert": lambda f: f["numerical_density"] * 0.6 + f["comparative_language"] * 0.4,
        "standard_alignment_expert": lambda f: f["standards_language"] * 0.7 + f["structured_data"] * 0.3,
        # Layer 4: Graph
        "knowledge_graph_expert": lambda f: f["entity_density"] * 0.6 + f["structured_data"] * 0.4,
    }

    LAYER_CONFIG = {
        "base": ["qa_expert", "summary_expert", "extraction_expert",
                 "classification_expert", "analysis_expert"],
        "analysis": ["temporal_analysis_expert", "benchmark_comparison_expert",
                     "greenwashing_detection_expert"],
        "verification": ["consistency_verification_expert", "standard_alignment_expert"],
        "graph": ["knowledge_graph_expert"],
    }

    def select_experts(
        self,
        chunk_text: str,
        threshold: float = 0.5,
        max_experts: int = 3,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Layered expert selection.

        Args:
            chunk_text: Input text chunk.
            threshold: Minimum score for expert activation.
            max_experts: Maximum number of experts (K).

        Returns:
            (selected_expert_ids, all_scores)
        """
        features = self.extract(chunk_text)
        scores = {
            name: func(features)
            for name, func in self.EXPERT_SCORING.items()
        }

        selected = []

        # Layer 1: always select best base expert
        base_scores = {e: scores[e] for e in self.LAYER_CONFIG["base"]}
        best_base = max(base_scores, key=base_scores.get)
        selected.append(best_base)

        # Layer 2: add analysis experts above threshold
        for expert in self.LAYER_CONFIG["analysis"]:
            if scores[expert] >= threshold and len(selected) < max_experts:
                selected.append(expert)

        # Layer 3: add verification experts above threshold
        for expert in self.LAYER_CONFIG["verification"]:
            if scores[expert] >= threshold and len(selected) < max_experts:
                selected.append(expert)

        # Layer 4: add graph expert if strong signal
        for expert in self.LAYER_CONFIG["graph"]:
            if scores[expert] >= threshold and len(selected) < max_experts:
                selected.append(expert)

        # Truncate to K
        selected = selected[:max_experts]

        # Fallback: if empty (shouldn't happen), use QA expert
        if not selected:
            selected = ["qa_expert"]

        return selected, scores

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _keyword_score(text: str, keywords: List[str], normalize_by: int) -> float:
        count = sum(1 for kw in keywords if kw.lower() in text)
        return min(count / normalize_by, 1.0)

    @staticmethod
    def _numerical_density(text: str) -> float:
        numbers = re.findall(r'\d+\.?\d*%?', text)
        return min(len(numbers) / 10.0, 1.0)

    @staticmethod
    def _year_score(text: str) -> float:
        years = set(re.findall(r'20\d{2}', text))
        return min(len(years) / 3.0, 1.0)

    @staticmethod
    def _structured_score(text: str) -> float:
        indicators = ["：", "、", "；", "（", "）", "1.", "2.", "3.", "-", "|"]
        count = sum(1 for ind in indicators if ind in text)
        return min(count / 8.0, 1.0)

    @staticmethod
    def _entity_density(text: str) -> float:
        entity_keywords = ["公司", "企业", "集团", "有限", "股份",
                           "工厂", "厂区", "基地", "项目"]
        count = sum(1 for kw in entity_keywords if kw in text)
        return min(count / 4.0, 1.0)

    @staticmethod
    def _verifiable_claims_score(text: str) -> float:
        patterns = [
            r'降低[了]?\s*\d+%',
            r'减少[了]?\s*\d+',
            r'提高[了]?\s*\d+%',
            r'达到[了]?\s*\d+',
            r'实现[了]?\s*\d+',
            r'reduced by \d+',
            r'increased by \d+',
        ]
        count = sum(1 for p in patterns if re.search(p, text))
        return min(count / 2.0, 1.0)


# ── CLI demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = FeatureExtractor()

    sample_chunk = (
        "本公司2023年度碳排放总量为125万吨CO2当量，较2022年减少8.3%。"
        "其中范围一排放45万吨(scope 1)，范围二排放80万吨(scope 2)。"
        "公司已承诺在2030年前实现碳达峰，并计划投资5亿元用于可再生能源项目。"
        "根据GRI标准和TCFD框架，公司在环境维度的信息披露覆盖率达到92%。"
    )

    print("Feature Vector:")
    features = extractor.extract(sample_chunk)
    for k, v in features.items():
        print(f"  {k:25s}: {v:.3f}")

    print("\nExpert Selection:")
    selected, scores = extractor.select_experts(sample_chunk, threshold=0.5, max_experts=3)
    print(f"  Selected: {selected}")
    print("\n  All Expert Scores:")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " ← SELECTED" if name in selected else ""
        print(f"    {name:40s}: {score:.3f}{marker}")
