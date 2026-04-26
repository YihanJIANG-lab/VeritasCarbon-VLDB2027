"""
02-01 Adaptive expert selector.

Selects the most suitable expert agent(s) dynamically based on chunk content features.
One of the core innovations of the CoDE framework.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ExpertSelector:
    """
    Adaptive expert selector.

    Selects 1-3 most suitable expert agents based on chunk features (length, topic, structure, etc.).
    """

    # Expert type definitions
    EXPERT_TYPES = {
        "qa_expert": {
            "name": "QA expert",
            "description": "Generates Q&A pairs",
            "preferred_features": ["question_keywords", "factual_content"]
        },
        "summary_expert": {
            "name": "Summary expert",
            "description": "Generates summaries",
            "preferred_features": ["long_text", "structured_content"]
        },
        "extraction_expert": {
            "name": "Extraction expert",
            "description": "Information extraction",
            "preferred_features": ["structured_data", "numbers", "entities"]
        },
        "classification_expert": {
            "name": "Classification expert",
            "description": "Classification tasks",
            "preferred_features": ["categorical_content", "esg_keywords"]
        },
        "analysis_expert": {
            "name": "Analysis expert",
            "description": "Comparative analysis",
            "preferred_features": ["comparative_content", "multiple_entities"]
        },
        "temporal_analysis_expert": {
            "name": "Temporal analysis expert",
            "description": "Cross-year comparison and trend analysis",
            "preferred_features": ["temporal_keywords", "year_mentions", "trend_indicators"]
        },
        "benchmark_comparison_expert": {
            "name": "Benchmark comparison expert",
            "description": "Industry benchmark comparison",
            "preferred_features": ["industry_keywords", "comparative_keywords", "benchmark_indicators"]
        },
        "greenwashing_detection_expert": {
            "name": "Greenwashing detection expert",
            "description": "Identifies greenwashing patterns",
            "preferred_features": ["vague_commitments", "promise_keywords", "verification_indicators"]
        },
        "standard_alignment_expert": {
            "name": "Standard alignment expert",
            "description": "Identifies ESG standards and assesses compliance",
            "preferred_features": ["standard_keywords", "compliance_indicators", "structured_content"]
        },
        "knowledge_graph_expert": {
            "name": "Knowledge graph expert",
            "description": "Extracts entities and relations",
            "preferred_features": ["entities", "relationship_indicators", "structured_data"]
        },
        "consistency_verification_expert": {
            "name": "Consistency verification expert",
            "description": "Checks data consistency",
            "preferred_features": ["numbers", "data_keywords", "comparative_keywords"]
        },
        "promise_performance_verification_expert": {
            "name": "Promise-performance verification expert",
            "description": "Identifies verifiable claims and generates tool calls (Phase 3, not used in Phase 2)",
            "preferred_features": ["verifiable_claims", "quantitative_statements", "performance_keywords"]
        }
    }
    
    def __init__(
        self,
        use_ml_classifier: bool = True,
        classifier_model: str = "bert-base-chinese",
        min_experts: int = 1,
        max_experts: int = 3,
        use_layered_selection: bool = True,
        layered_config: Optional[Dict] = None
    ):
        """
        Initialize the expert selector.

        Args:
            use_ml_classifier: Whether to use an ML classifier (otherwise rule-based).
            classifier_model: Classifier model name.
            min_experts: Minimum number of experts to select.
            max_experts: Maximum number of experts to select.
            use_layered_selection: Whether to use layered selection.
            layered_config: Layered selection config.
        """
        self.use_ml_classifier = use_ml_classifier
        self.classifier_model = classifier_model
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.use_layered_selection = use_layered_selection
        
        if layered_config is None:
            layered_config = {}
        self.base_layer_required = layered_config.get("base_layer_required", True)
        self.analysis_layer_threshold = layered_config.get("analysis_layer_threshold", 0.3)
        self.verification_layer_threshold = layered_config.get("verification_layer_threshold", 0.3)
        self.graph_layer_threshold = layered_config.get("graph_layer_threshold", 0.5)
        
        self.base_layer_experts = [
            "qa_expert", "summary_expert", "extraction_expert",
            "classification_expert", "analysis_expert"
        ]
        self.analysis_layer_experts = [
            "temporal_analysis_expert",
            "benchmark_comparison_expert",
            "greenwashing_detection_expert"
        ]
        self.verification_layer_experts = [
            "consistency_verification_expert",
            "standard_alignment_expert"
        ]
        self.graph_layer_experts = [
            "knowledge_graph_expert"
        ]
        
        self.classifier = None
        if use_ml_classifier:
            self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the ML classifier."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading classifier model: {self.classifier_model}")
            self.classifier = None  # TODO: load or train classifier
            logger.warning("Classifier not trained; using rule-based selection")
            self.use_ml_classifier = False
        except ImportError:
            logger.warning("transformers not installed; using rule-based selection")
            self.use_ml_classifier = False
    
    def extract_features(self, chunk_text: str) -> Dict[str, float]:
        """
        Extract features from chunk text.

        Args:
            chunk_text: Chunk text content.

        Returns:
            Feature dictionary.
        """
        features = {
            "length": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "has_numbers": self._has_numbers(chunk_text),
            "has_entities": self._has_entities(chunk_text),
            "has_question_keywords": self._has_question_keywords(chunk_text),
            "has_structured_data": self._has_structured_data(chunk_text),
            "esg_keyword_density": self._esg_keyword_density(chunk_text),
            "comparative_keywords": self._has_comparative_keywords(chunk_text),
            "temporal_keywords": self._has_temporal_keywords(chunk_text),
            "year_mentions": self._has_year_mentions(chunk_text),
            "industry_keywords": self._has_industry_keywords(chunk_text),
            "vague_commitments": self._has_vague_commitments(chunk_text),
            "standard_keywords": self._has_standard_keywords(chunk_text),
            "verifiable_claims": self._has_verifiable_claims(chunk_text),
            "performance_keywords": self._has_performance_keywords(chunk_text)
        }
        return features
    
    def _has_numbers(self, text: str) -> float:
        """Check if text contains numbers."""
        import re
        numbers = re.findall(r'\d+', text)
        return min(len(numbers) / 10.0, 1.0)  # normalize to 0-1
    
    def _has_entities(self, text: str) -> float:
        """Check for entities (company names, metrics, etc.) via ESG keywords."""
        entity_keywords = ["公司", "企业", "碳排放", "员工", "培训", "环保", "治理"]
        count = sum(1 for kw in entity_keywords if kw in text)
        return min(count / 5.0, 1.0)
    
    def _has_question_keywords(self, text: str) -> float:
        """Check for question-like keywords."""
        question_keywords = ["什么", "如何", "为什么", "哪些", "多少", "是否"]
        count = sum(1 for kw in question_keywords if kw in text)
        return min(count / 3.0, 1.0)
    
    def _has_structured_data(self, text: str) -> float:
        """Check for structured data (lists, tables, etc.)."""
        structured_indicators = ["：", "、", "；", "（", "）", "1.", "2.", "-"]
        count = sum(1 for ind in structured_indicators if ind in text)
        return min(count / 10.0, 1.0)
    
    def _esg_keyword_density(self, text: str) -> float:
        """Compute ESG keyword density."""
        esg_keywords = [
            "环境", "社会", "治理", "ESG", "CSR", "可持续发展",
            "碳排放", "环保", "社会责任", "公司治理", "员工", "培训",
            "供应链", "合规", "风险", "创新", "质量", "安全"
        ]
        text_lower = text.lower()
        count = sum(1 for kw in esg_keywords if kw in text_lower)
        return min(count / len(esg_keywords) * 2, 1.0)  # normalize
    
    def _has_comparative_keywords(self, text: str) -> float:
        """Check for comparative keywords."""
        comparative_keywords = ["对比", "比较", "差异", "优于", "高于", "低于", "相比"]
        count = sum(1 for kw in comparative_keywords if kw in text)
        return min(count / 3.0, 1.0)
    
    def _has_temporal_keywords(self, text: str) -> float:
        """Check for temporal keywords."""
        temporal_keywords = ["年", "月", "季度", "年度", "去年", "今年", "明年", "趋势", "变化", "增长", "下降"]
        count = sum(1 for kw in temporal_keywords if kw in text)
        return min(count / 5.0, 1.0)
    
    def _has_year_mentions(self, text: str) -> float:
        """Check for year mentions (2000-2099)."""
        import re
        years = re.findall(r'20\d{2}', text)
        return min(len(set(years)) / 3.0, 1.0)
    
    def _has_industry_keywords(self, text: str) -> float:
        """Check for industry/benchmark keywords."""
        industry_keywords = ["行业", "同行", "同类", "平均水平", "基准", "标准", "领先", "排名"]
        count = sum(1 for kw in industry_keywords if kw in text)
        return min(count / 4.0, 1.0)
    
    def _has_vague_commitments(self, text: str) -> float:
        """Check for vague commitment keywords."""
        vague_keywords = ["致力于", "计划", "将", "预计", "目标", "承诺", "努力", "争取"]
        count = sum(1 for kw in vague_keywords if kw in text)
        return min(count / 4.0, 1.0)
    
    def _has_standard_keywords(self, text: str) -> float:
        """Check for standard/framework keywords."""
        standard_keywords = ["GRI", "TCFD", "SASB", "ISO", "CDP", "标准", "指引", "框架", "规范", "合规"]
        count = sum(1 for kw in standard_keywords if kw in text)
        return min(count / 5.0, 1.0)
    
    def _has_verifiable_claims(self, text: str) -> float:
        """Check for verifiable/quantitative claims (e.g. 'reduced by 15%')."""
        import re
        quantitative_patterns = [
            r'降低[了]?\s*\d+%',
            r'减少[了]?\s*\d+',
            r'提高[了]?\s*\d+%',
            r'达到[了]?\s*\d+',
            r'实现[了]?\s*\d+'
        ]
        count = sum(1 for pattern in quantitative_patterns if re.search(pattern, text))
        return min(count / 2.0, 1.0)
    
    def _has_performance_keywords(self, text: str) -> float:
        """Check for performance/result keywords."""
        performance_keywords = ["效果", "成效", "成果", "成绩", "表现", "绩效", "达成", "完成", "实现"]
        count = sum(1 for kw in performance_keywords if kw in text)
        return min(count / 4.0, 1.0)
    
    def select_experts_rule_based(self, features: Dict[str, float]) -> List[str]:
        """
        Rule-based expert selection (when ML classifier is not used).

        Args:
            features: Chunk feature dict.

        Returns:
            List of selected expert IDs.
        """
        expert_scores = {}
        expert_scores["qa_expert"] = (
            features["has_question_keywords"] * 0.5 +
            features["esg_keyword_density"] * 0.5
        )
        
        expert_scores["summary_expert"] = (
            (1.0 if features["length"] > 300 else features["length"] / 300) * 0.5 +
            features["has_structured_data"] * 0.5
        )
        
        expert_scores["extraction_expert"] = (
            features["has_structured_data"] * 0.4 +
            features["has_numbers"] * 0.3 +
            features["has_entities"] * 0.3
        )
        
        expert_scores["classification_expert"] = (
            features["esg_keyword_density"] * 0.7 +
            features["has_entities"] * 0.3
        )
        
        expert_scores["analysis_expert"] = (
            features["comparative_keywords"] * 0.6 +
            features["has_entities"] * 0.4
        )
        
        expert_scores["temporal_analysis_expert"] = (
            features["temporal_keywords"] * 0.5 +
            features["year_mentions"] * 0.5
        )
        
        expert_scores["benchmark_comparison_expert"] = (
            features["industry_keywords"] * 0.6 +
            features["comparative_keywords"] * 0.4
        )
        
        expert_scores["greenwashing_detection_expert"] = (
            features["vague_commitments"] * 0.6 +
            features["verifiable_claims"] * 0.4
        )
        
        expert_scores["standard_alignment_expert"] = (
            features["standard_keywords"] * 0.7 +
            features["has_structured_data"] * 0.3
        )
        
        expert_scores["knowledge_graph_expert"] = (
            features["has_entities"] * 0.6 +
            features["has_structured_data"] * 0.4
        )
        
        expert_scores["consistency_verification_expert"] = (
            features["has_numbers"] * 0.6 +
            features["comparative_keywords"] * 0.4
        )
        
        # promise_performance_verification_expert: Phase 3 only
        # expert_scores["promise_performance_verification_expert"] = (
        #     features["verifiable_claims"] * 0.7 +
        #     features["performance_keywords"] * 0.3
        # )
        
        sorted_experts = sorted(
            expert_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected = [
            expert for expert, score in sorted_experts
            if score > 0.3
        ][:self.max_experts]
        
        if len(selected) < self.min_experts:
            selected = [expert for expert, _ in sorted_experts[:self.min_experts]]
        
        return selected
    
    def select_experts_ml_based(self, chunk_text: str) -> List[str]:
        """
        Select experts using ML classifier.

        Args:
            chunk_text: Chunk text.

        Returns:
            List of selected expert IDs.
        """
        # TODO: implement ML classifier; currently falls back to rule-based
        features = self.extract_features(chunk_text)
        return self.select_experts_rule_based(features)
    
    def select_experts_layered(self, chunk_text: str) -> List[str]:
        """
        Layered expert selection: base (>=1) -> analysis (0-2) -> verification (0-1) -> graph (optional).

        Args:
            chunk_text: Chunk text.

        Returns:
            List of selected expert IDs.
        """
        selected = []
        features = self.extract_features(chunk_text)

        if self.base_layer_required:
            base_scores = {}
            for expert in self.base_layer_experts:
                if expert in self.EXPERT_TYPES:
                    if expert == "qa_expert":
                        base_scores[expert] = (
                            features["has_question_keywords"] * 0.5 +
                            features["esg_keyword_density"] * 0.5
                        )
                    elif expert == "summary_expert":
                        base_scores[expert] = (
                            (1.0 if features["length"] > 300 else features["length"] / 300) * 0.5 +
                            features["has_structured_data"] * 0.5
                        )
                    elif expert == "extraction_expert":
                        base_scores[expert] = (
                            features["has_structured_data"] * 0.4 +
                            features["has_numbers"] * 0.3 +
                            features["has_entities"] * 0.3
                        )
                    elif expert == "classification_expert":
                        base_scores[expert] = (
                            features["esg_keyword_density"] * 0.7 +
                            features["has_entities"] * 0.3
                        )
                    elif expert == "analysis_expert":
                        base_scores[expert] = (
                            features["comparative_keywords"] * 0.6 +
                            features["has_entities"] * 0.4
                        )
            
            if base_scores:
                best_base = max(base_scores.items(), key=lambda x: x[1])
                selected.append(best_base[0])
                logger.debug(f"Base layer: {best_base[0]} (score: {best_base[1]:.2f})")

        analysis_selected = []
        if features["temporal_keywords"] > self.analysis_layer_threshold:
            analysis_selected.append("temporal_analysis_expert")
        if features["industry_keywords"] > self.analysis_layer_threshold:
            analysis_selected.append("benchmark_comparison_expert")
        if features["vague_commitments"] > self.analysis_layer_threshold:
            analysis_selected.append("greenwashing_detection_expert")
        
        selected.extend(analysis_selected[:2])
        if analysis_selected:
            logger.debug(f"Analysis layer: {analysis_selected[:2]}")

        if features["has_numbers"] > self.verification_layer_threshold:
            selected.append("consistency_verification_expert")
            logger.debug("Verification layer: consistency_verification_expert")
        elif features["standard_keywords"] > self.verification_layer_threshold:
            selected.append("standard_alignment_expert")
            logger.debug("Verification layer: standard_alignment_expert")

        if (features["has_entities"] > self.graph_layer_threshold and 
            features["has_structured_data"] > 0.3):
            selected.append("knowledge_graph_expert")
            logger.debug("Graph layer: knowledge_graph_expert")

        selected = selected[:self.max_experts]
        if len(selected) < self.min_experts and self.base_layer_experts:
            for expert in self.base_layer_experts:
                if expert not in selected:
                    selected.append(expert)
                    if len(selected) >= self.min_experts:
                        break
        
        return selected
    
    def select_experts(self, chunk_text: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Select experts (main entry).

        Args:
            chunk_text: Chunk text.

        Returns:
            (selected expert IDs, reason dict).
        """
        if self.use_layered_selection:
            selected = self.select_experts_layered(chunk_text)
            method = "layered"
        elif self.use_ml_classifier and self.classifier is not None:
            selected = self.select_experts_ml_based(chunk_text)
            method = "ml_based"
        else:
            features = self.extract_features(chunk_text)
            selected = self.select_experts_rule_based(features)
            method = "rule_based"
        
        features = self.extract_features(chunk_text)
        reasons = {
            "method": method,
            "features": features,
            "selected_count": len(selected)
        }
        
        logger.debug(f"Selected {len(selected)} experts for chunk: {selected}")
        
        return selected, reasons

