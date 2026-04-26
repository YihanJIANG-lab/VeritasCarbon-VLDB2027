"""
02-06 Multi-metric evaluation.

ROUGE, BLEU, BERTScore, BLEURT, FactCheck, Diversity, Domain Relevance.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MultiMetricEvaluator:
    """Multi-metric evaluator for instruction/answer quality."""
    def __init__(self, metrics: Optional[List[str]] = None):
        """metrics: list of metric names; None = use all."""
        self.metrics = metrics or [
            "rouge-l", "bleu", "bertscore", "bleurt", 
            "factcheck", "diversity", "domain_relevance"
        ]
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metric backends."""
        self.rouge_scorer = None
        self.bleu_scorer = None
        self.bertscore_model = None
        self.bleurt_model = None
        self.factcheck_model = None
        if "rouge-l" in self.metrics:
            try:
                from rouge_score import rouge_scorer
                self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                logger.info("ROUGE loaded")
            except ImportError:
                logger.warning("rouge_score not installed; skipping ROUGE")
                self.metrics = [m for m in self.metrics if m != "rouge-l"]
        if "bleu" in self.metrics:
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                self.bleu_scorer = sentence_bleu
                self.bleu_smoothing = SmoothingFunction().method1
                logger.info("BLEU loaded")
            except ImportError:
                logger.warning("nltk not installed; skipping BLEU")
                self.metrics = [m for m in self.metrics if m != "bleu"]
        if "bertscore" in self.metrics:
            try:
                from bert_score import score as bert_score
                self.bertscore_model = bert_score
                logger.info("BERTScore loaded")
            except ImportError:
                logger.warning("bert_score not installed; skipping BERTScore")
                self.metrics = [m for m in self.metrics if m != "bertscore"]
        if "bleurt" in self.metrics:
            try:
                import bleurt
                self.bleurt_model = bleurt.score
                logger.info("BLEURT loaded")
            except ImportError:
                logger.warning("bleurt not installed; skipping BLEURT")
                self.metrics = [m for m in self.metrics if m != "bleurt"]
    def compute_rouge_l(self, reference: str, candidate: str) -> float:
        """Compute ROUGE-L F1."""
        if not self.rouge_scorer:
            return 0.0
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logger.warning(f"ROUGE failed: {e}")
            return 0.0
    def compute_bleu(self, reference: str, candidate: str) -> float:
        """Compute BLEU score."""
        if not self.bleu_scorer:
            return 0.0
        try:
            ref_tokens = reference.split()
            cand_tokens = candidate.split()
            score = self.bleu_scorer(
                [ref_tokens], cand_tokens, smoothing_function=self.bleu_smoothing
            )
            return score
        except Exception as e:
            logger.warning(f"BLEU failed: {e}")
            return 0.0
    def compute_bertscore(
        self, references: List[str], candidates: List[str], lang: str = "zh"
    ) -> Dict[str, float]:
        """Compute BERTScore; returns precision, recall, f1."""
        if not self.bertscore_model:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        try:
            P, R, F1 = self.bertscore_model(
                candidates, references, lang=lang, verbose=False
            )
            return {
                "precision": float(P.mean()),
                "recall": float(R.mean()),
                "f1": float(F1.mean())
            }
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    def compute_bleurt(
        self, references: List[str], candidates: List[str]
    ) -> float:
        """Compute mean BLEURT score."""
        if not self.bleurt_model:
            return 0.0
        try:
            scores = self.bleurt_model(references, candidates)
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"BLEURT failed: {e}")
            return 0.0
    def compute_factcheck_score(
        self, reference: str, candidate: str, source_text: str
    ) -> float:
        """Simple fact consistency: overlap of numbers/entities with source (0-1)."""
        import re
        candidate_numbers = set(re.findall(r'\d+', candidate))
        source_numbers = set(re.findall(r'\d+', source_text))
        candidate_entities = set([w for w in candidate.split() if len(w) > 2])
        source_entities = set([w for w in source_text.split() if len(w) > 2])
        number_overlap = len(candidate_numbers & source_numbers) / max(len(candidate_numbers), 1)
        entity_overlap = len(candidate_entities & source_entities) / max(len(candidate_entities), 1)
        score = (number_overlap * 0.5 + entity_overlap * 0.5)
        return min(score, 1.0)
    def compute_diversity(self, texts: List[str], n: int = 2) -> float:
        """Distinct-n diversity over texts."""
        if not texts:
            return 0.0
        
        all_ngrams = []
        for text in texts:
            words = text.split()
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                all_ngrams.append(ngram)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def compute_domain_relevance(
        self, text: str, esg_keywords: Optional[List[str]] = None
    ) -> float:
        """ESG domain relevance (0-1) by keyword match. Default keywords for Chinese corpus."""
        if esg_keywords is None:
            esg_keywords = [
                "环境", "社会", "治理", "ESG", "CSR", "可持续发展",
                "碳排放", "环保", "社会责任", "公司治理", "员工", "培训",
                "供应链", "合规", "风险", "创新", "质量", "安全"
            ]
        text_lower = text.lower()
        matched_keywords = sum(1 for kw in esg_keywords if kw in text_lower)
        score = min(matched_keywords / len(esg_keywords) * 2, 1.0)
        return score
    def evaluate_single(
        self,
        reference: str,
        candidate: str,
        source_text: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate a single reference-candidate pair."""
        results = {}
        
        if "rouge-l" in self.metrics:
            results["rouge-l"] = self.compute_rouge_l(reference, candidate)
        
        if "bleu" in self.metrics:
            results["bleu"] = self.compute_bleu(reference, candidate)
        
        if "factcheck" in self.metrics and source_text:
            results["factcheck"] = self.compute_factcheck_score(
                reference, candidate, source_text
            )
        
        if "domain_relevance" in self.metrics:
            results["domain_relevance"] = self.compute_domain_relevance(candidate)
        return results
    def evaluate_batch(
        self,
        references: List[str],
        candidates: List[str],
        source_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Batch evaluate; returns mean metrics."""
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have same length")
        results = {}
        rouge_scores = []
        bleu_scores = []
        factcheck_scores = []
        domain_scores = []
        
        for i, (ref, cand) in enumerate(zip(references, candidates)):
            if "rouge-l" in self.metrics:
                rouge_scores.append(self.compute_rouge_l(ref, cand))
            
            if "bleu" in self.metrics:
                bleu_scores.append(self.compute_bleu(ref, cand))
            
            if "factcheck" in self.metrics and source_texts:
                factcheck_scores.append(
                    self.compute_factcheck_score(ref, cand, source_texts[i])
                )
            
            if "domain_relevance" in self.metrics:
                domain_scores.append(self.compute_domain_relevance(cand))
        
        if rouge_scores:
            results["rouge-l"] = np.mean(rouge_scores)
        
        if bleu_scores:
            results["bleu"] = np.mean(bleu_scores)
        
        if factcheck_scores:
            results["factcheck"] = np.mean(factcheck_scores)
        
        if domain_scores:
            results["domain_relevance"] = np.mean(domain_scores)
        
        if "bertscore" in self.metrics:
            bert_results = self.compute_bertscore(references, candidates)
            results.update({f"bertscore_{k}": v for k, v in bert_results.items()})
        
        if "bleurt" in self.metrics:
            results["bleurt"] = self.compute_bleurt(references, candidates)
        if "diversity" in self.metrics:
            results["diversity"] = self.compute_diversity(candidates)
        return results

