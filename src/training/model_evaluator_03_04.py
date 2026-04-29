"""
03-04 Model Evaluator: Post-training evaluation for fine-tuned ESG models.

Evaluates fine-tuned models on ESG-specific benchmarks using multiple metrics:
ROUGE-L, BLEU-4, BERTScore, domain relevance, and factual consistency.

Naming convention: filename_03_04.py (notebook 03, 4th module)
"""

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  ESG DOMAIN KEYWORDS (reused from evaluation pipeline)
# ═══════════════════════════════════════════════════════════════════

ESG_KEYWORDS_CORE = [
    "碳排放", "碳中和", "碳达峰", "温室气体", "气候变化", "碳足迹",
    "ESG", "可持续发展", "环境保护", "社会责任", "公司治理",
    "绿色金融", "绿色债券", "碳交易", "碳市场", "碳配额",
    "能源转型", "清洁能源", "可再生能源", "节能减排",
    "环境信息披露", "碳排放权", "低碳", "生态环境",
    "双碳", "碳减排", "碳排放量", "碳强度", "碳汇",
]


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single model."""
    model_short_name: str
    model_id: str
    param_count: str
    # Metrics
    rouge_l: float = 0.0
    bleu4: float = 0.0
    bertscore_f1: float = 0.0
    domain_relevance: float = 0.0
    factcheck: float = 0.0
    distinct_2: float = 0.0
    distinct_3: float = 0.0
    avg_response_len: float = 0.0
    # 95% Confidence intervals  {metric: (lower, upper)}
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Per-sample scores for statistical tests (not serialized to main table)
    per_sample_scores: Dict[str, List[float]] = field(default_factory=dict)
    # Meta
    num_samples: int = 0
    generation_time_seconds: float = 0.0
    is_finetuned: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("per_sample_scores", None)  # Exclude bulky per-sample data
        return d

    def print_summary(self) -> None:
        print(f"\n  {'─' * 55}")
        print(f"  Evaluation: {self.model_short_name} "
              f"({'fine-tuned' if self.is_finetuned else 'zero-shot'})")
        print(f"  {'─' * 55}")
        ci = self.confidence_intervals
        def _fmt(name, val):
            if name in ci:
                lo, hi = ci[name]
                return f"{val:.4f}  [{lo:.4f}, {hi:.4f}]"
            return f"{val:.4f}"
        print(f"  ROUGE-L:         {_fmt('rouge_l', self.rouge_l)}")
        print(f"  BLEU-4:          {_fmt('bleu4', self.bleu4)}")
        print(f"  BERTScore F1:    {_fmt('bertscore_f1', self.bertscore_f1)}")
        print(f"  Domain Rel.:     {_fmt('domain_relevance', self.domain_relevance)}")
        print(f"  FactCheck:       {_fmt('factcheck', self.factcheck)}")
        print(f"  Distinct-2:      {_fmt('distinct_2', self.distinct_2)}")
        print(f"  Distinct-3:      {_fmt('distinct_3', self.distinct_3)}")
        print(f"  Avg resp. len:   {self.avg_response_len:.0f} chars")
        print(f"  Samples:         {self.num_samples}")


# ═══════════════════════════════════════════════════════════════════
#  METRIC IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

def compute_rouge_l(predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
    """Compute ROUGE-L F1 score. Returns (mean, per_sample_scores)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip():
                scores.append(0.0)
                continue
            score = scorer.score(ref, pred)
            scores.append(score["rougeL"].fmeasure)
        mean = sum(scores) / len(scores) if scores else 0.0
        return mean, scores
    except ImportError:
        logger.warning("rouge_score not installed. Skipping ROUGE-L.")
        return 0.0, []


def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU-4 score (corpus-level)."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        # Tokenize by characters for Chinese
        refs = [[list(ref)] for ref in references]
        hyps = [list(pred) for pred in predictions]
        smooth = SmoothingFunction().method1
        return corpus_bleu(refs, hyps, smoothing_function=smooth)
    except ImportError:
        logger.warning("nltk not installed. Skipping BLEU-4.")
        return 0.0


def compute_bertscore(predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
    """Compute BERTScore F1. Returns (mean, per_sample_scores)."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            predictions, references,
            lang="zh",
            model_type="bert-base-chinese",
            verbose=False,
        )
        scores = F1.tolist()
        return F1.mean().item(), scores
    except ImportError:
        logger.warning("bert_score not installed. Skipping BERTScore.")
        return 0.0, []


def compute_domain_relevance(texts: List[str]) -> Tuple[float, List[float]]:
    """Compute ESG domain relevance. Returns (mean, per_sample_scores)."""
    if not texts:
        return 0.0, []
    scores = []
    for text in texts:
        if not text.strip():
            scores.append(0.0)
            continue
        hits = sum(1 for kw in ESG_KEYWORDS_CORE if kw in text)
        score = min(hits / 3.0, 1.0)
        scores.append(score)
    mean = sum(scores) / len(scores)
    return mean, scores


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """Compute Distinct-N diversity metric."""
    all_ngrams = []
    for text in texts:
        chars = list(text)
        ngrams = [tuple(chars[i:i + n]) for i in range(len(chars) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_factcheck_simple(
    predictions: List[str],
    sources: List[str],
) -> Tuple[float, List[float]]:
    """Simple factual consistency check. Returns (mean, per_sample_scores)."""
    if not sources:
        return 0.0, []
    scores = []
    for pred, src in zip(predictions, sources):
        pred_nums = set(re.findall(r'\d+[\.\d]*%?', pred))
        src_nums = set(re.findall(r'\d+[\.\d]*%?', src))
        if not pred_nums:
            scores.append(1.0)
        else:
            overlap = len(pred_nums & src_nums) / len(pred_nums)
            scores.append(overlap)
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, scores


# ═══════════════════════════════════════════════════════════════════
#  STATISTICAL TOOLS: Bootstrap CI & Paired Tests
# ═══════════════════════════════════════════════════════════════════

def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds.
    """
    import random
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return means[lo_idx], means[min(hi_idx, n_bootstrap - 1)]


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap significance test (two-sided).

    Tests H0: mean(A) == mean(B).
    Returns approximate p-value.
    """
    import random
    rng = random.Random(seed)
    n = len(scores_a)
    assert n == len(scores_b), "Score lists must be same length"
    if n == 0:
        return 1.0
    observed_diff = abs(sum(scores_a) / n - sum(scores_b) / n)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    count = 0
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_diff = abs(sum(sample) / n)
        if boot_diff >= observed_diff:
            count += 1
    return count / n_bootstrap


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> float:
    """Wilcoxon signed-rank test (two-sided). Returns p-value."""
    try:
        from scipy.stats import wilcoxon
        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        if all(d == 0 for d in diffs):
            return 1.0
        _, p = wilcoxon(scores_a, scores_b, alternative="two-sided")
        return p
    except ImportError:
        logger.warning("scipy not installed. Using bootstrap test instead.")
        return paired_bootstrap_test(scores_a, scores_b)


def compare_with_significance(
    result_a: "EvaluationResult",
    result_b: "EvaluationResult",
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare two models with statistical significance tests.

    Returns dict: {metric: {"diff": float, "p_value": float, "significant": bool}}
    """
    if metrics is None:
        metrics = ["rouge_l", "bertscore_f1", "domain_relevance", "factcheck"]

    results = {}
    for m in metrics:
        sa = result_a.per_sample_scores.get(m, [])
        sb = result_b.per_sample_scores.get(m, [])
        if not sa or not sb or len(sa) != len(sb):
            continue
        mean_a = sum(sa) / len(sa)
        mean_b = sum(sb) / len(sb)
        p = wilcoxon_test(sa, sb)
        results[m] = {
            "model_a": result_a.model_short_name,
            "model_b": result_b.model_short_name,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "diff": float(mean_a - mean_b),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }
    return results


# ═══════════════════════════════════════════════════════════════════
#  GENERATION & EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════

def generate_responses(
    model,
    tokenizer,
    instructions: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    batch_size: int = 4,
    template_name: str = "alpaca_en",
) -> List[str]:
    """Generate responses for a list of instructions.

    Args:
        model: HuggingFace model (with or without LoRA).
        tokenizer: Corresponding tokenizer.
        instructions: List of instruction strings.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (low=deterministic).
        batch_size: Batch size for generation.
        template_name: Prompt template for formatting.

    Returns:
        List of generated response strings.
    """
    import torch
    from ..training.data_loader_03_02 import format_alpaca_prompt

    model.eval()
    responses = []

    # Decoder-only models require left-padding during generation so that
    # all sequences end at the same position before the first new token.
    _orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Flags for models with known incompatibilities (persist across batches)
    _force_no_cache = False
    _force_bs1 = False

    for i in range(0, len(instructions), batch_size):
        effective_bs = 1 if _force_bs1 else batch_size
        batch_instructions = instructions[i:i + effective_bs]
        batch_prompts = []
        for inst in batch_instructions:
            prompt = format_alpaca_prompt(
                {"instruction": inst, "input": "", "output": ""},
                template_name=template_name,
                include_output=False,
            )
            batch_prompts.append(prompt)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            if _force_no_cache:
                gen_kwargs["use_cache"] = False
            try:
                outputs = model.generate(**gen_kwargs)
            except AttributeError as e:
                if "seen_tokens" in str(e) or "'NoneType'" in str(e):
                    if not _force_no_cache:
                        logger.warning("  Cache incompatibility detected; switching to use_cache=False for all batches")
                        _force_no_cache = True
                    gen_kwargs["use_cache"] = False
                    outputs = model.generate(**gen_kwargs)
                else:
                    raise
            except RuntimeError as e:
                err_str = str(e)
                if "size of tensor" in err_str:
                    if not _force_no_cache:
                        # Shape mismatch often caused by cache/position encoding incompatibility
                        logger.warning(f"  Shape mismatch; switching to use_cache=False")
                        _force_no_cache = True
                        gen_kwargs["use_cache"] = False
                        try:
                            outputs = model.generate(**gen_kwargs)
                        except RuntimeError:
                            # If still fails with batch>1, fall back to batch_size=1
                            if not _force_bs1:
                                logger.warning(f"  Still failing; switching to batch_size=1")
                                _force_bs1 = True
                            raise
                    elif not _force_bs1:
                        logger.warning(f"  Shape mismatch in batch; switching to batch_size=1")
                        _force_bs1 = True
                        # Redo this batch one-by-one
                        for inst in batch_instructions:
                            prompt = format_alpaca_prompt(
                                {"instruction": inst, "input": "", "output": ""},
                                template_name=template_name,
                                include_output=False,
                            )
                            single_input = tokenizer(
                                [prompt], return_tensors="pt", truncation=True, max_length=1536,
                            ).to(model.device)
                            single_kwargs = dict(
                                **single_input,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                do_sample=temperature > 0,
                                top_p=0.9,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=False,
                            )
                            single_out = model.generate(**single_kwargs)
                            gen_text = tokenizer.decode(
                                single_out[0][single_input["input_ids"].shape[1]:],
                                skip_special_tokens=True,
                            )
                            responses.append(gen_text.strip())
                        continue  # Skip normal post-processing for this batch
                    else:
                        raise
                else:
                    raise

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            )
            responses.append(generated.strip())

    tokenizer.padding_side = _orig_padding_side
    return responses


def evaluate_model(
    model_short_name: str,
    model_id: str,
    param_count: str,
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    is_finetuned: bool = True,
) -> EvaluationResult:
    """Run full evaluation suite on generated predictions.

    Args:
        model_short_name: Short identifier.
        model_id: Full model ID.
        param_count: Parameter count string.
        predictions: Generated responses.
        references: Ground truth responses.
        sources: Source texts (for factcheck).
        is_finetuned: Whether this model was fine-tuned.

    Returns:
        EvaluationResult object.
    """
    result = EvaluationResult(
        model_short_name=model_short_name,
        model_id=model_id,
        param_count=param_count,
        num_samples=len(predictions),
        is_finetuned=is_finetuned,
    )

    logger.info(f"Evaluating {model_short_name} on {len(predictions)} samples...")

    # ROUGE-L
    result.rouge_l, rouge_scores = compute_rouge_l(predictions, references)
    result.per_sample_scores["rouge_l"] = rouge_scores
    logger.info(f"  ROUGE-L: {result.rouge_l:.4f}")

    # BLEU-4 (corpus-level, no per-sample)
    result.bleu4 = compute_bleu4(predictions, references)
    logger.info(f"  BLEU-4: {result.bleu4:.4f}")

    # BERTScore
    result.bertscore_f1, bert_scores = compute_bertscore(predictions, references)
    result.per_sample_scores["bertscore_f1"] = bert_scores
    logger.info(f"  BERTScore: {result.bertscore_f1:.4f}")

    # Domain relevance
    result.domain_relevance, dr_scores = compute_domain_relevance(predictions)
    result.per_sample_scores["domain_relevance"] = dr_scores
    logger.info(f"  Domain Rel.: {result.domain_relevance:.4f}")

    # Factcheck
    if sources:
        result.factcheck, fc_scores = compute_factcheck_simple(predictions, sources)
        result.per_sample_scores["factcheck"] = fc_scores
    logger.info(f"  FactCheck: {result.factcheck:.4f}")

    # Diversity
    result.distinct_2 = compute_distinct_n(predictions, 2)
    result.distinct_3 = compute_distinct_n(predictions, 3)
    logger.info(f"  Distinct-2: {result.distinct_2:.4f}")
    logger.info(f"  Distinct-3: {result.distinct_3:.4f}")

    # Avg response length
    result.avg_response_len = (
        sum(len(p) for p in predictions) / len(predictions) if predictions else 0
    )

    # Bootstrap 95% confidence intervals
    for metric_name, scores in result.per_sample_scores.items():
        if scores:
            lo, hi = bootstrap_ci(scores)
            result.confidence_intervals[metric_name] = (lo, hi)
            logger.info(f"  {metric_name} 95% CI: [{lo:.4f}, {hi:.4f}]")

    return result


# ═══════════════════════════════════════════════════════════════════
#  RESULTS COMPARISON & LATEX EXPORT
# ═══════════════════════════════════════════════════════════════════

def compare_results(results: List[EvaluationResult]) -> str:
    """Generate a formatted comparison table.

    Returns:
        Formatted string table.
    """
    metrics = ["rouge_l", "bleu4", "bertscore_f1", "domain_relevance",
               "factcheck", "distinct_2", "distinct_3"]
    metric_names = {
        "rouge_l": "ROUGE-L",
        "bleu4": "BLEU-4",
        "bertscore_f1": "BERTScore",
        "domain_relevance": "Domain Rel.",
        "factcheck": "FactCheck",
        "distinct_2": "Distinct-2",
        "distinct_3": "Distinct-3",
    }

    # Find best per metric
    best = {}
    for m in metrics:
        best[m] = max(results, key=lambda r: getattr(r, m)).model_short_name

    header = (f"{'Model':<25} {'Params':>8} {'Type':>10} " +
              " ".join(f"{metric_names[m]:>12}" for m in metrics))
    lines = [header, "─" * len(header)]

    for r in sorted(results, key=lambda x: x.param_count):
        ft_label = "fine-tuned" if r.is_finetuned else "zero-shot"
        values = []
        for m in metrics:
            val = getattr(r, m)
            s = f"{val:.4f}"
            if r.model_short_name == best[m]:
                s = f"*{s}"  # Mark best
            values.append(f"{s:>12}")
        line = f"  {r.model_short_name:<23} {r.param_count:>8} {ft_label:>10} " + " ".join(values)
        lines.append(line)

    return "\n".join(lines)


def export_latex_table(
    results: List[EvaluationResult],
    output_path: str,
    caption: str = "Model scale ablation results.",
    label: str = "tab:model_scale",
) -> str:
    """Export results as a LaTeX table.

    Args:
        results: List of EvaluationResult objects.
        output_path: File path for the .tex output.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX string.
    """
    metrics = ["rouge_l", "bleu4", "domain_relevance", "factcheck", "distinct_2"]
    metric_names = {
        "rouge_l": "ROUGE-L",
        "bleu4": "BLEU-4",
        "domain_relevance": "Domain Rel.",
        "factcheck": "FactCheck",
        "distinct_2": "Distinct-2",
    }

    # Find best per metric
    best = {}
    for m in metrics:
        best_val = max(getattr(r, m) for r in results)
        best[m] = [r.model_short_name for r in results if getattr(r, m) == best_val]

    col_spec = "lcc" + "c" * len(metrics)
    header = " & ".join(
        ["Model", "Params", "Type"] + [metric_names[m] for m in metrics]
    )

    rows = []
    for r in results:
        ft_label = "FT" if r.is_finetuned else "ZS"
        cells = [r.display_name if hasattr(r, 'display_name') else r.model_short_name,
                 r.param_count, ft_label]
        for m in metrics:
            val = getattr(r, m)
            cell = f"{val:.4f}"
            if r.model_short_name in best[m]:
                cell = r"\\textbf{" + cell + "}"
            cells.append(cell)
        rows.append(" & ".join(cells) + r" \\\\")

    latex_lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        f"\\\\caption{{{caption}}}",
        f"\\\\label{{{label}}}",
        r"\\resizebox{\\columnwidth}{!}{%",
        f"\\\\begin{{tabular}}{{{col_spec}}}",
        r"\\toprule",
        header + r" \\\\",
        r"\\midrule",
    ] + rows + [
        r"\\bottomrule",
        r"\\end{tabular}}",
        r"\\end{table}",
    ]

    latex = "\n".join(latex_lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)

    logger.info(f"LaTeX table saved to: {output_path}")
    return latex
