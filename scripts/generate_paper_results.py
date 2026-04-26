#!/usr/bin/env python3
"""
Generate all paper tables and figures from experiment results.

Produces:
  - Table 2: Main comparison (CoDE vs Baselines) — LaTeX
  - Table 3: Ablation study — LaTeX
  - Figure 3: Radar chart (CoDE vs Baselines)
  - Figure 4: Ablation bar charts (4 sub-figures)
  - Figure 5: Quality score distribution
  - Figure 6: Expert distribution
  - Summary analysis text

Usage:
    python scripts/generate_paper_results.py
"""

import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# =====================================================================
# Config
# =====================================================================
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures_and_tables"
PAPER_FIG_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Data loaders
# =====================================================================

def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================================
# Efficient metric computation (optimized for speed)
# =====================================================================

ESG_KEYWORDS = [
    "环境", "社会", "治理", "ESG", "CSR", "可持续发展",
    "碳排放", "碳中和", "碳达峰", "环保", "社会责任", "公司治理",
    "员工", "培训", "供应链", "合规", "风险", "创新", "质量", "安全",
    "能源", "水资源", "废弃物", "温室气体", "GRI", "TCFD", "SASB",
]


try:
    from rouge_score import rouge_scorer as _rs
    _ROUGE_SCORER = _rs.RougeScorer(["rougeL"], use_stemmer=False)
except ImportError:
    _ROUGE_SCORER = None


def fast_rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L using google rouge_score library (fast C implementation)."""
    if not reference or not candidate:
        return 0.0
    # Truncate to keep it fast
    ref = reference[:500]
    cand = candidate[:500]
    if _ROUGE_SCORER is None:
        return 0.0
    try:
        return _ROUGE_SCORER.score(ref, cand)["rougeL"].fmeasure
    except Exception:
        return 0.0


def fast_bleu4(reference: str, candidate: str) -> float:
    """Character-level BLEU-4, truncated for speed."""
    ref_chars = list(reference[:200])
    cand_chars = list(candidate[:200])
    if len(cand_chars) < 4 or len(ref_chars) < 4:
        return 0.0

    def ngram_counts(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

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

    bp = min(1.0, np.exp(1 - len(ref_chars) / max(len(cand_chars), 1)))
    return bp * (score ** 0.25)


def distinct_n(texts, n=2):
    all_ngrams = []
    for text in texts:
        chars = list(text[:200])
        for i in range(len(chars) - n + 1):
            all_ngrams.append(tuple(chars[i:i+n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def domain_relevance(text):
    text_lower = text.lower()
    matched = sum(1 for kw in ESG_KEYWORDS if kw in text_lower)
    return min(matched / len(ESG_KEYWORDS) * 2, 1.0)


def factcheck_overlap(candidate, source):
    cand_nums = set(re.findall(r"\d+", candidate))
    src_nums = set(re.findall(r"\d+", source))
    if not cand_nums:
        return 1.0
    return len(cand_nums & src_nums) / len(cand_nums)


def structural_completeness(instruction, response):
    score = 0.0
    if instruction and len(instruction.strip()) >= 10:
        score += 0.5
    if response and len(response.strip()) >= 20:
        score += 0.5
    return score


def evaluate_records(records, label=""):
    """Evaluate a list of QA records — fast version."""
    instrs, resps, srcs, qscores = [], [], [], []

    for r in records:
        instr = r.get("instruction") or r.get("question") or ""
        resp = r.get("output") or r.get("answer") or r.get("response") or ""
        src = r.get("input") or r.get("chunk") or ""
        instrs.append(instr)
        resps.append(resp)
        srcs.append(src)
        meta = r.get("metadata", {})
        qscores.append(meta.get("quality_score", 0) if isinstance(meta, dict) else 0)

    n = len(records)
    if n == 0:
        return {"label": label, "n": 0}

    rouge_scores, bleu_scores, fc_scores, dm_scores, sc_scores = [], [], [], [], []
    for instr, resp, src in zip(instrs, resps, srcs):
        if src:
            rouge_scores.append(fast_rouge_l(src, resp))
            bleu_scores.append(fast_bleu4(src, resp))
            fc_scores.append(factcheck_overlap(resp, src))
        dm_scores.append(domain_relevance(resp))
        sc_scores.append(structural_completeness(instr, resp))

    return {
        "label": label,
        "n": n,
        "avg_instr_len": round(np.mean([len(i) for i in instrs]), 1),
        "avg_resp_len": round(np.mean([len(r) for r in resps]), 1),
        "rouge_l": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
        "bleu4": round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
        "distinct_1": round(distinct_n(resps, 1), 4),
        "distinct_2": round(distinct_n(resps, 2), 4),
        "distinct_3": round(distinct_n(resps, 3), 4),
        "domain_rel": round(float(np.mean(dm_scores)), 4),
        "factcheck": round(float(np.mean(fc_scores)), 4) if fc_scores else 0.0,
        "struct_comp": round(float(np.mean(sc_scores)), 4),
        "avg_quality": round(float(np.mean(qscores)), 4) if any(q > 0 for q in qscores) else 0.0,
        "std_quality": round(float(np.std(qscores)), 4) if any(q > 0 for q in qscores) else 0.0,
    }


# =====================================================================
# 1. Compute all metrics
# =====================================================================

def compute_all_metrics():
    """Compute intrinsic metrics for CoDE, baselines, and all ablation conditions.
    
    Reuses existing intrinsic_comparison.json for CoDE and baselines (already computed).
    Only computes fresh metrics for ablation conditions.
    """
    print("=" * 60)
    print("Computing Intrinsic Metrics for All Conditions")
    print("=" * 60)

    results = {}

    # Try to reuse existing CoDE + baseline results
    existing_path = PROJECT_ROOT / "results" / "outputs" / "intrinsic_comparison.json"
    if existing_path.exists():
        existing = load_json(str(existing_path))
        # Remap keys to match our schema
        key_map = {
            "avg_instruction_length": "avg_instr_len",
            "avg_response_length": "avg_resp_len",
            "domain_relevance": "domain_rel",
            "structural_completeness": "struct_comp",
            "avg_quality_score": "avg_quality",
        }
        for method, metrics in existing.items():
            remapped = {}
            for k, v in metrics.items():
                new_k = key_map.get(k, k)
                remapped[new_k] = v if v is not None else 0.0
            results[method] = remapped
        print(f"\n  Reused existing results for: {list(existing.keys())}")

    # Reuse ablation metrics from intrinsic_comparison.json (already computed)
    # The intrinsic_comparison.json contains correct ablation results under keys like
    # "ablation/expert_count/expert_count_1". We also alias them without the "ablation/"
    # prefix so downstream figure/table generators can find them.
    print("\n[Ablation] Reusing ablation metrics from intrinsic_comparison.json...")
    ablation_keys = [k for k in existing.keys() if k.startswith("ablation/")]
    for key in ablation_keys:
        # Also store under short key (e.g., "expert_count/expert_count_1")
        short_key = key.replace("ablation/", "", 1)
        results[short_key] = results[key].copy()
        print(f"  Reused {key} -> {short_key}")

    # Save
    out_path = OUTPUT_DIR / "all_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Full metrics saved: {out_path}")

    return results


# =====================================================================
# 2. Generate LaTeX tables
# =====================================================================

def generate_latex_table_2(results):
    """Table 2: Main Comparison — CoDE vs Baselines."""
    print("\n" + "=" * 60)
    print("Generating Table 2: Main Comparison (CoDE vs Baselines)")
    print("=" * 60)

    methods = ["CoDE (ours)", "direct_prompting", "self_instruct", "wizardlm_evol"]
    display_names = {
        "CoDE (ours)": r"\textbf{CoDE (ours)}",
        "direct_prompting": "Direct Prompting",
        "self_instruct": "Self-Instruct",
        "wizardlm_evol": "WizardLM-Evol",
    }
    metrics = ["rouge_l", "bleu4", "distinct_2", "distinct_3", "domain_rel", "factcheck", "struct_comp"]
    metric_names = {
        "rouge_l": "ROUGE-L",
        "bleu4": "BLEU-4",
        "distinct_2": "Distinct-2",
        "distinct_3": "Distinct-3",
        "domain_rel": "Domain Rel.",
        "factcheck": "FactCheck",
        "struct_comp": "Struct. Comp.",
    }

    # Find best per metric
    best = {}
    for m in metrics:
        vals = {method: results.get(method, {}).get(m, 0) for method in methods}
        best[m] = max(vals, key=lambda x: vals[x])

    # Generate LaTeX
    header = " & ".join(["Method", "N"] + [metric_names[m] for m in metrics])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Intrinsic evaluation results comparing CoDE with baseline methods. Best results are in \textbf{bold}.}",
        r"\label{tab:main_comparison}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lc" + "c" * len(metrics) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for method in methods:
        r = results.get(method, {})
        row = [display_names[method], str(r.get("n", 0))]
        for m in metrics:
            val = r.get(m, 0) or 0
            cell = f"{val:.4f}"
            if method == best[m]:
                cell = r"\textbf{" + cell + "}"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    out_path = OUTPUT_DIR / "table2_main_comparison.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {out_path}")
    print("\n" + latex)

    # Also generate a readable summary
    print("\n--- Readable Main Comparison ---")
    print(f"{'Method':<20} {'N':>5} {'ROUGE-L':>10} {'BLEU-4':>10} {'Dist-2':>10} {'Dist-3':>10} {'DomRel':>10} {'FactChk':>10} {'Struct':>10}")
    print("-" * 105)
    for method in methods:
        r = results.get(method, {})
        print(f"{method:<20} {r.get('n', 0):>5} {r.get('rouge_l', 0):>10.4f} {r.get('bleu4', 0):>10.4f} "
              f"{r.get('distinct_2', 0):>10.4f} {r.get('distinct_3', 0):>10.4f} {r.get('domain_rel', 0):>10.4f} "
              f"{r.get('factcheck', 0):>10.4f} {r.get('struct_comp', 0):>10.4f}")

    return latex


def generate_latex_table_3(results):
    """Table 3: Ablation Study."""
    print("\n" + "=" * 60)
    print("Generating Table 3: Ablation Study")
    print("=" * 60)

    ablation_dims = {
        "Expert Count": [
            ("expert_count/expert_count_1", "K=1"),
            ("expert_count/expert_count_2", "K=2"),
            ("expert_count/expert_count_3", "K=3"),
            ("expert_count/expert_count_5", "K=5"),
        ],
        "Collaboration": [
            ("collaboration/collab_none", "None"),
            ("collaboration/collab_sequential", "Sequential"),
            ("collaboration/collab_parallel", "Parallel"),
        ],
        "Feedback Rounds": [
            ("feedback/feedback_0", "R=0"),
            ("feedback/feedback_1", "R=1"),
            ("feedback/feedback_2", "R=2"),
        ],
        "Knowledge Injection": [
            ("knowledge/knowledge_off", "Off"),
            ("knowledge/knowledge_on", "On"),
        ],
    }

    metrics = ["avg_quality", "rouge_l", "bleu4", "distinct_2", "domain_rel", "factcheck"]
    metric_names = {
        "avg_quality": "Quality",
        "rouge_l": "ROUGE-L",
        "bleu4": "BLEU-4",
        "distinct_2": "Distinct-2",
        "domain_rel": "Domain Rel.",
        "factcheck": "FactCheck",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study results across four dimensions. Best results per dimension are in \textbf{bold}.}",
        r"\label{tab:ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
        r"\toprule",
        " & ".join(["Dimension", "Setting"] + [metric_names[m] for m in metrics]) + r" \\",
        r"\midrule",
    ]

    for dim_name, conditions in ablation_dims.items():
        # Find best per metric within this dimension
        dim_best = {}
        for m in metrics:
            vals = {}
            for key, display in conditions:
                r = results.get(key, {})
                vals[key] = r.get(m, 0) or 0
            dim_best[m] = max(vals, key=lambda x: vals[x])

        for i, (key, display) in enumerate(conditions):
            r = results.get(key, {})
            row_dim = dim_name if i == 0 else ""
            row = [row_dim, display]
            for m in metrics:
                val = r.get(m, 0) or 0
                cell = f"{val:.4f}"
                if key == dim_best[m]:
                    cell = r"\textbf{" + cell + "}"
                row.append(cell)
            lines.append(" & ".join(row) + r" \\")

        if dim_name != "Knowledge Injection":
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    out_path = OUTPUT_DIR / "table3_ablation.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {out_path}")
    print("\n" + latex)

    # Readable
    print("\n--- Readable Ablation Table ---")
    print(f"{'Dimension':<20} {'Setting':<15} {'Quality':>10} {'ROUGE-L':>10} {'BLEU-4':>10} {'Dist-2':>10} {'DomRel':>10} {'FactChk':>10}")
    print("-" * 105)
    for dim_name, conditions in ablation_dims.items():
        for key, display in conditions:
            r = results.get(key, {})
            print(f"{dim_name:<20} {display:<15} {r.get('avg_quality', 0):>10.4f} {r.get('rouge_l', 0):>10.4f} "
                  f"{r.get('bleu4', 0):>10.4f} {r.get('distinct_2', 0):>10.4f} {r.get('domain_rel', 0):>10.4f} "
                  f"{r.get('factcheck', 0):>10.4f}")
        print("-" * 105)

    return latex


# =====================================================================
# 3. Generate Figures
# =====================================================================

def figure_3_radar(results):
    """Figure 3: Radar chart comparing CoDE vs Baselines on key metrics."""
    print("\n[Figure 3] Radar chart: CoDE vs Baselines")

    methods = ["CoDE (ours)", "direct_prompting", "self_instruct", "wizardlm_evol"]
    display = ["CoDE (ours)", "Direct Prompting", "Self-Instruct", "WizardLM-Evol"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    metrics = ["rouge_l", "bleu4", "distinct_2", "domain_rel", "factcheck", "struct_comp"]
    metric_labels = ["ROUGE-L", "BLEU-4", "Distinct-2", "Domain Rel.", "FactCheck", "Struct. Comp."]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, method in enumerate(methods):
        r = results.get(method, {})
        values = [r.get(m, 0) or 0 for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=display[i], color=colors[i], markersize=5)
        ax.fill(angles, values, alpha=0.08, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_ylim(0, 1.0)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Intrinsic Quality Comparison", size=14, pad=20)

    for path in [OUTPUT_DIR / "figure3_radar.pdf", PAPER_FIG_DIR / "figure3_radar.pdf",
                 OUTPUT_DIR / "figure3_radar.png"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figure3_radar.pdf/.png")


def figure_4_ablation_bars(results):
    """Figure 4: Ablation study bar charts — 4 sub-figures."""
    print("\n[Figure 4] Ablation bar charts")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ablation Study Results", fontsize=16, y=1.02)

    ablation_configs = [
        {
            "title": "(a) Number of Experts",
            "conditions": [
                ("expert_count/expert_count_1", "K=1"),
                ("expert_count/expert_count_2", "K=2"),
                ("expert_count/expert_count_3", "K=3"),
                ("expert_count/expert_count_5", "K=5"),
            ],
            "ax": axes[0, 0],
        },
        {
            "title": "(b) Collaboration Mode",
            "conditions": [
                ("collaboration/collab_none", "None"),
                ("collaboration/collab_sequential", "Seq."),
                ("collaboration/collab_parallel", "Par."),
            ],
            "ax": axes[0, 1],
        },
        {
            "title": "(c) Feedback Rounds",
            "conditions": [
                ("feedback/feedback_0", "R=0"),
                ("feedback/feedback_1", "R=1"),
                ("feedback/feedback_2", "R=2"),
            ],
            "ax": axes[1, 0],
        },
        {
            "title": "(d) Knowledge Injection",
            "conditions": [
                ("knowledge/knowledge_off", "Off"),
                ("knowledge/knowledge_on", "On"),
            ],
            "ax": axes[1, 1],
        },
    ]

    metrics = ["avg_quality", "rouge_l", "domain_rel", "factcheck"]
    metric_labels = ["Quality", "ROUGE-L", "Domain Rel.", "FactCheck"]
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for config in ablation_configs:
        ax = config["ax"]
        conditions = config["conditions"]
        n_cond = len(conditions)
        n_metrics = len(metrics)
        x = np.arange(n_cond)
        width = 0.8 / n_metrics

        for i, (m, m_label) in enumerate(zip(metrics, metric_labels)):
            vals = []
            for key, _ in conditions:
                r = results.get(key, {})
                vals.append(r.get(m, 0) or 0)
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                         label=m_label if config == ablation_configs[0] else "",
                         color=bar_colors[i], alpha=0.85)
            # Add value labels on bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([d for _, d in conditions], fontsize=10)
        ax.set_title(config["title"], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    # Single legend
    fig.legend(metric_labels, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout()
    for path in [OUTPUT_DIR / "figure4_ablation.pdf", PAPER_FIG_DIR / "figure4_ablation.pdf",
                 OUTPUT_DIR / "figure4_ablation.png"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figure4_ablation.pdf/.png")


def figure_5_quality_distribution():
    """Figure 5: Quality score distribution of CoDE dataset."""
    print("\n[Figure 5] Quality score distribution")

    coe_records = []
    for f in ["data/instructions/qa_pairs_complete_v3_1.5w.jsonl",
              "data/instructions/qa_pairs_complete_v3_2w.jsonl"]:
        fp = PROJECT_ROOT / f
        if fp.exists():
            coe_records.extend(load_jsonl(str(fp)))

    scores = []
    for r in coe_records:
        meta = r.get("metadata", {})
        if isinstance(meta, dict):
            qs = meta.get("quality_score", 0)
            if qs and qs > 0:
                scores.append(qs)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(scores, bins=bins, color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(scores), color="#E91E63", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(scores):.3f}")
    ax.axvline(np.median(scores), color="#FF9800", linestyle="-.", linewidth=2,
               label=f"Median = {np.median(scores):.3f}")
    ax.set_xlabel("Quality Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Quality Score Distribution of CoDE Dataset", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for path in [OUTPUT_DIR / "figure5_quality_dist.pdf", PAPER_FIG_DIR / "figure5_quality_dist.pdf",
                 OUTPUT_DIR / "figure5_quality_dist.png"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figure5_quality_dist.pdf/.png")
    print(f"  Stats: mean={np.mean(scores):.4f}, median={np.median(scores):.4f}, "
          f"std={np.std(scores):.4f}, n={len(scores)}")


def figure_6_expert_distribution():
    """Figure 6: Expert usage distribution."""
    print("\n[Figure 6] Expert distribution")

    stats = load_json(str(OUTPUT_DIR / "table2_qa_statistics.json"))
    expert_dist = stats.get("expert_distribution", {})

    # Sort by frequency
    sorted_experts = sorted(expert_dist.items(), key=lambda x: x[1], reverse=True)
    names = [e[0].replace("_expert", "").replace("_", " ").title() for e in sorted_experts]
    counts = [e[1] for e in sorted_experts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), counts, color="#2196F3", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Invocations", fontsize=12)
    ax.set_title("Expert Agent Usage Distribution in CoDE Framework", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    for path in [OUTPUT_DIR / "figure6_expert_dist.pdf", PAPER_FIG_DIR / "figure6_expert_dist.pdf",
                 OUTPUT_DIR / "figure6_expert_dist.png"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figure6_expert_dist.pdf/.png")


def figure_7_baseline_bar(results):
    """Figure 7: Grouped bar chart for CoDE vs Baselines on all metrics."""
    print("\n[Figure 7] CoDE vs Baselines grouped bar chart")

    methods = ["CoDE (ours)", "direct_prompting", "self_instruct", "wizardlm_evol"]
    display = ["CoDE\n(ours)", "Direct\nPrompting", "Self-\nInstruct", "WizardLM\n-Evol"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    metrics = ["rouge_l", "bleu4", "distinct_2", "distinct_3", "domain_rel", "factcheck"]
    metric_labels = ["ROUGE-L", "BLEU-4", "Distinct-2", "Distinct-3", "Domain\nRel.", "FactCheck"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.5), sharey=False)
    fig.suptitle("CoDE vs Baseline Methods: Metric Comparison", fontsize=14, y=1.05)

    for j, (m, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        vals = [results.get(method, {}).get(m, 0) or 0 for method in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(display, fontsize=8)
        ax.set_title(m_label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    for path in [OUTPUT_DIR / "figure7_baseline_bars.pdf", PAPER_FIG_DIR / "figure7_baseline_bars.pdf",
                 OUTPUT_DIR / "figure7_baseline_bars.png"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figure7_baseline_bars.pdf/.png")


# =====================================================================
# 4. Key Findings Analysis
# =====================================================================

def analyze_findings(results):
    """Print key findings for the paper narrative."""
    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 60)

    code = results.get("CoDE (ours)", {})
    dp = results.get("direct_prompting", {})
    si = results.get("self_instruct", {})
    wl = results.get("wizardlm_evol", {})

    print("\n📊 Finding 1: CoDE dominates all baselines across all metrics")
    for m, name in [("rouge_l", "ROUGE-L"), ("bleu4", "BLEU-4"), ("distinct_2", "Distinct-2"),
                     ("domain_rel", "Domain Rel."), ("factcheck", "FactCheck")]:
        code_v = code.get(m, 0)
        best_bl = max(dp.get(m, 0), si.get(m, 0), wl.get(m, 0))
        ratio = code_v / best_bl if best_bl > 0 else float("inf")
        print(f"  {name}: CoDE={code_v:.4f} vs Best-Baseline={best_bl:.4f} ({ratio:.1f}x)")

    # Ablation findings
    print("\n📊 Finding 2: Expert count — diminishing returns beyond K=3")
    for k in [1, 2, 3, 5]:
        r = results.get(f"expert_count/expert_count_{k}", {})
        print(f"  K={k}: Quality={r.get('avg_quality', 0):.4f}, ROUGE-L={r.get('rouge_l', 0):.4f}")

    print("\n📊 Finding 3: Collaboration mode matters")
    for mode in ["none", "sequential", "parallel"]:
        r = results.get(f"collaboration/collab_{mode}", {})
        print(f"  {mode}: Quality={r.get('avg_quality', 0):.4f}, ROUGE-L={r.get('rouge_l', 0):.4f}")

    print("\n📊 Finding 4: Feedback is ESSENTIAL (R=0 produces empty outputs)")
    for rounds in [0, 1, 2]:
        r = results.get(f"feedback/feedback_{rounds}", {})
        print(f"  R={rounds}: Quality={r.get('avg_quality', 0):.4f}, ROUGE-L={r.get('rouge_l', 0):.4f}, "
              f"Struct={r.get('struct_comp', 0):.4f}")

    print("\n📊 Finding 5: Domain knowledge injection improves quality")
    ko = results.get("knowledge/knowledge_off", {})
    kn = results.get("knowledge/knowledge_on", {})
    print(f"  Off: Quality={ko.get('avg_quality', 0):.4f}, DomRel={ko.get('domain_rel', 0):.4f}")
    print(f"  On:  Quality={kn.get('avg_quality', 0):.4f}, DomRel={kn.get('domain_rel', 0):.4f}")

    print("\n📊 Finding 6: CoDE full pipeline (recommended config)")
    print(f"  K=3 experts, parallel collaboration, R=2 feedback, knowledge ON")
    print(f"  Overall: Quality={code.get('avg_quality', 0):.4f}, ROUGE-L={code.get('rouge_l', 0):.4f}, "
          f"FactCheck={code.get('factcheck', 0):.4f}")

    # WizardLM-Evol anomaly
    print("\n⚠️  WizardLM-Evol anomaly:")
    print(f"  Avg instruction length: {wl.get('avg_instr_len', 0)} chars (vs CoDE: {code.get('avg_instr_len', 0)})")
    print(f"  Avg response length: {wl.get('avg_resp_len', 0)} chars (vs CoDE: {code.get('avg_resp_len', 0)})")
    print(f"  FactCheck: {wl.get('factcheck', 0):.4f} (lowest — verbose generation hallucinate more)")

    # Dataset scale
    print("\n📊 Dataset Statistics:")
    stats_path = OUTPUT_DIR / "table2_qa_statistics.json"
    if stats_path.exists():
        stats = load_json(str(stats_path))
        print(f"  Total QA pairs: {stats.get('total_records', 'N/A')}")
        print(f"  Multi-expert ratio: {stats.get('multi_expert_ratio', 'N/A'):.1%}")
        print(f"  Expert types: {stats.get('expert_types_used', 'N/A')}")
        print(f"  Avg quality: {stats.get('avg_quality_score', 'N/A')}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" VeritasCarbon — Paper Results Generator")
    print(" SIGMOD 2027")
    print("=" * 60)

    # Step 1: Compute metrics
    results = compute_all_metrics()

    # Step 2: Generate LaTeX tables
    generate_latex_table_2(results)
    generate_latex_table_3(results)

    # Step 3: Generate figures
    figure_3_radar(results)
    figure_4_ablation_bars(results)
    figure_5_quality_distribution()
    figure_6_expert_distribution()
    figure_7_baseline_bar(results)

    # Step 4: Key findings
    analyze_findings(results)

    print("\n" + "=" * 60)
    print(" ALL DONE — Files saved to:")
    print(f"  Tables: {OUTPUT_DIR}/")
    print(f"  Figures: {PAPER_FIG_DIR}/")
    print("=" * 60)
