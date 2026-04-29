"""
03-04 Dataset statistics and figure generation for VLDB 2027 paper.

Produces:
  Table 1: Corpus statistics (per-layer doc count, chunk count, dedup rate)
  Table 2: QA dataset overview (expert distribution, quality, topic coverage)
  Figure 1: Expert usage frequency bar chart
  Figure 2: Quality score distribution histogram
  Figure 3: Topic coverage sunburst / bar
  Figure 4: Layer x Year heatmap (chunk distribution)

All figures are saved to results/figures_and_tables/.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try import matplotlib; graceful fallback
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed; figures will be skipped")

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False
    logger.warning("pandas not installed; CSV tables will be skipped")


# =========================================================================
# Data loaders
# =========================================================================

def load_jsonl(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def count_files_in_dir(directory: Path, extensions=None) -> int:
    if extensions is None:
        extensions = [".pdf", ".docx", ".txt"]
    total = 0
    for ext in extensions:
        total += len(list(directory.rglob(f"*{ext}")))
    return total


# =========================================================================
# Table 1: Corpus Statistics
# =========================================================================

def compute_corpus_statistics(
    raw_corpus_dir: str = "data/raw_corpus",
    chunks_file: str = "data/processed_corpus/chunks_clean_fixed.jsonl",
    output_dir: str = "results/figures_and_tables",
) -> Dict:
    """Compute per-layer statistics for the corpus."""
    raw_path = Path(raw_corpus_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Count raw documents per layer
    layer_doc_counts = {}
    for layer_dir in sorted(raw_path.iterdir()):
        if layer_dir.is_dir() and layer_dir.name.startswith("Layer"):
            count = count_files_in_dir(layer_dir)
            layer_doc_counts[layer_dir.name] = count

    # Count chunks per layer & extract year info
    layer_chunk_counts = Counter()
    layer_year_counts = defaultdict(Counter)  # layer -> {year: count}
    total_chunks = 0

    if Path(chunks_file).exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                total_chunks += 1
                chunk_id = chunk.get("chunk_id", "") or chunk.get("doc_id", "")

                # Detect layer
                layer = "Unknown"
                for lname in ["Layer1", "Layer2", "Layer3", "Layer4",
                              "第一层", "第二层", "第三层", "第四层"]:
                    if lname in chunk_id:
                        layer = lname.replace("第一层", "Layer1").replace(
                            "第二层", "Layer2").replace(
                            "第三层", "Layer3").replace(
                            "第四层", "Layer4")
                        break
                layer_chunk_counts[layer] += 1

                # Year detection (e.g. 2021, 2022, ...)
                years = re.findall(r"20[12]\d", chunk_id)
                for y in set(years):
                    yr = int(y)
                    if 2015 <= yr <= 2026:
                        layer_year_counts[layer][yr] += 1

    stats = {
        "per_layer": {},
        "total_documents": sum(layer_doc_counts.values()),
        "total_chunks": total_chunks,
    }
    for layer in sorted(set(list(layer_doc_counts.keys()) + list(layer_chunk_counts.keys()))):
        stats["per_layer"][layer] = {
            "documents": layer_doc_counts.get(layer, 0),
            "chunks": layer_chunk_counts.get(layer, 0),
            "year_distribution": dict(sorted(layer_year_counts.get(layer, {}).items())),
        }

    # Save JSON
    with open(out_path / "table1_corpus_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Save CSV
    if HAS_PD:
        rows = []
        for layer, info in stats["per_layer"].items():
            rows.append({
                "Layer": layer,
                "Documents": info["documents"],
                "Chunks": info["chunks"],
                "Years": ", ".join(str(y) for y in sorted(info["year_distribution"].keys())),
            })
        rows.append({
            "Layer": "Total",
            "Documents": stats["total_documents"],
            "Chunks": stats["total_chunks"],
            "Years": "",
        })
        df = pd.DataFrame(rows)
        df.to_csv(out_path / "table1_corpus_statistics.csv", index=False)

    logger.info(f"Table 1 saved: {out_path / 'table1_corpus_statistics.json'}")
    return stats


# =========================================================================
# Table 2 + Figures: QA Dataset Overview
# =========================================================================

def compute_qa_statistics(
    qa_files: List[str],
    output_dir: str = "results/figures_and_tables",
) -> Dict:
    """Analyse the generated QA dataset: expert distribution, quality, topics."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all records
    records = []
    for f in qa_files:
        if Path(f).exists():
            records.extend(load_jsonl(f))
    logger.info(f"Loaded {len(records)} QA records")

    if not records:
        return {}

    # --- Aggregate ---
    expert_counter = Counter()
    topic_counter = Counter()
    quality_scores = []
    instr_lengths = []
    resp_lengths = []
    multi_expert_count = 0
    meta_expert_count = 0

    for r in records:
        meta = r.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

        # Experts
        experts = meta.get("selected_experts") or meta.get("experts") or []
        for e in experts:
            expert_counter[e] += 1
        if len(experts) > 1:
            multi_expert_count += 1

        # Meta-expert
        if meta.get("use_meta_expert"):
            meta_expert_count += 1

        # Topics
        topics = meta.get("core_topics") or []
        for t in topics:
            topic_counter[t] += 1

        # Quality
        qs = meta.get("quality_score", 0) or 0
        quality_scores.append(qs)

        # Lengths
        instr = r.get("instruction") or r.get("question") or ""
        resp = r.get("output") or r.get("answer") or r.get("response") or ""
        instr_lengths.append(len(instr))
        resp_lengths.append(len(resp))

    stats = {
        "total_records": len(records),
        "multi_expert_ratio": round(multi_expert_count / len(records), 4),
        "meta_expert_ratio": round(meta_expert_count / len(records), 4),
        "expert_types_used": len(expert_counter),
        "expert_distribution": dict(expert_counter.most_common()),
        "top_30_topics": dict(topic_counter.most_common(30)),
        "avg_quality_score": round(float(np.mean(quality_scores)), 4),
        "std_quality_score": round(float(np.std(quality_scores)), 4),
        "avg_instruction_length": round(float(np.mean(instr_lengths)), 1),
        "avg_response_length": round(float(np.mean(resp_lengths)), 1),
    }

    # Save JSON
    with open(out_path / "table2_qa_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Save CSV
    if HAS_PD:
        overview_rows = [
            {"Metric": "Total QA pairs", "Value": stats["total_records"]},
            {"Metric": "Multi-expert ratio", "Value": stats["multi_expert_ratio"]},
            {"Metric": "Meta-expert ratio", "Value": stats["meta_expert_ratio"]},
            {"Metric": "Expert types used", "Value": stats["expert_types_used"]},
            {"Metric": "Avg quality score", "Value": stats["avg_quality_score"]},
            {"Metric": "Std quality score", "Value": stats["std_quality_score"]},
            {"Metric": "Avg instruction length (chars)", "Value": stats["avg_instruction_length"]},
            {"Metric": "Avg response length (chars)", "Value": stats["avg_response_length"]},
        ]
        pd.DataFrame(overview_rows).to_csv(out_path / "table2_qa_overview.csv", index=False)

        expert_df = pd.DataFrame(
            [{"Expert": k, "Count": v} for k, v in expert_counter.most_common()]
        )
        expert_df.to_csv(out_path / "table2_expert_distribution.csv", index=False)

    # --- Figures ---
    if HAS_MPL:
        _plot_expert_bar(expert_counter, out_path)
        _plot_quality_hist(quality_scores, out_path)
        _plot_topic_bar(topic_counter, out_path)

    logger.info(f"Table 2 + figures saved: {out_path}")
    return stats


# =========================================================================
# Layer × Year heatmap (from chunks)
# =========================================================================

def plot_layer_year_heatmap(
    chunks_file: str = "data/processed_corpus/chunks_clean_fixed.jsonl",
    output_dir: str = "results/figures_and_tables",
):
    """Generate Layer × Year heatmap from chunk IDs."""
    if not HAS_MPL:
        logger.warning("matplotlib not available; skipping heatmap")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layer_year = defaultdict(Counter)

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id", "") or chunk.get("doc_id", "")

            layer = "Other"
            for lname in ["Layer1", "Layer2", "Layer3", "Layer4",
                          "第一层", "第二层", "第三层", "第四层"]:
                if lname in chunk_id:
                    layer = lname.replace("第一层", "Layer1").replace(
                        "第二层", "Layer2").replace(
                        "第三层", "Layer3").replace(
                        "第四层", "Layer4")
                    break

            years = re.findall(r"20[12]\d", chunk_id)
            for y in set(years):
                yr = int(y)
                if 2015 <= yr <= 2026:
                    layer_year[layer][yr] += 1

    if not layer_year:
        return

    layers = sorted(layer_year.keys())
    all_years = sorted({y for c in layer_year.values() for y in c})

    matrix = np.zeros((len(layers), len(all_years)))
    for i, lay in enumerate(layers):
        for j, yr in enumerate(all_years):
            matrix[i, j] = layer_year[lay].get(yr, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(all_years) * 0.8), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels([str(y) for y in all_years], rotation=45)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Year")
    ax.set_ylabel("Corpus Layer")
    ax.set_title("Chunk Distribution: Layer × Year")
    plt.colorbar(im, ax=ax, label="Chunk Count")
    plt.tight_layout()
    fig.savefig(out_path / "fig4_layer_year_heatmap.png", dpi=300)
    fig.savefig(out_path / "fig4_layer_year_heatmap.pdf")
    plt.close(fig)
    logger.info("Figure 4 saved")


# =========================================================================
# Internal plotting helpers
# =========================================================================

def _plot_expert_bar(expert_counter: Counter, out_path: Path):
    """Figure 1: Expert usage frequency."""
    names = [k for k, v in expert_counter.most_common()]
    counts = [v for k, v in expert_counter.most_common()]

    short_names = [n.replace("_expert", "").replace("_", "\n") for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Usage Count")
    ax.set_title("Expert Usage Frequency in CoDE Generation")

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(cnt), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path / "fig1_expert_usage.png", dpi=300)
    fig.savefig(out_path / "fig1_expert_usage.pdf")
    plt.close(fig)
    logger.info("Figure 1 saved")


def _plot_quality_hist(quality_scores: List[float], out_path: Path):
    """Figure 2: Quality score distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(quality_scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(quality_scores), color="red", linestyle="--",
               label=f"Mean = {np.mean(quality_scores):.3f}")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Quality Score Distribution of CoDE-Generated QA Pairs")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path / "fig2_quality_distribution.png", dpi=300)
    fig.savefig(out_path / "fig2_quality_distribution.pdf")
    plt.close(fig)
    logger.info("Figure 2 saved")


def _plot_topic_bar(topic_counter: Counter, out_path: Path, top_n: int = 20):
    """Figure 3: Top topic coverage."""
    top = topic_counter.most_common(top_n)
    names = [t[0] for t in top]
    counts = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), counts, color="coral")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top-{top_n} ESG Topics in Generated QA Pairs")
    plt.tight_layout()
    fig.savefig(out_path / "fig3_topic_coverage.png", dpi=300)
    fig.savefig(out_path / "fig3_topic_coverage.pdf")
    plt.close(fig)
    logger.info("Figure 3 saved")


# =========================================================================
# Comparison bar chart: CoDE vs Baselines
# =========================================================================

def plot_comparison_bar(
    comparison_json: str = "results/outputs/intrinsic_comparison.json",
    output_dir: str = "results/figures_and_tables",
    metrics: Optional[List[str]] = None,
):
    """Generate grouped bar chart comparing CoDE vs baselines on selected metrics."""
    if not HAS_MPL:
        return

    if metrics is None:
        metrics = ["domain_relevance", "distinct_2", "structural_completeness",
                   "factcheck", "avg_response_length"]

    comp_path = Path(comparison_json)
    if not comp_path.exists():
        logger.warning(f"Comparison file not found: {comp_path}")
        return

    with open(comp_path, "r", encoding="utf-8") as f:
        comparison = json.load(f)

    # Filter to main methods (skip ablation sub-results)
    main_methods = {k: v for k, v in comparison.items() if "ablation" not in k}
    if not main_methods:
        return

    method_names = list(main_methods.keys())
    n_methods = len(method_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for ax, metric in zip(axes, metrics):
        values = []
        for m in method_names:
            v = main_methods[m].get(metric, 0) or 0
            # Normalise response length for visual comparability
            if metric == "avg_response_length":
                v = v / 100.0  # scale to hundreds
            values.append(v)

        bars = ax.bar(range(n_methods), values, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(
            [n.replace("_", "\n") for n in method_names],
            rotation=45, ha="right", fontsize=8,
        )
        label = metric.replace("_", " ").title()
        if metric == "avg_response_length":
            label += " (×100)"
        ax.set_title(label, fontsize=10)

    plt.suptitle("Intrinsic Quality: CoDE vs Baselines", fontsize=13, y=1.02)
    plt.tight_layout()
    out = Path(output_dir)
    fig.savefig(out / "fig5_coe_vs_baselines.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig5_coe_vs_baselines.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 5 (comparison) saved")


# =========================================================================
# Ablation chart
# =========================================================================

def plot_ablation_results(
    ablation_summary: str = "results/ablation/ablation_summary.json",
    output_dir: str = "results/figures_and_tables",
):
    """Generate sub-plots for each ablation dimension."""
    if not HAS_MPL:
        return

    summary_path = Path(ablation_summary)
    if not summary_path.exists():
        logger.warning(f"Ablation summary not found: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        ablation = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dims = list(ablation.keys())
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        conditions = ablation[dim]
        labels = list(conditions.keys())
        avg_qs = [conditions[l].get("avg_quality", 0) for l in labels]

        short_labels = [l.split("_", 1)[-1] if "_" in l else l for l in labels]
        bars = ax.bar(range(len(labels)), avg_qs, color="teal", alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Avg Quality Score")
        ax.set_title(dim.replace("_", " ").title())

        for bar, v in zip(bars, avg_qs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Ablation Study Results", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out / "fig6_ablation_results.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig6_ablation_results.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 6 (ablation) saved")


# =========================================================================
# All-in-one runner
# =========================================================================

def generate_all_tables_and_figures(
    raw_corpus_dir: str = "data/raw_corpus",
    chunks_file: str = "data/processed_corpus/chunks_clean_fixed.jsonl",
    qa_files: Optional[List[str]] = None,
    comparison_json: str = "results/outputs/intrinsic_comparison.json",
    ablation_summary: str = "results/ablation/ablation_summary.json",
    output_dir: str = "results/figures_and_tables",
):
    """Generate all tables and figures for the paper."""
    if qa_files is None:
        qa_files = [
            "data/instructions/qa_pairs_complete_v3_1.5w.jsonl",
            "data/instructions/qa_pairs_complete_v3_2w.jsonl",
        ]

    logger.info("=" * 60)
    logger.info("Generating Table 1: Corpus Statistics")
    logger.info("=" * 60)
    compute_corpus_statistics(raw_corpus_dir, chunks_file, output_dir)

    logger.info("=" * 60)
    logger.info("Generating Table 2 + Figures 1-3: QA Statistics")
    logger.info("=" * 60)
    compute_qa_statistics(qa_files, output_dir)

    logger.info("=" * 60)
    logger.info("Generating Figure 4: Layer × Year Heatmap")
    logger.info("=" * 60)
    plot_layer_year_heatmap(chunks_file, output_dir)

    logger.info("=" * 60)
    logger.info("Generating Figure 5: CoDE vs Baselines")
    logger.info("=" * 60)
    plot_comparison_bar(comparison_json, output_dir)

    logger.info("=" * 60)
    logger.info("Generating Figure 6: Ablation Results")
    logger.info("=" * 60)
    plot_ablation_results(ablation_summary, output_dir)

    logger.info("=" * 60)
    logger.info("All tables and figures generated.")
    logger.info("=" * 60)


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    generate_all_tables_and_figures()
