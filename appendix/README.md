# Appendix: Supplementary Materials for VeritasCarbon (VLDB 2027)

This directory contains supplementary materials referenced in the paper
*"VeritasCarbon: Scalable and Traceable Instruction Data Generation for ESG Domains"*.

## Contents

| File / Directory | Description |
|---|---|
| `prompts/` | All 11 expert agent prompt templates + MetaExpert orchestrator prompt |
| `quality_scoring.py` | Reference implementation of the quality scoring function Q(o, c) |
| `feature_extraction.py` | Feature vector computation for layered expert selection |
| `expert_summary.md` | Summary table of all 11 CoDE expert agents |
| `A1_data_availability.tex` | Data and code availability statement |
| `A2_reproducibility_checklist.tex` | 30-minute reproducibility checklist |
| `A3_prompt_templates.tex` | Prompt templates (LaTeX formatted) |
| `A4_hyperparameters.tex` | Complete hyperparameter configuration |
| `A5_expert_selection.tex` | Expert selection logic |
| `A6_error_analysis.tex` | Error analysis with high/low quality examples |
| `A7_schema.tex` | Dataset JSON schema |
| `A8_latency.tex` | Latency and cost breakdown |
| `dataset_contribution.tex` | Dataset contribution statement |

## Prompt Templates (`prompts/`)

Individual prompt files for each of the 11 expert agents and the MetaExpert orchestrator.
Each file contains both **Default mode** and **Dynamic mode** prompt templates.

| File | Expert | Tier |
|---|---|---|
| `01_qa_expert.txt` | Q&A Expert | General Understanding |
| `02_summary_expert.txt` | Summary Expert | General Understanding |
| `03_extraction_expert.txt` | Extraction Expert | General Understanding |
| `04_classification_expert.txt` | Classification Expert | General Understanding |
| `05_analysis_expert.txt` | Analysis Expert | General Understanding |
| `06_temporal_analysis_expert.txt` | Temporal Analysis Expert | Domain Reasoning |
| `07_benchmark_comparison_expert.txt` | Benchmark Comparison Expert | Domain Reasoning |
| `08_greenwashing_detection_expert.txt` | Greenwashing Detection Expert | Domain Reasoning |
| `09_standard_alignment_expert.txt` | Standard Alignment Expert | Verification |
| `10_knowledge_graph_expert.txt` | Knowledge Graph Expert | Verification |
| `11_consistency_verification_expert.txt` | Consistency Verification Expert | Verification |
| `12_meta_expert.txt` | MetaExpert Orchestrator | Orchestration |

## Quality Scoring Function Q(o, c)

`quality_scoring.py` provides the reference implementation of the composite quality
scoring function defined in Section 2.3 of the paper:

```
Q(o, c) = w_f * Fidelity(o,c) + w_r * Relevance(o,c) + w_g * Grounding(o,c)
         + w_s * Structure(o) + w_d * Diversity(o)
```

Default weights: `w_f=0.25, w_r=0.25, w_g=0.20, w_s=0.15, w_d=0.15`

## Feature Vector Computation

`feature_extraction.py` implements the rule-based feature extractor used for
layered expert selection (Section 3.3). The feature vector includes:
- Keyword presence (carbon, emission, energy, governance, ...)
- Numerical data density
- Temporal markers (years, quarters)
- Standard references (GRI, TCFD, SASB, ISO 14001, CDP)
- Comparative language
- Entity types

## Citation

If you use these materials, please cite:
```bibtex
@article{jiang2027veritascarbon,
  title={VeritasCarbon: Scalable and Traceable Instruction Data Generation for ESG Domains},
  author={Jiang, Yihan and Peng, Fei and Woon, Kok Sin and Ren, Qianping and Xu, Yichang and Yang, Yujing},
  journal={Proceedings of the VLDB Endowment},
  year={2027}
}
```
