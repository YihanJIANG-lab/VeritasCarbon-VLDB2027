# VeritasCarbon

> **VeritasCarbon: A Scalable Multi-Agent Framework for Generating Traceable ESG Instruction Data**
>
> **Submitted to VLDB 2027** 鈥?Research Track

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Overview

VeritasCarbon converts **17,721 ESG disclosure documents** into **35,009 traceable instruction鈥搑esponse pairs** through a multi-agent pipeline. At its core is **CoDE (Council of Domain Experts)**, a framework that organizes 11 specialized expert agents into a four-layer hierarchy, coordinating their generation through parallel or sequential collaboration with iterative MetaExpert feedback.

**Key Results:**
- 4.0脳 higher ROUGE-L, 4.6脳 higher BLEU-4, 14.6脳 higher domain relevance vs. best baseline
- All experiments use the same Qwen2-72B-Instruct model (4-bit quantized, on-premises) for fair comparison
- Every generated QA pair retains source mapping for full data provenance

<p align="center">
  <img src="paper/figures/fig1_pipeline_clean.png" width="85%" alt="VeritasCarbon Pipeline"/>
  <br>
  <em>Figure 1: The VeritasCarbon pipeline 鈥?from raw ESG documents to traceable instruction data.</em>
</p>

## Repository Structure

```
VeritasCarbon-VLDB2027/
鈹溾攢鈹€ configs/
鈹?  鈹斺攢鈹€ config.yaml              # Central configuration (model, CoDE, paths)
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ data_processing/         # Document parsing, chunking, quality check
鈹?  鈹?  鈹溾攢鈹€ document_parser_01_02.py
鈹?  鈹?  鈹溾攢鈹€ text_chunker_01_03.py
鈹?  鈹?  鈹斺攢鈹€ data_quality_check_01_04.py
鈹?  鈹溾攢鈹€ instruction_generation/  # CoDE framework and experiments
鈹?  鈹?  鈹溾攢鈹€ expert_selector_02_01.py      # 4-layer expert selection
鈹?  鈹?  鈹溾攢鈹€ domain_knowledge_02_02.py     # ESG knowledge injection
鈹?  鈹?  鈹溾攢鈹€ coe_framework_02_03.py        # Multi-expert collaboration
鈹?  鈹?  鈹溾攢鈹€ expert_agents_02_04.py        # 11 specialized expert types
鈹?  鈹?  鈹溾攢鈹€ meta_expert_02_09.py          # MetaExpert orchestration
鈹?  鈹?  鈹溾攢鈹€ baseline_local_03_01.py       # 3 baseline methods
鈹?  鈹?  鈹溾攢鈹€ ablation_local_03_02.py       # 4-dimension ablation
鈹?  鈹?  鈹溾攢鈹€ intrinsic_evaluation_03_03.py # 7 intrinsic metrics
鈹?  鈹?  鈹斺攢鈹€ dataset_statistics_03_04.py   # Tables & figures generation
鈹?  鈹溾攢鈹€ evaluation/
鈹?  鈹溾攢鈹€ training/                  # QLoRA fine-tuning and evaluation
鈹?  鈹?  鈹溾攢鈹€ data_loader_03_02.py
鈹?  鈹?  鈹溾攢鈹€ model_evaluator_03_04.py
鈹?  鈹?  鈹溾攢鈹€ model_registry_03_01.py
鈹?  鈹?  鈹斺攢鈹€ qlora_trainer_03_03.py
鈹?  鈹斺攢鈹€ utils/
鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ raw_corpus/              # Sample ESG documents (full corpus: 17,721 docs)
鈹?  鈹?  鈹溾攢鈹€ Layer1/samples/      # Domain textbooks (2 samples)
鈹?  鈹?  鈹溾攢鈹€ Layer2/samples/      # CSR reports (2 samples + metadata)
鈹?  鈹?  鈹溾攢鈹€ Layer3/samples/      # Regulatory guidelines (2 samples)
鈹?  鈹?  鈹溾攢鈹€ Layer4/samples/      # Industry analyses (2 samples)
鈹?  鈹?  鈹斺攢鈹€ CORPUS_MANIFEST.md   # Full listing of all 17,721 documents
鈹?  鈹溾攢鈹€ processed_corpus/        # Semantically segmented chunks
鈹?  鈹?  鈹斺攢鈹€ chunks_sampled_20000_by_year.jsonl
鈹?  鈹溾攢鈹€ instructions/            # Generated QA pairs (full set on Hugging Face)
鈹?  鈹?  鈹溾攢鈹€ qa_pairs_complete_v3_1.5w.jsonl  (15,000 pairs)
鈹?  鈹?  鈹斺攢鈹€ qa_pairs_complete_v3_2w.jsonl    (20,000 pairs)
鈹?  鈹溾攢鈹€ instruction_datasets/
鈹?  鈹?  鈹斺攢鈹€ train.jsonl           # Final training set
鈹?  鈹斺攢鈹€ sample/                   # Representative 2,000-pair sample (in repo)
鈹?      鈹斺攢鈹€ veritascarbon_sample_2000.jsonl
鈹溾攢鈹€ results/
鈹?  鈹溾攢鈹€ baselines/               # Direct / Self-Instruct / WizardLM-Evol
鈹?  鈹溾攢鈹€ ablation/                # Expert count / Collaboration / Feedback / Knowledge
鈹?  鈹溾攢鈹€ figures_and_tables/      # Generated figures and LaTeX tables
鈹?  鈹斺攢鈹€ outputs/                 # Intrinsic evaluation CSVs
鈹溾攢鈹€ notebooks/
鈹?  鈹溾攢鈹€ 01_DataPreprocess.ipynb
鈹?  鈹溾攢鈹€ 02_InstructionGeneration_v3.ipynb
鈹?  鈹溾攢鈹€ 03_VLDB2027_Experiments.ipynb
鈹?  鈹斺攢鈹€ 03_VLDB2027_Experiments_output.ipynb
鈹溾攢鈹€ scripts/                     # Generation and monitoring utilities
鈹溾攢鈹€ paper/figures/               # Paper figures (cleaned)
鈹溾攢鈹€ requirements.txt
鈹斺攢鈹€ README.md
```

## Installation

```bash
# Clone
git clone https://github.com/YihanJIANG-lab/VeritasCarbon-VLDB2027.git
cd VeritasCarbon-VLDB2027

# Create environment
conda create -n VeritasCarbon python=3.10 -y
conda activate VeritasCarbon
pip install -r requirements.txt

# Download model (Qwen2-72B-Instruct, 4-bit quantized via Unsloth)
# Place at: models/Qwen2-72B-Instruct/
```

## Quick Start

### 1. Data Preprocessing
```bash
# Process raw ESG documents into semantic chunks
jupyter notebook notebooks/01_DataPreprocess.ipynb
```

### 2. Instruction Generation (CoDE Framework)
```bash
# Generate QA pairs using Council of Domain Experts
jupyter notebook notebooks/02_InstructionGeneration_v3.ipynb
```

### 3. Run Experiments
```bash
# Baselines, ablation, intrinsic evaluation
jupyter notebook notebooks/03_VLDB2027_Experiments.ipynb
```

## Reproducibility & Artifact Evaluation

This package supports three levels of reproducibility aligned with PVLDB / ACM artifact badging:

| Track | Scope | Command / Path | Expected Time |
|-------|-------|----------------|---------------|
| **A 鈥?Results Replicated** | Reproduce main comparison (Table 2) and intrinsic metrics from the 2,000-pair sample | `notebooks/03_VLDB2027_Experiments.ipynb` (Section 1鈥?) | ~30 min on A800 |
| **B 鈥?Full Evaluation** | Reproduce all tables and figures (1鈥?) using pre-computed results | `results/figures_and_tables/` + `results/outputs/` | ~5 min (rendering only) |
| **C 鈥?Full Regeneration** | Re-run the entire pipeline from raw documents to 35,009 QA pairs | `notebooks/01_DataPreprocess.ipynb` 鈫?`02_InstructionGeneration_v3.ipynb` | See Section 4.4 of the paper |

All experiments use **random seed 42** and the same **Qwen2-72B-Instruct (4-bit)** model for fair comparison. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for step-by-step instructions.

## CoDE Framework

<p align="center">
  <img src="paper/figures/fig2_code_arch_clean.png" width="85%" alt="CoDE Architecture"/>
  <br>
  <em>Figure 2: CoDE internal architecture 鈥?(A) 4-layer expert hierarchy, (B) MetaExpert orchestration, (C) collaboration modes, (D) feedback loop.</em>
</p>

The CoDE (Council of Domain Experts) framework operates in three stages:

1. **Layered Expert Selection**: 11 agents organized into 4 layers (Base 鈫?Analysis 鈫?Verification 鈫?Graph). For each chunk, a feature vector triggers layer-by-layer activation, truncated to K experts (default K=3).

2. **Multi-Expert Collaboration**: Selected experts collaborate in parallel (independent generation + voting) or sequential (context-passing chain) mode.

3. **MetaExpert Feedback**: The MetaExpert extracts topics, synthesizes work instructions, and runs R feedback rounds (default R=2) with quality threshold 蟿=0.7.

### Expert Types (11 Specialists)

| Layer | Experts | Activation |
|-------|---------|------------|
| Base (Layer 1) | QA, Summary, Extraction, Classification, Analysis | Always 鈮? |
| Analysis (Layer 2) | Temporal, Benchmark, Greenwashing | Feature 鈮?0.3 |
| Verification (Layer 3) | Consistency, Standard Alignment | Numerical/standards |
| Graph (Layer 4) | Knowledge Graph | Entity-relation 鈮?0.5 |

## Dataset: VeritasCarbon-ESG-35K

| Statistic | Value |
|-----------|-------|
| Total QA pairs | 35,009 |
| Source documents | 17,721 |
| Semantic chunks | 20,000 |
| Expert types | 11 |
| Avg. instruction length | 106.5 chars |
| Avg. response length | 380.4 chars |
| Quality score (mean 卤 std) | 0.667 卤 0.103 |

### Data Availability

We release the dataset under a **tiered strategy** (see [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for full details):

- **Sample (2,000 pairs)**: Included directly in this repository (`data/sample/veritascarbon_sample_2000.jsonl`). Drawn via `random.seed(42)` from the full pool; matches the evaluation protocol in Table 2 and is sufficient to replicate the main comparison experiment.
- **Full dataset (35,009 pairs)**: Hosted on [Hugging Face Datasets](https://huggingface.co/datasets/Yihan-JIANG/VeritasCarbon-ESG-35K) (`~153 MB`). Loadable in one line: `load_dataset("Yihan-JIANG/VeritasCarbon-ESG-35K")`.
- **Reproducible from source**: The raw corpus (~2.7 GB) contains copyrighted material and cannot be redistributed in full; we provide `CORPUS_MANIFEST.md` with complete provenance. Running the provided notebooks regenerates the identical 35,009-pair dataset.

**Format** (JSONL):
```json
{
  "instruction": "Based on the reported emissions data, analyze the trend...",
  "response": "The company's Scope 1 emissions decreased by 12.3%...",
  "metadata": {
    "chunk_id": "Layer1_doc_042_chunk_007",
    "expert_type": "analysis",
    "quality_score": 0.72,
    "source_mapping": ["Layer1/doc_042"]
  }
}
```

## Experimental Results

### Main Comparison (Table 1)

| Method | ROUGE-L | BLEU-4 | Distinct-2 | Domain Rel. | FactCheck |
|--------|---------|--------|------------|-------------|-----------|
| Direct Prompting | 0.0838 | 0.0424 | 0.0124 | 0.0197 | 0.8702 |
| Self-Instruct | 0.0407 | 0.0098 | 0.0033 | 0.0025 | 0.9336 |
| WizardLM-Evol | 0.0376 | 0.0119 | 0.0038 | 0.0249 | 0.5508 |
| **CoDE (Ours)** | **0.3380** | **0.1932** | **0.1061** | **0.3637** | **0.9478** |

### Key Ablation Findings

- **Expert Count**: K=3 optimal (quality 0.6453, +4.5% over K=1)
- **Collaboration**: Parallel best (quality 0.6473, +4.8% over none)
- **Feedback Rounds**: R=0 鈫?R=2 shows clear improvement (quality 0.6264 鈫?0.6494, +3.7%)
- **Knowledge Injection**: +1.2% quality improvement (0.6273 鈫?0.6349)

## Configuration

Edit `configs/config.yaml`:

```yaml
model:
  name: Qwen2-72B-Instruct
  quantization: 4bit
  framework: unsloth

code:
  max_experts: 3          # K
  collaboration: parallel  # parallel | sequential
  feedback_rounds: 2       # R
  quality_threshold: 0.7   # 蟿

seed: 42
```

## Raw Corpus

The full corpus (17,721 documents, ~2.7 GB) is not included due to copyright.
Sample files are provided in `data/raw_corpus/Layer*/samples/`.
See `data/raw_corpus/CORPUS_MANIFEST.md` for the complete file listing.

**Corpus Layers:**

| Layer | Description | Documents | Chunks |
|-------|-------------|-----------|--------|
| Layer 1 | Domain textbooks | 97 | 8,877 |
| Layer 2 | CSR reports (2006鈥?024) | 17,425 | 5,395 |
| Layer 3 | Regulatory guidelines | 92 | 3,237 |
| Layer 4 | Industry analyses | 107 | 2,491 |

## Citation

```bibtex
@misc{jiang2026veritascarbon,
  title     = {VeritasCarbon: Traceable Instruction Data Generation for ESG Domain via a Council of Domain Experts},
  author    = {Jiang, Yihan and Peng, Fei and Woon, Kok Sin and Ren, Qianping and Xu, Yichang and Yang, Yujing},
  howpublished = {Under review at PVLDB Vol. 20 (VLDB 2027)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The generated dataset (VeritasCarbon-ESG-35K) is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
