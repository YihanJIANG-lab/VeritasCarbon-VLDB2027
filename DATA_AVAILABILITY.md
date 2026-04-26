# Data Availability Statement

This document describes the data release strategy for the VeritasCarbon project, in compliance with PVLDB Volume 20 artifact availability requirements.

## Release Philosophy

We adopt a **tiered release strategy** that balances reproducibility, transparency, and practical constraints (copyright, storage size):

| Tier | Content | Location | Size |
|------|---------|----------|------|
| **Tier 1** | Representative sample (2,000 QA pairs) | In this repository (`data/sample/`) | ~8.7 MB |
| **Tier 2** | Corpus metadata & provenance | In this repository (`data/raw_corpus/CORPUS_MANIFEST.md`) | ~4 KB |
| **Tier 3** | Full dataset (35,009 QA pairs) | [Hugging Face Datasets](https://huggingface.co/datasets/Yihan-JIANG/VeritasCarbon-ESG-35K) | ~153 MB |

The **2,000-pair sample size** is chosen to match the evaluation protocol in the paper (Table 2), enabling reviewers to directly replicate the main comparison experiment without downloading the full corpus.

---

## Tier 1: Sample Dataset (`data/sample/`)

We provide a **statistically representative sample of 2,000 QA pairs** (`veritascarbon_sample_2000.jsonl`) to enable immediate inspection of data quality, format, and provenance, **and** to directly replicate the main intrinsic evaluation (Table 2).

- **Sampling method**: Simple random sampling (`random.seed(42)`) from the full 35,009-pair pool
- **Schema**: Unified `instruction` / `input` / `output` / `source_chunk_id` / `metadata`
- **Rich metadata**: Each record includes `quality_score`, `selected_experts`, `collaboration_mode`, `knowledge_items_count`, etc.
- **Usage**:
  - **Qualitative inspection**: Load directly to review data format and quality distribution
  - **Quantitative replication**: Run `src/instruction_generation/intrinsic_evaluation_03_03.py` with `--max_per_dataset 2000` on this sample to reproduce the CoDE column of Table 2

---

## Tier 2: Corpus Metadata

`data/raw_corpus/CORPUS_MANIFEST.md` contains the complete inventory of all **17,721 source documents** used to construct the dataset, including:

- Document IDs, titles, publication years, and layer assignments
- Source URLs or archive references (where publicly available)
- Chunk-to-document mapping logic

This ensures full **data provenance** even when the raw documents themselves cannot be redistributed due to copyright.

---

## Tier 3: Full Dataset (Hugging Face)

The complete **VeritasCarbon-ESG-35K** dataset (35,009 instruction–response pairs) is hosted on **Hugging Face Datasets**:

> **https://huggingface.co/datasets/Yihan-JIANG/VeritasCarbon-ESG-35K**

### Why not all in GitHub?

1. **Size**: The full generated data is ~153 MB. Combined with raw corpus metadata and code, this is manageable but better served by a dedicated data platform.
2. **Copyright**: The *raw corpus* (source PDFs/TXTs, ~2.7 GB) contains copyrighted material (CSR reports, regulatory documents) that we are not authorized to redistribute. We provide sample excerpts under fair use but cannot distribute the full corpus.
3. **Community standard**: Hugging Face Datasets is the de facto standard for releasing instruction-following datasets in the AI/ML community, offering persistent DOI-like identifiers and seamless `datasets` library integration.

### How to access

```python
from datasets import load_dataset

dataset = load_dataset("Yihan-JIANG/VeritasCarbon-ESG-35K", split="train")
print(len(dataset))  # 35009
```

Or download the JSONL directly from the repository's Files tab.

### How to reproduce from scratch

1. Obtain the raw ESG corpus (~2.7 GB). The manifest in `CORPUS_MANIFEST.md` lists all 17,721 documents with source information. Many Layer 2 CSR reports are publicly available from stock exchanges (e.g., CSMAR, CNINFO). Layer 1 textbooks and Layer 4 industry analyses can be sourced from publishers or open-access repositories.
2. Run the notebooks in order:
   ```bash
   jupyter notebook notebooks/01_DataPreprocess.ipynb      # → chunks
   jupyter notebook notebooks/02_InstructionGeneration_v3.ipynb  # → 35,009 QA pairs
   ```
3. The default configuration (`configs/config.yaml`) and all hyperparameters are fixed.

---

## Data Format

### Sample / Full Dataset (`*.jsonl`)

Each line is a JSON object:

```json
{
  "instruction": "Evaluate the consistency between the company's carbon reduction commitments and its actual performance...",
  "input": "<source chunk text (up to ~2,000 chars)>",
  "output": "<generated response (up to ~1,500 chars)>",
  "source_chunk_id": "Layer2_CSR_2024_000920_chunk_21",
  "metadata": {
    "quality_score": 0.647,
    "selected_experts": ["analysis_expert", "evaluation_expert"],
    "collaboration_mode": "parallel",
    "max_iterations": 2,
    "knowledge_items_count": 3,
    "model": "Qwen2-72B-Instruct",
    "framework": "CoDE"
  }
}
```

### Corpus Chunks (`chunks_sampled_20000_by_year.jsonl`)

Each line contains a semantically segmented text chunk with year/layer metadata. Used as input to the instruction generation pipeline.

---

## License

- **Code**: MIT License (see `LICENSE`)
- **Generated dataset (VeritasCarbon-ESG-35K)**: CC BY-SA 4.0
- **Raw corpus samples**: Provided under fair use for research purposes; full documents remain under their original publishers' copyrights.

---

## Contact

For questions about data access, reproduction, or licensing, please open an issue on GitHub or contact the corresponding author.
