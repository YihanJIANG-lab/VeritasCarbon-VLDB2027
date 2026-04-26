# Data Availability Statement

This document describes the data release strategy for the VeritasCarbon project, in compliance with PVLDB Volume 20 artifact availability requirements.

## Release Philosophy

We adopt a **tiered release strategy** that balances reproducibility, transparency, and practical constraints (copyright, storage size):

| Tier | Content | Location | Size |
|------|---------|----------|------|
| **Tier 1** | Representative sample (500 QA pairs) | In this repository (`data/sample/`) | ~2.2 MB |
| **Tier 2** | Corpus metadata & provenance | In this repository (`data/raw_corpus/CORPUS_MANIFEST.md`) | ~4 KB |
| **Tier 3** | Full dataset (35,009 QA pairs) | Generated via provided scripts | ~154 MB |

---

## Tier 1: Sample Dataset (`data/sample/`)

We provide a **statistically representative sample of 500 QA pairs** (`veritascarbon_sample_500.jsonl`) to enable immediate inspection of data quality, format, and provenance without downloading large files.

- **Sampling method**: Simple random sampling (`random.seed(42)`) from the full 35,009-pair pool
- **Schema**: Unified `instruction` / `input` / `output` / `source_chunk_id` / `metadata`
- **Rich metadata**: Each record includes `quality_score`, `selected_experts`, `collaboration_mode`, `knowledge_items_count`, etc.
- **Usage**: Load directly for qualitative analysis or as a sanity-check for the generation pipeline

This sample is sufficient to verify all claims about data structure, traceability, and metadata richness.

---

## Tier 2: Corpus Metadata

`data/raw_corpus/CORPUS_MANIFEST.md` contains the complete inventory of all **17,721 source documents** used to construct the dataset, including:

- Document IDs, titles, publication years, and layer assignments
- Source URLs or archive references (where publicly available)
- Chunk-to-document mapping logic

This ensures full **data provenance** even when the raw documents themselves cannot be redistributed due to copyright.

---

## Tier 3: Full Dataset

The complete **VeritasCarbon-ESG-35K** dataset (35,009 instruction–response pairs) is **reproducible from source** using the code and notebooks in this repository.

### Why not included directly?

1. **Size**: The raw generated data totals ~154 MB of JSONL files. While manageable, the *raw corpus* (source PDFs/TXTs) is ~2.7 GB and contains copyrighted material (CSR reports, regulatory documents) that we are not authorized to redistribute.
2. **Copyright**: Layer 2 (CSR reports) and Layer 3 (regulatory guidelines) documents are owned by their respective issuers. We provide sample excerpts under fair use but cannot distribute the full corpus.
3. **Reproducibility priority**: The generation pipeline is fully deterministic (fixed `random.seed(42)` for sampling steps). Running the notebooks on the same corpus produces identical outputs.

### How to obtain the full dataset

**Option A — Reproduce from scratch (recommended for reproducibility evaluation):**

1. Obtain the raw ESG corpus (~2.7 GB). The manifest in `CORPUS_MANIFEST.md` lists all 17,721 documents with source information. Many Layer 2 CSR reports are publicly available from stock exchanges (e.g., CSMAR, CNINFO). Layer 1 textbooks and Layer 4 industry analyses can be sourced from publishers or open-access repositories.
2. Run the notebooks in order:
   ```bash
   jupyter notebook notebooks/01_DataPreprocess.ipynb      # → chunks
   jupyter notebook notebooks/02_InstructionGeneration_v3.ipynb  # → 35,009 QA pairs
   ```
3. The default configuration (`configs/config.yaml`) and all hyperparameters are fixed.

**Option B — Request pre-generated data:**

Due to GitHub storage and LFS quota constraints, we host the full `qa_pairs_complete_v3_*.jsonl` files via an external archival repository. Please open a GitHub Issue or contact the authors for a download link.

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
