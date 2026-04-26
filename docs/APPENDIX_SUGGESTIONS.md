# Appendix Content Suggestions (VLDB 2027)

Below is a ranked list of appendix materials that strengthen reproducibility, transparency, and reviewer confidence. Items marked ⭐ are **high-impact** and strongly recommended.

---

## Tier 1: Essential (Strongly Recommended)

### ⭐ A.1 Data Availability Statement (1 page)

Place the exact text from `DATA_AVAILABILITY.md` into the paper appendix. VLDB reviewers expect to see this explicitly.

**Key sentence to include**:
> *"The full VeritasCarbon-ESG-35K dataset (35,009 pairs) is available at [HF URL] under CC BY-SA 4.0. A representative 2,000-pair sample is included in the GitHub repository to replicate Table 2 without downloading the full corpus."*

### ⭐ A.2 30-Minute Reproducibility Checklist (1 page)

A condensed version of `REPRODUCIBILITY.md` formatted as a table:

| Claim | Verification Command | Expected Output | Time |
|-------|---------------------|-----------------|------|
| Table 2 CoDE ROUGE-L | `cat results/outputs/intrinsic_comparison.json` | 0.3380 | 30 sec |
| ... | ... | ... | ... |

This signals to reviewers: *"You can verify everything in under 30 minutes."*

### ⭐ A.3 Prompt Templates for All 11 Expert Agents (2–3 pages)

This is **the single most requested artifact** in multi-agent LLM papers. Reviewers want to see:

- The exact system prompt for each expert (Analysis, Evaluation, Comparison, etc.)
- The MetaExpert prompt (orchestration + feedback)
- The knowledge-injection prompt template

**Format**: One prompt per page, in monospace font, with clear labels.

> *Why this matters*: Prompt engineering is the core methodology. Hiding prompts = hiding the method.

---

## Tier 2: High Value (Recommended)

### A.4 Complete Hyperparameter Table (0.5 page)

| Parameter | Value | Location |
|-----------|-------|----------|
| Model | Qwen2-72B-Instruct | `configs/config.yaml` |
| Quantization | 4-bit (NF4) | Unsloth |
| Temperature | 0.7 | `coe_framework_02_03.py` |
| Max tokens (instruction) | 512 | — |
| Max tokens (response) | 1024 | — |
| Chunk size | 2,000 chars | `text_chunker_01_03.py` |
| Chunk overlap | 200 chars | — |
| Expert selection threshold | 0.5 | `expert_selector_02_01.py` |
| Feedback rounds (default) | 2 | `meta_expert_02_09.py` |
| Quality threshold (accept) | 0.6 | — |
| Random seed | 42 | All sampling steps |

### A.5 Expert Selection Decision Logic (0.5 page)

Document the rule-based expert selector in pseudocode or flowchart form. Reviewers need to understand **why** specific experts are assigned to specific chunks.

Example:
```
IF chunk contains "carbon emission" AND "target" → assign [analysis_expert, evaluation_expert]
IF chunk contains "employee" AND "training" → assign [extraction_expert, comparison_expert]
```

### A.6 Error Analysis: High-Quality vs. Low-Quality Examples (1 page)

Pick 2–3 concrete examples from the dataset:
- **High-quality** (quality_score > 0.8): Show what makes it good
- **Low-quality** (quality_score < 0.5): Show failure modes and why MetaExpert couldn't fix it

Include the source chunk, assigned experts, and final quality score.

### A.7 Dataset Schema & Field Definitions (0.5 page)

A formal specification of the JSON schema, especially the `metadata` object:

```json
{
  "quality_score": "Float in [0,1]. Composite of correctness (0.4), completeness (0.3), relevance (0.2), structure (0.1)",
  "selected_experts": "List of expert agent IDs invoked for this record",
  "collaboration_mode": "parallel | sequential",
  ...
}
```

---

## Tier 3: Bonus (Impressive if Space Permits)

### A.8 Latency & Cost Breakdown (0.5 page)

From `results/scalability/scalability_results.json`:

| Stage | Avg Time / Record | Total (35K) |
|-------|------------------|-------------|
| Expert Selection | 0.02 s | ~12 min |
| Knowledge Retrieval | 0.15 s | ~88 min |
| Parallel Generation | 2.8 s | ~27 hrs |
| MetaExpert Feedback (R=2) | 1.9 s | ~18 hrs |
| **Total** | **~5 s** | **~46 hrs** |

*Note: Times measured on A100 (80GB) with Unsloth 4-bit Qwen2-72B-Instruct.*

### A.9 Corpus License & Copyright Declaration (0.3 page)

Explicitly state:
- Which layers contain copyrighted material
- Why redistribution is not possible
- How readers can obtain the raw documents (CSMAR, CNINFO, etc.)
- Fair use justification for the sample excerpts

This preempts reviewer concerns about legal compliance.

### A.10 Comparison to Existing ESG Datasets (0.5 page)

A small table positioning VeritasCarbon-ESG-35K against prior work:

| Dataset | Size | Language | Traceable | Multi-task | Domain |
|---------|------|----------|-----------|------------|--------|
| ESG-BERT Corpus | 50K | EN | ❌ | ❌ | Classification |
| FiQA | 17K | EN | ❌ | ✅ | Finance QA |
| C-ESG | 8K | ZH | ❌ | ❌ | Classification |
| **VeritasCarbon-ESG-35K** | **35K** | **ZH** | **✅** | **✅** | **Instruction** |

*(Use real datasets from your literature review.)*

---

## Tier 4: Nice-to-Have

### A.11 Ablation Visualization (Figure)

If not already in the main paper, include a radar chart or bar chart showing the 4 ablation dimensions side by side.

### A.12 Generated vs. Human-Written Comparison

If you have any human-written ESG QAs for comparison, show 1–2 examples highlighting where CoDE excels (e.g., multi-layer reasoning) or where humans still win (e.g., nuanced stakeholder analysis).

---

## What NOT to Put in the Appendix

- ❌ Raw JSONL dumps (use HF/GitHub links instead)
- ❌ Full LaTeX tables that are already in the paper (redundant)
- ❌ Code listings longer than 1 page (point to GitHub)
- ❌ Excessive ablation dimensions beyond the 4 core ones (dilutes focus)

---

## Suggested Appendix Order

1. **A.1** Data & Code Availability
2. **A.2** Reproducibility Checklist
3. **A.3** Prompt Templates
4. **A.4** Hyperparameters
5. **A.5** Expert Selection Logic
6. **A.6** Error Analysis
7. **A.7** Dataset Schema
8. **A.8** Latency Breakdown *(optional)*
9. **A.9** Copyright Declaration *(optional)*

**Total estimated length**: 6–8 pages. This is well within typical VLDB appendix limits.
