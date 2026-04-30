# CoDE Expert Agent Summary

Summary table of all 11 expert agents in the Council of Domain Experts (CoDE) framework.
All experts share the same prompt structure: **persona → task → requirements → knowledge context → chunk text → output format**. Only the specific task description and output labels differ.

## Expert Agent Hierarchy

### Layer 1: General Understanding (Base Layer)

| Expert | Core Task | Output Labels |
|---|---|---|
| **Q&A Expert** | Generate a specific question–answer pair grounded in the text | Question / Answer |
| **Summary Expert** | Generate a summarization task (100–200 words) highlighting ESG content | Instruction / Summary |
| **Extraction Expert** | Extract structured ESG information (emissions, employee count, etc.) | Instruction / Extraction result |
| **Classification Expert** | Classify text by ESG dimension (E/S/G/Mixed) with rationale | Instruction / Classification result |
| **Analysis Expert** | Provide in-depth cross-domain analysis and critical assessment | Instruction / Analysis result |

### Layer 2: Domain Reasoning (Analysis Layer)

| Expert | Core Task | Output Labels |
|---|---|---|
| **Temporal Analysis Expert** | Analyze multi-year trends, commitment consistency, and data gaps | Instruction / Answer |
| **Benchmark Comparison Expert** | Compare company data against industry averages and flag anomalies | Instruction / Answer |
| **Greenwashing Detection Expert** | Identify misleading claims and quantify greenwashing risk (0–1 scale) | Instruction / Answer |

### Layer 3: Verification Layer

| Expert | Core Task | Output Labels |
|---|---|---|
| **Standard Alignment Expert** | Identify followed standards (GRI, TCFD, SASB, etc.) and compliance gaps | Instruction / Answer |
| **Consistency Verification Expert** | Detect data contradictions and selective disclosure patterns | Instruction / Answer |
| **Knowledge Graph Expert** | Extract entities and relations; detect broken commitment–action links | Instruction / Answer |

### MetaExpert Orchestrator

The **MetaExpert** is not a domain expert but an orchestration agent. It:
1. Designs tailored work instructions for subordinate experts based on chunk content and collaboration context
2. Evaluates expert outputs on four dimensions: Correctness (0.4), Completeness (0.3), Relevance (0.2), Structure (0.1)
3. Provides iterative feedback if quality falls below threshold
4. Selects the best candidate across feedback rounds

## Prompt Modes

Each expert supports two prompt modes:
- **Default mode**: The expert generates a task directly from the text chunk
- **Dynamic mode**: The expert receives an additional work instruction from the MetaExpert and aligns its output to that instruction

The dynamic mode prepends `Work instruction: {dynamic_instruction}` to the requirements block; the rest of the prompt is identical.

## Full Prompt Templates

See the `prompts/` directory for the complete prompt text of each expert.
