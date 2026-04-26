# SIGMOD Venue Distillation — 交接文档

**日期**: 2026-04-08  
**来源**: AutoResearch Pipeline (`/hpc2hdd/home/yjiang909/AutoResearch-VSC+Copilot-backup-2026-03-31/AutoResearch-VSC+Copilot/veritascarbon_distill/`)  
**目标**: 为 VeritasCarbon SIGMOD 投稿提供基于数据驱动的审稿标准蒸馏结果

---

## 1. 做了什么

针对 VeritasCarbon 项目投稿 SIGMOD，完成了完整的 **会议风格蒸馏 (Venue Distillation)** 流程：

1. **收集论文** — 通过 OpenAlex API 爬取 60 篇 SIGMOD 2022-2026 年接收论文 (关键词: ESG, carbon, sustainability, environmental)，同时通过 OpenReview 爬取 120 篇 ICLR/NeurIPS 2024-2025 被拒论文作为负例
2. **五层蒸馏** — 使用 LLM (GPT-4o) + 正则/API 对每篇论文进行多维度分析,聚合统计规律:
   - **Layer 1 — Rigor Facets** (实验严谨性): ablation 18%, real_data 77%, scalability 45%, e2e_eval 85%
   - **Layer 2 — Idea DNA** (创新类型分布): new_formulation 60%, new_application 27%, paradigm_shift 10%
   - **Layer 3 — Narrative Structure** (叙事结构): 图表频率、section 流向、intro 漏斗模式
   - **Layer 4 — Rejection Anti-Patterns** (被拒反模式): writing_clarity 63.3%, novelty_insufficient 34.2%, dataset_limitation 28.3%
   - **Layer 5 — Citation & Reference Patterns** (引用画像): SIGMOD 接收论文 median=38 refs vs 被拒论文 median=18 refs; 引用密度 3.06/1k words; Related Work 占引用 26.9%
3. **生成 Skill 文件** — 将蒸馏结果打包为可嵌入 prompt 的 markdown 格式 (`skills/sigmod_esg.md`)
4. **论文重写** — 基于蒸馏结果,对 VeritasCarbon 原始稿件进行全面修订,生成符合 SIGMOD 审稿标准的 LaTeX 文件
5. **修订报告** — 生成 593 行的详细修订报告,包含差距分析、被拒风险评估、优先修订路线图,以及重写后的 Introduction/Related Work/Method/Experiments 全文

## 2. 目录结构

```
docs/venue_distillation/
├── config.yaml                          # 蒸馏配置 (topic_id, API源, 关键词等)
├── data/
│   ├── original_paper.txt               # 原始稿件文本版
│   └── VeritasCarbon Traceable Instruction Data Generation for ESG Domain via a Council of Domain Experts_1.pdf  # 原始稿件PDF版（用户指定原文）
├── skills/
│   └── sigmod_esg.md                    # ★ 核心产出: 可嵌入prompt的SIGMOD审稿标准
├── distilled/
│   ├── distillation_aggregated.json     # 聚合的Layer 1-5统计结果 (含 aggregated_citation_patterns)
│   ├── facets.json                      # 逐篇Layer 1: 实验严谨性评分 (1845行)
│   ├── idea_dna.json                    # 逐篇Layer 2: 创新类型分析 (1253行)
│   ├── narrative_structure.json         # 逐篇Layer 3: 叙事结构分析 (1548行)
│   ├── rejected_distillation.json       # 被拒论文分析 (218行)
│   ├── citation_profiles_rejected.json  # Layer 5: 120篇被拒论文逐篇引用画像
│   └── accepted_citation_meta.json      # Layer 5: 60篇SIGMOD论文引用元数据 (OpenAlex API)
├── output/
│   ├── veritascarbon_sigmod.tex         # ★ 重写后的完整论文 LaTeX
│   ├── veritascarbon_sigmod.pdf         # 编译后的PDF
│   ├── veritascarbon_sigmod_20260407_172146.tex  # 带时间戳的备份
│   ├── revision_report_20260407_164039.md  # ★ 修订报告 (593行,含差距分析+重写)
│   ├── references.bib / references_v2.bib / references_old.bib  # 参考文献
│   ├── figures/code_architecture.png    # 架构图
│   ├── veritascarbon_pipeline.png       # 流水线图
│   └── _rewrite_checkpoint.json         # 重写断点
└── scripts/                             # 蒸馏流程的全部脚本
    ├── collect_accepted.py              # 爬取SIGMOD接收论文 (OpenAlex)
    ├── collect_accepted_v2.py           # 改进版爬取脚本
    ├── collect_rejected.py              # 爬取被拒论文 (OpenReview)
    ├── distill_accepted.py              # Layer 1-3 蒸馏
    ├── distill_rejected.py              # Layer 4 被拒论文蒸馏
    ├── extract_citation_profiles.py     # Layer 5 引用画像提取 (regex + OpenAlex API)
    ├── generate_skill.py                # 聚合 → skill.md 生成
    ├── aggregate_checkpoint.py          # 断点续跑聚合
    ├── polish_paper.py                  # 论文润色
    └── rewrite_paper.py                 # 全文重写 (518行,最核心脚本)
```

## 3. 核心文件说明

### `data/VeritasCarbon Traceable Instruction Data Generation for ESG Domain via a Council of Domain Experts_1.pdf` — 原文 PDF（用户指定）

这是本次交接中明确指定的“原文”文件，已复制到新工作区。对应的文本版原稿为 `data/original_paper.txt`。

### `output/veritascarbon_sigmod.tex` / `output/veritascarbon_sigmod.pdf` — 修改后稿件

这是基于蒸馏结果重写后的版本，供对比原文与修订稿使用。

### `skills/sigmod_esg.md` — SIGMOD 审稿标准 Skill (5 层)

这是最终产出。直接嵌入写作 prompt 中即可让 LLM 按 SIGMOD 标准写作。内容包括:

- **Layer 1 实验设计标准**: ablation 比例 18%, 真实数据使用 77%, 可扩展性测试 45%, 端到端评估 85%
- **常见数据集**: BIRD, Spider, WikiTableQuestions, TPC-H, IMDB 等
- **Layer 2 创新类型分布**: 60% 新形式化, 27% 新应用, 10% 范式转换
- **8 组 gap→hypothesis 范例**: 从真实 SIGMOD 论文提取的研究空白→假设映射
- **Layer 3 叙事结构**: 图表频率分布(performance_table: 48, architecture_diagram: 40), section 流向模板
- **Layer 4 被拒反模式**: 按频率排序的 6 大被拒原因及应对建议
- **Layer 5 引用画像**: SIGMOD 接收论文 median=38 refs (mean=43.4, IQR=26-52); 被拒论文 median=18 refs; 引用密度 3.06/1k words; 分章节引用分布; 近 3 年文献占比 39.3%

### `output/revision_report_20260407_164039.md` — 修订报告

包含:
- 整体结构评估 (与 SIGMOD 标准的差距)
- 缺失章节的识别 (Discussion, Corpus Analysis, Applications)
- 被拒风险映射 (VeritasCarbon 稿件对应的反模式匹配)
- 6 项优先修订路线图
- **完整重写的 LaTeX 章节**: Introduction, Related Work, Method/Framework, Experiments

### `distilled/*.json` — 原始蒸馏数据

可用于二次分析或更新 skill 文件。每个 JSON 包含逐篇论文的分析结果:
- `facets.json` / `idea_dna.json` / `narrative_structure.json` — Layer 1-3 (LLM 蒸馏)
- `rejected_distillation.json` — Layer 4 (被拒论文分析)
- `citation_profiles_rejected.json` — Layer 5: 120 篇被拒论文的逐篇引用画像 (regex 提取)
- `accepted_citation_meta.json` — Layer 5: 60 篇 SIGMOD 论文的引用元数据 (OpenAlex API)

## 4. 如何使用

### 4.1 将 Skill 嵌入写作 Prompt

```python
with open("docs/venue_distillation/skills/sigmod_esg.md") as f:
    skill = f.read()

prompt = f"""
{skill}

请根据以上 SIGMOD 审稿标准，改写以下论文章节:
...
"""
```

### 4.2 重新运行蒸馏 (如需更新)

```bash
conda activate AI4Sci
cd docs/venue_distillation/scripts

# Step 1: 爬取论文 (需要网络)
python collect_accepted_v2.py --config ../config.yaml
python collect_rejected.py --config ../config.yaml

# Step 2: 蒸馏
python distill_accepted.py --config ../config.yaml
python distill_rejected.py --config ../config.yaml

# Step 2.5: Layer 5 引用画像提取 (纯regex+API, 无需LLM)
python extract_citation_profiles.py

# Step 3: 生成 skill (含全5层)
python generate_skill.py

# Step 4: 重写论文
python rewrite_paper.py --config ../config.yaml --paper <原始论文路径>
```

> **注意**: 脚本依赖 `pipeline/rigor_distiller.py` 和 `pipeline/llm_caller.py`（在 AutoResearch Pipeline 中）。如需独立运行,需将相关依赖复制过来或调整 import 路径。

### 4.3 引用修订报告中的重写内容

`revision_report_20260407_164039.md` 中包含完整重写的 LaTeX 代码,可直接复制到论文中:
- Introduction (约 180 行 LaTeX)
- Related Work (约 120 行 LaTeX)
- Method/Framework (约 200 行 LaTeX)
- Experiments (约 150 行 LaTeX)

## 5. Pipeline 架构中的新增能力 — Layer 5: Citation Patterns

在本次会话中,还在 AutoResearch Pipeline 中新增了 **Layer 5: 引用与参考模式** 分析能力,修改了以下文件:

| 文件 | 修改内容 |
|------|----------|
| `pipeline/rigor_distiller.py` | 新增 `CitationProfile` dataclass, `_extract_citation_profile()`, `format_citation_patterns()` 等 |
| `scripts/distill_rejected.py` | 新增 `compute_citation_profile()`, `_compare_citation_profiles()` |
| `pipeline/orchestrator.py` | 将 `citation_patterns` 注入论文写作链 |
| `pipeline/paper_writer.py` | 在 Introduction/Related Work prompt 中添加 `{citation_patterns}` |
| `pipeline/stage_prompts.py` | 在 `S19_PAPER_REVISION` 中添加 `{citation_patterns}` |

> **注意**: Layer 5 的独立提取脚本 (`extract_citation_profiles.py`) 和生成脚本 (`generate_skill.py`) 已包含在本目录中,可独立运行。AutoResearch Pipeline 中也有对应的集成代码 (`pipeline/rigor_distiller.py`)。

## 6. 已知局限

1. **原始论文数据未复制** — `data/accepted/papers.jsonl` (2.3MB, 1335篇) 和 `data/rejected/rejected_papers.jsonl` (7.5MB, 120篇) 未复制到本目录,如需要可从源目录获取
2. **脚本依赖 AutoResearch Pipeline** — `distill_accepted.py` 和 `distill_rejected.py` import 了 `pipeline.rigor_distiller` 和 `pipeline.llm_caller`,独立运行需解决 import 路径。但 `extract_citation_profiles.py` 和 `generate_skill.py` 是**完全独立的**,无外部依赖
3. **SIGMOD 论文无全文** — SIGMOD 论文通过 OpenAlex 获取,仅有 abstract,无 full text。Layer 5 中 SIGMOD 论文的数据仅限 reference count (通过 OpenAlex API 元数据)。被拒论文 (OpenReview) 有全文,因此有完整的引用密度/分章节分布等详细分析
4. **被拒论文来自 ICLR/NeurIPS** — 不是 SIGMOD 的被拒论文 (SIGMOD 无公开被拒数据),Layer 4/5 的被拒论文分析具有通用参考价值但非 SIGMOD 特异

## 7. 建议的后续工作

1. **获取 SIGMOD 论文全文**: 通过 Semantic Scholar API 或 PDF 下载获取 SIGMOD 论文全文,以运行完整的引用画像提取 (引用密度、分章节分布等)
2. **收集更多 SIGMOD 论文**: 当前 60 篇覆盖 ESG 关键词,可扩展到全部 SIGMOD 论文以获得更鲁棒的统计
3. **论文微调**: 基于 `revision_report` 中的重写内容,结合 `SIGMOD_PAPER_OUTLINE_v2.md` 进行最终论文整合
4. **实验补充**: 修订报告建议增加 ablation study、scalability analysis 和 end-to-end evaluation,这些是 SIGMOD 的高频要求
5. **引用数量对标**: Layer 5 数据显示 SIGMOD 接收论文 median=38 refs,确保 VeritasCarbon 稿件引用数 ≥38

---

*本文档由 AutoResearch Pipeline Agent 自动生成,源自 veritascarbon_distill/ 蒸馏项目。最后更新: 2026-04-08 (补充 Layer 5 数据)*
