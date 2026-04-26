# VeritasCarbon SIGMOD 2027 — 论文大纲与合作者对接文档 (v2)

> **最后更新**: 2026-03-18  
> **目标会议**: ACM SIGMOD 2027 — DI&DS Application Track  
> **截止日期**: Round 2 Abstract Apr 10 / Full Paper Apr 17, 2026; Round 3 Jul 10/17, 2026  
> **结论: 实验已全部完成, 不需要再做新实验**

---

## 来源标注

| 标签 | 路径 | 说明 |
|------|------|------|
| 📄 旧版 | `VeritasCarbon： Traceable Instruction Data Generation for ESG Domain via a Council of Domain Experts.pdf` | VLDB 投稿版完整论文 |
| 📊 新实验 | `VeritasCarbon_SIGMOD/results/` + `notebooks/03_SIGMOD_Experiments.ipynb` | SIGMOD 实验 (Mar 3–18 完成) |

---

## 第一部分：论文大纲 (Section-by-Section)

---

### Abstract

| 来源 | 状态 |
|------|------|
| 📄 旧版有完整 Abstract | 需更新数字 |

**需要修改的内容**:
1. 删除 PVLDB 格式引用 → SIGMOD DI&DS 格式
2. Baselines 样本量: "500" → "2,000 (与 CoDE 相同规模)"
3. 更新倍率: "3.8× ROUGE-L, 4.2× BLEU-4, 15.9× domain relevance" → **"3.9× ROUGE-L, 4.5× BLEU-4, 14.7× domain relevance"**
4. R=0 描述: 删除 "setting R=0 collapses generation entirely" → 改为 "R=2 feedback yields 24% higher ROUGE-L than no refinement"

---

### 1. Introduction (~1.5 页)

| 来源 | 状态 |
|------|------|
| 📄 旧版完整可用 | 小幅调整 |

**旧版已有**: ESG 领域 LLM 缺乏高质量指令数据 → 提出 VeritasCarbon → 4 步 pipeline → CoDE 框架 → 数据集发布 → 3 条贡献

**需要修改**:
- 贡献 (1): 追加 "样本量对等的公平比较设计"
- 贡献点中去除 "feedback is architecturally required" 表述
- 改为 SIGMOD/DI&DS 叙事: 强调 data integration pipeline + 系统可复现性

---

### 2. Related Work (~1 页)

| 来源 | 状态 |
|------|------|
| 📄 旧版完整 (3 小节) | 需补充新文献 |

**旧版已有**:
- 2.1 Instruction Data Generation (Self-Instruct, WizardLM, Alpaca, Orca)
- 2.2 Domain-Specific LLMs (BloombergGPT, FinGPT, ClimateBERT)
- 2.3 Workflow-Oriented Data Transformation & Multi-Agent Generation

**需要修改**:
- 2.1/2.3 补充 2024–2025 multi-agent 文献 (AutoGen, CrewAI, MetaGPT)
- 可新增 2.4: Data Quality & Provenance for ML Pipelines (SIGMOD 社区相关)

---

### 3. The VeritasCarbon Framework (~3 页)

| 来源 | 状态 |
|------|------|
| 📄 旧版完整 (5 小节 + 2 张图) | 微调即可 |

**旧版完整可用的子节**:
- **3.1 Problem Formulation** — 📄 直接复用 (公式化定义 C, E, D, 三约束优化)
- **3.2 Corpus Processing Pipeline** — 📄 直接复用 (4 层文档、语义分块、去重)
- **3.3 CoDE: Council of Domain Experts** — 📄 直接复用
  - 3.3.1 Layered Expert Selection (11 专家、4 层层次、K 截断)
  - 3.3.2 Multi-Expert Collaboration (Sequential / Parallel)
  - 3.3.3 MetaExpert and Feedback Refinement (R 轮反馈)
  - 3.3.4 Quality Scoring and Source Mapping (7 指标、τ=0.7)
- **3.4 Domain Knowledge Injection** — 📄 直接复用
- **3.5 Implementation** — 📄 基本复用 (Qwen2-72B 4-bit, A800 GPU)

**需要修改**: 3.5 补充 checkpoint-and-resume 策略 + 超集采样方法的一句描述

**图表**:
- Figure 1: Pipeline 示意图 — 📄 旧版 (`paper/figures/fig1_pipeline_clean.png`)
- Figure 2: CoDE 架构图 — 📄 旧版 (`paper/figures/fig2_code_arch_clean.png`)

---

### 4. Experimental Evaluation (~3.5 页) ⭐ 主要更新区域

#### 4.1 Experimental Setup

| 来源 | 状态 |
|------|------|
| 📄 旧版 + 📊 新实验覆盖 | 需重写数字 |

**旧版**: baselines 各 500, ablation 各 200, CoDE n=2000  
**📊 新版**: baselines 各 **2,000**, ablation 各 **500**, CoDE 全集 **35,009**

强调: 所有方法使用相同 Qwen2-72B 模型 + 相同 chunk 采样, baseline 与 CoDE **样本量对等**

---

#### 4.2 Main Results (Table 1) ⭐

| 来源 | 状态 |
|------|------|
| 📊 新实验 (Mar 18 eval) | ✅ 最终数据 |

**数据来源**: `results/outputs/intrinsic_comparison.json`

| Method | N | ROUGE-L | BLEU-4 | Distinct-2 | Distinct-3 | Domain Rel. | FactCheck | Struct. |
|--------|--:|--------:|-------:|-----------:|-----------:|------------:|----------:|--------:|
| **CoDE (ours)** | 35,009 | **0.3228** | **0.1893** | **0.0233** | **0.1067** | **0.3667** | **0.9462** | 0.998 |
| Direct Prompting | 2,000 | 0.0838 | 0.0424 | 0.0124 | 0.0291 | 0.0197 | 0.8702 | 1.000 |
| Self-Instruct | 2,000 | 0.0407 | 0.0098 | 0.0033 | 0.0113 | 0.0025 | 0.9336 | 1.000 |
| WizardLM-Evol | 2,000 | 0.0376 | 0.0119 | 0.0038 | 0.0102 | 0.0249 | 0.5508 | 1.000 |

**论文中汇报的倍率**:
- ROUGE-L: 0.3228 / 0.0838 = **3.9×** (vs 最强 baseline Direct Prompting)
- BLEU-4: 0.1893 / 0.0424 = **4.5×**
- Domain Relevance: 0.3667 / 0.0249 = **14.7×** (vs WizardLM-Evol)
- Distinct-3: 0.1067 / 0.0291 = **3.7×**

**图表**: fig5_coe_vs_baselines.pdf (分组柱状图) + figure3_radar.pdf (雷达图)

---

#### 4.3 Ablation Study (Table 2) ⭐

| 来源 | 状态 |
|------|------|
| 📊 新实验 (Mar 18 eval, n=500) | ✅ 最终数据 |

**数据来源**: `results/ablation/ablation_summary.json`

**(a) Expert Count (K)**

| K | Quality | ROUGE-L | BLEU-4 | Distinct-3 | Domain Rel. | FactCheck |
|--:|--------:|--------:|-------:|-----------:|------------:|----------:|
| 1 | 0.618 | 0.130 | 0.077 | 0.132 | 0.080 | 0.949 |
| 2 | 0.641 | **0.177** | 0.116 | 0.159 | 0.092 | **0.958** |
| 3 | 0.645 | 0.157 | **0.117** | **0.157** | 0.092 | 0.948 |
| 5 | **0.646** | 0.171 | 0.114 | 0.147 | **0.096** | 0.950 |

分析: K=1→K=3 quality +4.5%, 但 K=5 不再显著提升且 diversity 下降。**K=3 最佳平衡点**。

**(b) Collaboration Mode**

| Mode | Quality | ROUGE-L | BLEU-4 | Distinct-3 | Domain Rel. | FactCheck |
|------|--------:|--------:|-------:|-----------:|------------:|----------:|
| None | 0.618 | 0.132 | 0.086 | 0.129 | 0.079 | 0.948 |
| Sequential | 0.640 | 0.162 | **0.116** | **0.161** | 0.091 | **0.950** |
| Parallel | **0.647** | **0.159** | 0.115 | 0.153 | **0.091** | 0.937 |

分析: 两种协作模式均显著优于 None (+4.7% quality)。Parallel quality 最高, Sequential BLEU/diversity 最优。

**(c) Feedback Rounds (R)** ⭐ 关键变化

| R | 语义 | Quality | ROUGE-L | BLEU-4 | Distinct-3 | Domain Rel. | FactCheck |
|--:|------|--------:|--------:|-------:|-----------:|------------:|----------:|
| 0 | 1次生成, 0轮反馈 | 0.626 | 0.143 | 0.091 | 0.137 | 0.076 | **0.959** |
| 1 | 1次生成, 1轮反馈 | 0.633 | 0.152 | 0.091 | 0.131 | 0.077 | 0.958 |
| 2 | 1次生成, 2轮反馈 | **0.649** | **0.178** | **0.124** | **0.160** | **0.096** | 0.953 |

分析:
- ⚠️ **与旧版重大差异**: R=0 现在产生 **有效输出** (quality=0.626), 不再为全零
- R=0→R=2: quality +3.7%, ROUGE-L +24.2%, BLEU-4 +36.9%
- 反馈机制是 **质量放大器**, 非架构必需组件
- R=2 改善最大, 特别是 BLEU-4 和 ROUGE-L

**(d) Knowledge Injection**

| Setting | Quality | ROUGE-L | BLEU-4 | Distinct-3 | Domain Rel. | FactCheck |
|---------|--------:|--------:|-------:|-----------:|------------:|----------:|
| Off | 0.627 | 0.159 | 0.093 | 0.134 | 0.088 | 0.947 |
| On | **0.635** | **0.163** | **0.105** | **0.149** | **0.097** | 0.949 |

分析: 知识注入带来小幅稳定改善 (quality +1.2%, domain relevance +10.6%)

**图表**: fig6_ablation_results.pdf (分组柱状图)

---

#### 4.4 Dataset Analysis (Table 3)

| 来源 | 状态 |
|------|------|
| 📄 旧版 + 📊 新实验统计 | ✅ 可复用 |

| 统计项 | 值 |
|--------|-----|
| Total QA pairs | 35,009 |
| Source chunks | 20,000 |
| Source documents | 17,721 |
| Corpus layers | 4 |
| Expert types | 11 |
| Multi-expert ratio | 45.1% |
| Avg. quality score | 0.667 (σ=0.103) |
| Avg. instruction length | 106.5 chars |
| Avg. response length | 380.4 chars |

Top experts: Summary (29,343) > Consistency Verification (13,230) > Classification (5,110)  
Top topics: 环境 (13,245) > 社会 (10,243) > 风险 (8,456) > 碳排放 (3,551)

**图表**: fig1_expert_usage.pdf, fig2_quality_distribution.pdf, fig3_topic_coverage.pdf, fig4_layer_year_heatmap.pdf

---

#### 4.5 Scalability Analysis (新增章节)

| 来源 | 状态 |
|------|------|
| 📊 新实验 | ✅ 最终数据 |

**数据来源**: `results/scalability/scalability_results.json`

| Chunks | Gen. Time | Throughput | API Calls | Tokens |
|-------:|----------:|-----------:|----------:|-------:|
| 500 | 1,308s | 22.9 pairs/min | 1,500 | 400K |
| 1,000 | 2,723s | 22.0 | 3,000 | 800K |
| 2,000 | 5,088s | 23.6 | 6,000 | 1.6M |
| 5,000 | 13,050s | 23.0 | 15,000 | 4.0M |
| 10,000 | 25,247s | 23.8 | 30,000 | 8.0M |
| 20,000 | 46,052s | 26.1 | 60,000 | 16.0M |

关键发现: 吞吐量在 22–26 pairs/min 保持稳定, 呈近似线性扩展。

**图表**: scalability 相关 pdf 在 `results/scalability/` 下

**对论文的意义**: SIGMOD DI&DS 赛道重视系统可扩展性, 这是旧版没有的新增内容。

---

#### 4.6 Case Study

| 来源 | 状态 |
|------|------|
| 📄 旧版有一个案例 | 可增选新案例 |

旧版: Scope 1/Scope 2 碳排放数据分析案例, 对比 CoDE vs 3 baselines 的输出差异。可从 n=2000 新数据中补选。

---

#### 4.7 Discussion

| 来源 | 状态 |
|------|------|
| 📄 旧版有部分 | 需重写 R=0 段落 |

**📄 可复用**:
- CoDE 优势三因素分析 (专家分工 + MetaExpert 指导 + 质量门控)
- WizardLM-Evol 异常分析 (冗长输出放大幻觉)
- Source Traceability 讨论

**需要重写**:
- ~~"R=0 confirms feedback is architecturally required"~~ → "Feedback serves as a quality amplifier, progressively improving generation fidelity"

**建议新增**:
- 样本量对等设计的合理性
- Baselines 在大样本下 diversity 暴跌现象

---

### 5. Conclusion & Future Work

| 来源 | 状态 |
|------|------|
| 📄 旧版有 (~0.5 页) | 需扩展 |

扩展方向: data management pipeline 贡献总结 + downstream fine-tuning 评估展望 + curriculum learning 方向

---

### References

| 来源 | 状态 |
|------|------|
| 📄 旧版 ~25 条 | 需补充 |

需补充: 2024–2025 multi-agent / data-centric AI / SIGMOD 社区文献

---

## 第二部分：实验成果总览

### 旧版已有 (📄 VLDB Paper)

| 实验 | 规模 | 用于论文 |
|------|------|---------|
| Baselines | 3 方法 × n=500 | ❌ 已被新数据替代 |
| Ablation | 4 维度 × n=200 | ❌ 已被新数据替代 |
| CoDE 全量生成 | 20K chunks → 35,009 QA | ✅ 沿用 |
| Dataset Statistics | 语料/QA 统计 | ✅ 沿用 |
| Pipeline 架构图 (Fig 1, 2) | 2 张 | ✅ 沿用 |

### 新增成果 (📊 SIGMOD 实验)

| 实验 | 规模 | 日期 | 状态 |
|------|------|------|------|
| Baseline: Direct Prompting | n=2,000 | Mar 4 | ✅ 数据 + Eval |
| Baseline: Self-Instruct | n=2,000 | Mar 4 | ✅ 数据 + Eval |
| Baseline: WizardLM-Evol | n=2,000 | Mar 6 | ✅ 数据 + Eval |
| Ablation: Expert Count (K=1,2,3,5) | 4组 × n=500 | Mar 8 | ✅ 数据 + Eval |
| Ablation: Collaboration (None/Seq/Par) | 3组 × n=500 | Mar 9 | ✅ 数据 + Eval |
| Ablation: Feedback (R=0,1,2) | 3组 × n=500 | Mar 9–11 | ✅ 数据 + Eval (R=0 bug fixed) |
| Ablation: Knowledge (Off/On) | 2组 × n=500 | Mar 11 | ✅ 数据 + Eval |
| Scalability | 500→20K chunks | Mar 3 | ✅ 完成 |
| Dataset Statistics 更新 | 语料+QA 统计表 | Mar 11 | ✅ 完成 |
| Paper-Ready Figures | 6 张新图 + 旧图 | Mar 11 | ✅ 完成 |
| Intrinsic Evaluation (全量) | 35K CoDE + 3×2K baselines + 12×500 ablation | **Mar 18** | ✅ 完成 |

---

### 新旧实验关键对比

| 指标 | 旧版 (VLDB) | 新版 (SIGMOD) | 变化 |
|------|------------|-------------|------|
| Baseline 样本量 | 3 × 500 | 3 × **2,000** | 4× 扩大, 样本量对等 |
| Ablation 样本量 | 4维度 × 200 | 4维度 × **500** | 2.5× 扩大 |
| CoDE eval 样本 | 2,000 | **35,009** (全集) | 17.5× |
| ROUGE-L 倍率 | 3.8× | **3.9×** | 稳健 |
| BLEU-4 倍率 | 4.2× | **4.5×** | 略增 |
| Domain Rel 倍率 | 15.9× | **14.7×** | 稳健 |
| R=0 quality | 0.000 (bug) | **0.626** (fixed) | 根本性修正 |
| R=2 quality | 0.631 | **0.649** | 稳健 |

---

## 第三部分：是否还需要新实验？

### ✅ 不需要。理由如下:

1. **Baselines 已完成** : 3 个方法 × 2,000 样本, 与 CoDE 样本量对等, eval 已跑完
2. **Ablation 已完成** : 4 维度 12 组 × 500 样本, eval 已跑完 (含 R=0 bug 修复后数据)
3. **CoDE 全量评估** : 35,009 条 QA 的完整 7-metric 评估已完成
4. **Scalability** : 500→20K 扩展性测试已完成
5. **Dataset Statistics** : 语料统计、专家分布、Topics、质量分布均已完成
6. **图表生成** : 6 张新图 + 统计表格均已生成

不需要新实验的核心原因:
- 内在指标 (intrinsic evaluation) 对于 DI&DS Application Track 已经足够
- 下游 fine-tuning 评估可作为 Future Work 讨论, 不是必须
- 统计显著性测试不需要新实验, 只需对现有数据做 bootstrap (论文写作阶段处理)

---

## 第四部分：接下来的工作 (纯论文撰写)

### 🔴 高优先级 — 论文核心内容

| # | 任务 | 说明 | 输入物料 |
|---|------|------|---------|
| 1 | **更新 Table 1 (Main Results)** | 用新数据 (CoDE n=35009, baselines n=2000) 替换旧版 | `intrinsic_comparison.json` |
| 2 | **更新 Table 2 (Ablation)** | 用新数据 (n=500) 替换旧版 (n=200), R=0 不再为 0 | `ablation_summary.json` |
| 3 | **重写 Feedback 段落** | 旧版 "architecturally required" → "quality amplifier + progressive improvement" | 本文第五部分的建议表述 |
| 4 | **新写 Scalability 节 (4.5)** | 旧版没有; DI&DS 赛道建议加入 | `scalability_results.json` |
| 5 | **更新 Abstract / Introduction** | 更新数字、倍率、贡献点 | Table 1 最终数据 |

### 🟡 重要 — 论文质量

| # | 任务 | 说明 |
|---|------|------|
| 6 | **LaTeX 模板切换** | 使用 SIGMOD 2027 acmart 模板; 当前 `table2_main_comparison.tex` 和 `table3_ablation.tex` 需用新数据重新生成 |
| 7 | **Related Work 补充** | 增加 2024–2025 multi-agent/data-centric AI 文献 |
| 8 | **Discussion 更新** | 新增: 样本量对等设计讨论 + baselines diversity 随规模递减分析 |
| 9 | **Figures 整合** | 新 fig1–fig6 (Mar 11) 已在 `paper/figures/` 下; 需要: (a) fig5/fig6 用新 n=500 ablation 数据重画, (b) Radar chart 用新 Table 1 数据重画 |
| 10 | **Case Study 更新** | 可从 n=2000 新 baseline 数据中选更有代表性的例子 |

### 🟢 建议 — 增强竞争力

| # | 任务 | 说明 |
|---|------|------|
| 11 | **统计显著性** | 对 Table 1 做 bootstrap CI 或 permutation test (不需新实验, 用现有 JSONL 数据) |
| 12 | **Error Analysis** | 分析 CoDE 失败案例 (quality < 0.5) 的模式分布 |
| 13 | **格式检查** | 确认页数限制 (DI&DS 通常 12 页), 检查匿名要求 |

---

## 第五部分：R=0 Feedback 讨论 — 新旧表述对比

### ❌ 旧版 (VLDB, 已废弃)

> "When feedback is completely disabled (R = 0), the system produces entirely empty outputs: all quality metrics drop to zero, and structural completeness is 0.0. … This confirms that the MetaExpert feedback mechanism is not merely an optimization but a **basic architectural requirement** of the CoDE framework."

**废弃原因**: 这是一个代码 bug (off-by-one: `max_iterations=0` → while loop 不执行), 不是架构设计。已修复。

### ✅ 新版建议表述

**4.3 Ablation — Feedback Rounds 段落:**

> Disabling iterative refinement entirely (R = 0) still produces valid instruction–response pairs (quality 0.626, ROUGE-L 0.143), confirming that the base generation pipeline functions without MetaExpert feedback. However, each additional feedback round yields progressive improvement, with the gains accelerating at R = 2: quality rises to 0.649 (+3.7% over R = 0), ROUGE-L to 0.178 (+24.2%), and BLEU-4 to 0.124 (+36.9%). This pattern indicates that the MetaExpert feedback loop functions as a **quality amplifier** — the initial generation captures core content, while iterative refinement substantially improves source fidelity and lexical precision.

**4.7 Discussion — Feedback Analysis 段落:**

> The feedback ablation reveals a non-linear improvement curve: R = 2 outperforms R = 1 by a wider margin (+2.6% quality) than R = 1 over R = 0 (+1.0%), particularly on ROUGE-L (+16.9%) and BLEU-4 (+36.3%). This suggests the first round primarily corrects structural issues, while the second round refines content alignment. The consistently high FactCheck scores across all R values (≥ 0.95) indicate that factual grounding is established during initial generation and preserved through refinement. We adopt R = 2 as the default configuration.

---

## 第六部分：结果文件清单

### 最终数据文件 (直接用于论文)

| 文件 | 内容 | 状态 |
|------|------|------|
| `results/outputs/intrinsic_comparison.json` | Table 1 + Table 2 全部 metrics | ✅ Mar 18 最新 |
| `results/ablation/ablation_summary.json` | Ablation 4维度12组 (n=500) | ✅ Mar 18 最新 |
| `results/figures_and_tables/all_metrics.json` | 合并全部 metrics | ✅ Mar 18 最新 |
| `results/figures_and_tables/table1_corpus_statistics.json` | 语料统计 | ✅ |
| `results/figures_and_tables/table2_qa_statistics.json` | QA 数据集统计 | ✅ |
| `results/scalability/scalability_results.json` | 扩展性测试 | ✅ |

### 需要用新数据重新生成的文件

| 文件 | 问题 | 处理 |
|------|------|------|
| `results/figures_and_tables/table2_main_comparison.tex` | 旧数据: CoDE n=2000, baselines n=500 | 用 `intrinsic_comparison.json` 重新生成 |
| `results/figures_and_tables/table3_ablation.tex` | 旧数据: n=200, R=0=0 | 用 `ablation_summary.json` 重新生成 |
| `results/figures_and_tables/fig5_coe_vs_baselines.pdf` | 可能用旧数据画的 | 建议重画 |
| `results/figures_and_tables/fig6_ablation_results.pdf` | 可能用旧 n=200 数据画的 | 建议重画 |
| `results/figures_and_tables/figure3_radar.pdf` | 旧版 radar chart | 用新 Table 1 数据重画 |

### 原始数据文件

| 文件 | 内容 |
|------|------|
| `results/baselines/*_results.jsonl` | 3 × 2,000 条 baseline 原始数据 |
| `results/ablation/*/` | 4 维度 12 组 × 500 条 ablation 原始数据 |
| `data/instructions/qa_pairs_complete_v3_*.jsonl` | CoDE 全量 35,009 条 QA 数据 |
| `data/processed_corpus/chunks_sampled_20000_by_year.jsonl` | 20,000 chunks 采样数据 |
