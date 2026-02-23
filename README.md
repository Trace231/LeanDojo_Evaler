# LeanDojo-v2 Dynamic Analysis Playground

一个偏研究导向、偏实战的 Lean 4 自动证明分析仓库。  
核心目标不是单纯追求 pass@1，而是分析 **模型在证明搜索里的行为** ：

- 它什么时候自信但错（calibration / hallucination）
- 它什么时候在盲目试错（efficiency / blind search）
- 成功路径和失败路径在 log-prob / PPL 上到底怎么分叉

当前主线脚本：`run_dataset_eval_bfs.py`  
主分析脚本：`search_analysis_master.py`

---

## Empirical Results Overview

本仓库的核心贡献在于对自动定理证明搜索过程的深入分析。以下为典型实验的关键结果概览：

### Search Success Profile

下图展示了证明搜索过程中成功率与搜索深度的关系，观察模型在不同搜索阶段的成功概率变化：

![搜索成功曲线](analysis_output%203/aggregate_success_curve.png)

此曲线揭示了两个重要现象：（1）早期阶段的分支数量和成功率变化遵循明确的分布特征；（2）成功路径和失败路径在搜索树中呈现不对称的树深度分布。

### Distribution of Key Metrics

下图为搜索树中关键指标的分布直方图，包括节点深度、对数概率及困惑度：

![主要指标分布](analysis_output%203/distribution_histograms.png)

从分布特征可以观察：节点深度呈现长尾特征，暗示少数成功路径需要更深的搜索深度；对数概率的多峰分布反映了模型的不同决策前景。

### Misleading Index Analysis

错误指引指数（Misleading Index）用于量化模型何时给予失败分支不适当的高置信度。下图展示了误导指数的密度分布：

![误导指数密度分布](analysis_output%203/mislead_indices_density.png)

该指标对于诊断模型的 hallucination 风险至关重要：指数越高表示模型的评分与实际路径可行性之间的不一致性越严重。

---

## What Is In This Repo

- `lean_agent/prover/proof_search.py`
  - 基于 LeanDojo 的 `BestFirstSearchProver`（不改核心搜索策略）
  - 已增加动态统计采集与导出
- `run_dataset_eval_bfs.py`
  - 用 `BestFirstSearchProver` 跑 benchmark
  - 用 `TacticGenerator` 规范接模型（当前默认 DeepSeek 适配器）
- `search_analysis_master.py`
  - 汇总 `logs/search_trees/*.json` 做统计、CSV、图表
- `extract_gt_logprobs.py`
  - 计算 GT 路径的 teacher-forced logprob，用于更强对照分析

---

## Main Metrics (with formulas)

### 1) Search tree likelihood / perplexity

设第 `t` 步 tactic 的对数概率为 `lp_t`，深度为 `d`：

- 累积对数概率  
  `cum_lp(d) = sum_{t=1..d} lp_t`
- 深度归一困惑度  
  `PPL(d) = exp(- cum_lp(d) / d)`

用于对比：
- 成功路径 vs 失败路径
- 早期节点 vs 后期节点

### 2) Blind search ratio

定义一次扩展为 blind expansion，当该扩展：
- 没有产生任何子节点，或
- 产生的子节点全是错误节点（`ErrorNode`）

公式：

`blind_search_ratio = blind_expansions / total_expansions`

### 3) Mislead index (Delta form)

为了避免负对数概率下比例指标不直观，统一用差值：

- `MI_branch = max(fail_branch_cum_lp) - success_path_cum_lp`
- `MI_global_A = max(fail_node_cum_lp) - gt_cum_lp`
- `MI_global_B = fail_cum_lp_at_depth_d - avg_success_cum_lp_at_depth_d`

判据：
- `Delta > 0`：模型给错误分支更高分（hallucination risk）

---

## Quick Start (End-to-End)

### 0) Environment

```bash
conda create -n leandojo python=3.11 -y
conda activate leandojo

pip install lean-dojo-v2
pip install git+https://github.com/stanford-centaur/PyPantograph
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

建议环境变量：

```bash
export GITHUB_ACCESS_TOKEN=<your_token>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

---

### 1) Download standard benchmark

```bash
python get_normalized_data.py
```

输出（典型）：
- `data/leandojo_benchmark/random/test_official.json`
- `data/leandojo_benchmark/random/train_official.json`
- `data/leandojo_benchmark/random/validation_official.json`

---

### 2) Filter to supported toolchain set (important)

如果你看到大量 `unsupported Lean version` 或 `INIT_ERROR`，先过滤：

```bash
python - <<'PY'
import json
from lean_dojo_v2.lean_dojo.data_extraction.lean import (
    LeanGitRepo, get_lean4_version_from_config, is_supported_version
)

src = "data/leandojo_benchmark/random/test_official.json"
dst = "data/leandojo_benchmark/random/test_supported.json"

data = json.load(open(src))
cache, out = {}, []
for row in data:
    key = (row["url"], row["commit"])
    if key not in cache:
        try:
            repo = LeanGitRepo(row["url"], row["commit"])
            cfg = repo.get_config("lean-toolchain")["content"]
            ver = get_lean4_version_from_config(cfg)
            cache[key] = bool(is_supported_version(ver))
        except Exception:
            cache[key] = False
    if cache[key]:
        out.append(row)
json.dump(out, open(dst, "w"))
print("official:", len(data), "supported:", len(out), "->", dst)
PY
```

建议再做 theorem-level dedup：

```bash
python - <<'PY'
import json
src = "data/leandojo_benchmark/random/test_supported.json"
dst = "data/leandojo_benchmark/random/test_supported_dedup.json"
data = json.load(open(src))
seen, out = set(), []
for x in data:
    key = (x["url"], x["commit"], x["file_path"], x["full_name"],
           tuple(x.get("start", [])), tuple(x.get("end", [])))
    if key not in seen:
        seen.add(key)
        out.append(x)
json.dump(out, open(dst, "w"))
print("before:", len(data), "after:", len(out), "->", dst)
PY
```

---

### 3) Run BFSP evaluation

先小样本 smoke test：

```bash
python run_dataset_eval_bfs.py \
  --dataset_path data/leandojo_benchmark/random/test_supported_dedup.json \
  --model_name deepseek-ai/DeepSeek-Prover-V2-7B \
  --dtype bf16 \
  --timeout 600 \
  --max_expansions 256 \
  --num_sampled_tactics 16 \
  --max_theorems 5 \
  --analysis_event_dir logs/search_events
```

正式跑（按显存和时长调）：

```bash
python run_dataset_eval_bfs.py \
  --dataset_path data/leandojo_benchmark/random/test_supported_dedup.json \
  --model_name deepseek-ai/DeepSeek-Prover-V2-7B \
  --dtype bf16 \
  --timeout 1200 \
  --max_expansions 768 \
  --num_sampled_tactics 32 \
  --max_theorems 100 \
  --analysis_event_dir logs/search_events
```

输出：
- 树：`logs/search_trees/*.json`
- 实时事件：`logs/search_events/*.events.jsonl`
- 统计摘要：`logs/search_events/*.summary.json`

---

### 4) Run analysis

```bash
python search_analysis_master.py \
  --input_dir logs/search_trees \
  --output_dir analysis_output
```

关键输出：
- `analysis_output/basic_statistics.txt`
- `analysis_output/efficiency_statistics.txt`
- `analysis_output/mislead_indices.csv`
- `analysis_output/root_cause_report.txt`
- `analysis_output/evolution_plots/*`

---

## Interfaces You Can Extend

### Search side (fixed strategy contract)

- `BestFirstSearchProver` in `lean_agent/prover/proof_search.py`
  - 你可以加统计、事件、导出
  - 不建议改 priority / step 逻辑本体（会破坏可比性）

### Generator side (plug-in point)

`run_dataset_eval_bfs.py` 里的生成器遵循：

- `generate(state, file_path, theorem_full_name, theorem_pos, num_samples)`
- `batch_generate(...)`

所以你可以平滑替换模型，不动 BFSP。

### Analysis side

- `search_analysis_master.py` 读取标准树 JSON
- 你可以继续加字段（例如 frontier entropy、error taxonomies）

---

## Models You Can Explore (besides DeepSeek)

只要能适配到 `TacticGenerator` 接口，都能测：

- ReProver family
- Llemma / Llemma-v2
- Qwen2.5-Math / Qwen2.5-Coder (instruction-tuned variants)
- CodeLlama / StarCoder style code LMs
- GPT-family or Claude-family via API wrapper
- LeanDojo native RAG models (`RetrievalAugmentedGenerator`)

建议做两条对照线：

1. same BFSP + different generator  
2. same generator + different decoding setup

---

## Advanced Analysis & Technical Insights

### Search Efficiency Profiling

下图展示了搜索过程中节点扩展的散点分布，横轴为总扩展数，纵轴为单个定理的搜索阶段：

![节点扩展散点分析](analysis_output%203/expansion_scatter.png)

此图用于识别搜索策略的效率特征。散点的聚集程度反映了不同定理对搜索空间复杂度的差异。特别地，离群的点表示那些需要非常深的搜索树或广泛探索的定理，这类问题往往对应模型知识薄弱的领域。

### Leaf Node Complexity Analysis

叶节点的困惑度（Perplexity）和累积对数概率（cumulative log-probability）分布是诊断搜索行为的重要指标：

![叶节点累积对数概率密度](analysis_output%203/leaf_cumlp_density_aggregate.png)

![叶节点PPL密度分布](analysis_output%203/leaf_ppl_density_aggregate.png)

这两组分布揭示：
- 高困惑度的叶节点通常对应模型置信度低的探索分支
- 累积对数概率的双峰分布表明成功和失败分支间存在明确的统计可分性
- 密度的尾部特征指示了搜索中的"死胡同"现象的普遍性

### Priority Evolution Dynamics

优先级分数在搜索过程中的演变特征：

![优先级演变曲线](analysis_output%203/priority_evolution_aggregate.png)

优先级演变曲线展示了最优先级路径（orange）和中位优先级路径（blue）在搜索过程中的变化趋势。该曲线对于理解：
1. **模型的决策稳定性**：优先级曲线的平滑度反映模型评分的一致性
2. **搜索方向性**：上升趋势表示搜索算法有效地指向高置信度区域
3. **Hallucination 痕迹**：陡峭的下降可能指示模型先前的高置信度估计被后续探索推翻

### Per-Theorem Evolution Patterns

不同定理的搜索演变模式存在显著差异。`priority_evolution_per_theorem/` 目录包含了具体定理层级的优先级演变曲线，便于深入诊断特定问题的搜索困难。

---

## Common Pitfalls (and quick fixes)

- **All INIT_ERROR**
  - 通常是 Lean version / cache / import path 问题，不是模型本身
- **Cache may have been corrupted**
  - 清掉对应 commit 的 cache 目录后重跑
- **CUDA OOM**
  - 用 `--dtype bf16`
  - 降 `--num_sampled_tactics`（如 32 -> 16 -> 8）
- **Repeated theorem in logs**
  - 对 dataset 做 theorem-level dedup

---

## Project Vibe

这个项目不是只看一个 pass@1 数字。  
我们更关心：模型为什么成功、为什么失败、失败时到底是在“知识断层”还是“搜索瞎撞”。  
如果你拿这个仓库做实验，建议每次都保留：

- 同一批数据
- 同一搜索策略
- 只改一个变量（模型或解码参数）

这样结论会干净很多，也更容易写进论文。