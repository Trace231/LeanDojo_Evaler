# LeanDojo-v2 全量搜索动态分析流水线 (A100 旗舰版)

本工具链旨在为基于 LLM 的 Lean 4 定理证明器（如 ReProver / DeepSeek-Prover-V2）提供深度的**模型校准（幻觉）**与**搜索效率**分析。通过在底层拦截并改造搜索树机制，我们将黑盒的定理证明过程完全透明化，并用严谨的数学指标量化模型在推理中的“盲目自信”与“搜索迷茫”。

---

## 🔬 第一部分：核心技术改造与数学原理

为了实现全量动态分析，我们在 LeanDojo-v2 的核心层级进行了深度代码注入，并重构了评估算法：

### 1. 搜索树的数据采集强化 (`search_tree.py` & `proof_search.py`)
原生的 LeanDojo 仅保留了搜索的最终结果。我们在 Best-First Search (BFS) 主循环中进行了拦截，并重写了序列化逻辑：
* **搜索拓扑追踪**：新增 `order_of_expansion` 字段，精准复现模型陷入死胡同时的“活跃前沿 (Active Frontier)”。
* **步长归一化困惑度 (PPL)**：由于累积对数似然 (Cumulative Log-Prob) 会随深度自然衰减，我们引入 PPL 消除深度干扰：
  $ \text{PPL} = \exp\left(-\frac{\text{cumulative\_logprob}}{\text{depth}}\right) $
* **防环序列化 (DAG Protection)**：Lean 的状态树可能合并退化为有向无环图 (DAG)。我们在 `to_dict()` 中引入 `_visited` 集合，彻底阻断了 Python 递归爆栈的风险。
* **物理报错快照**：拦截 `LeanError`，将引擎的原始报错直接挂载到失败节点上，作为诊断依据。

### 2. 标准对数似然提取 (`extract_gt_logprobs.py`)
我们需要提取人类专家编写的标准策略 (Ground Truth Tactics) 的得分，作为物理基准。本脚本解决了两大架构难题：
* **彻底的架构解耦 (Causal LM vs Seq2Seq)**：
  * 对于 T5/ByT5 (Seq2Seq)：通过 Encoder-Decoder 掩码获取序列得分。
  * 对于 DeepSeek (Causal LM)：拒绝平均 Loss 陷阱。自回归模型中 `logits[t]` 预测 `labels[t+1]`，脚本通过精准的 Token 偏移 (`shift_logits = logits[:, :-1, :]`) 和交叉熵还原 (`log_softmax → gather → sum`)，完美对齐了 Beam Search 时的严谨分数。
* **Offline 极速推断模式**：直接抽取数据集里的 `state_before` 字段进行纯文本推断，完全绕过 Lean Dojo 环境的初始化与 I/O 瓶颈。

### 3. 误导指数 (Mislead Index) 的数学重构
由于 Log-Prob 始终为负数，传统除法会产生“误差越大，指数越小”的反直觉现象。我们将其重构为**差值公式 ($\Delta$)**：
* **内部偏离指数**：$\text{MI}_{\text{branch}} = \max(\text{失败分支得分}) - \text{最终成功路径得分}$
* **全局幻觉指数 A**：$\text{MI}_{\text{global\_A}} = \max(\text{失败节点得分}) - \text{专家基准得分 (GT)}$
* **全局幻觉指数 B**：$\text{MI}_{\text{global\_B}} = \text{失败节点得分} - \text{同深度成功定理的平均得分}$

*研判标准*：当 $\Delta > 0$ 时，即发生“幻觉倒挂”（模型赋予错误路线的概率高于正确路线）。差值越大，模型偏离越严重。

---

## 🛠️ 第二部分：A100 环境配置与官方 Bug 修复

### 1. 基础依赖与深度学习环境
```bash
conda create -n leandojo python=3.11 -y
conda activate leandojo

# 安装带有 CUDA 12.4+ 加速的 PyTorch (请根据 A100 驱动调整)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# 安装 LeanDojo 栈与分布式加速组件
pip install lean-dojo-v2
pip install git+[https://github.com/stanford-centaur/PyPantograph](https://github.com/stanford-centaur/PyPantograph)
pip install deepspeed huggingface_hub
```

### 2. 修复 LeanDojo-v2 官方源码 Bug
官方包中存在相对路径导入错误，会导致 `ModuleNotFoundError: No module named 'lean_dojo_v2.lean_agent.database'`。请在终端直接运行以下命令一键修复：
```bash
sed -i 's/from \.database\.dynamic_database/from lean_dojo_v2.database.dynamic_database/g' $(python -c "import site; print(site.getsitepackages()[0])")/lean_dojo_v2/lean_agent/__init__.py
```

### 3. 配置全局环境变量
```bash
# GitHub 令牌 (拉取 Lean 仓库必备)
export GITHUB_ACCESS_TOKEN="<你的_GitHub_Token>"
# HF 镜像加速 (解决 13GB 大模型下载龟速问题，跑满 A100 万兆网卡)
export HF_ENDPOINT="[https://hf-mirror.com](https://hf-mirror.com)"
```

---

## 🚀 第三部分：端到端执行流水线

### 步骤 1：准备测试数据集 (下载与采样)
官方 Benchmark 包含 2000+ 定理。为兼顾效率，我们提供全量与采样双模式：
```bash
# 编写并运行下载脚本 download_benchmark.py
# (此脚本会从 kaiyuy/leandojo-lean4-tactic-state-comments 拉取测试集)

# 选项 A: 快速测试，随机采样 100 条定理
python download_benchmark.py --sample 100
# 产出文件: data/leandojo_benchmark/random/test_sampled_100.json

# 选项 B: 跑满全量测试集
python download_benchmark.py
# 产出文件: data/leandojo_benchmark/random/test.json
```

### 步骤 2：极速提取 Ground Truth 物理基准
在 A100 上开启 `--offline` 模式，模型将加载到显存并极速完成 Teacher Forcing 打分。
```bash
# 以采样 100 条的数据集为例
python extract_gt_logprobs.py \
    --dataset_path data/leandojo_benchmark/random/test_sampled_100.json \
    --ckpt_path "deepseek-ai/DeepSeek-Prover-V2-7B" \
    --output_path logs/gt_logprobs.json \
    --details_path logs/gt_details.json \
    --offline
```

### 步骤 3：启动本地启发式证明与数据采集
将 7B 模型完整加载到 A100 显存中，执行真正的证明搜索，并自动导出带全量拓扑信息的 JSON 树。
```python
# 编写 run_a100_eval.py
import os
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

# 此处 URL 与 Commit 需对应你数据集中定理所在的仓库
url = "[https://github.com/leanprover-community/mathlib4](https://github.com/leanprover-community/mathlib4)"
commit = "<对应数据集的_commit_hash>"

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1, 
    batch_size=8 # A100 显存充裕
)

agent = HFAgent(trainer=trainer)
agent.setup_github_repository(url=url, commit=commit)
print("🔥 A100 证明搜索启动，并行记录搜索树...")
agent.prove()
```
执行命令：`python run_a100_eval.py`

### 步骤 4：执行终极多维分析
数据融合，生成全量报告与可视化图表：
```bash
python search_analysis_master.py \
    --input_dir logs/search_trees \
    --output_dir analysis_output \
    --ground_truth logs/gt_logprobs.json
```

---

## 📊 第四部分：分析产出物深度解读 (`analysis_output/`)

分析脚本会对“成功案例”与“失败案例”，以及“原始累积量”与“平均化归一量”进行严格的分流计算。跑完后，你将获得五大核心报表：

1. **宏观基础统计 (`basic_statistics.txt`)**
   * 成功定理的平均节点展开数（衡量搜索直觉的敏锐度）。
   * 失败定理的死因归类（Timeout 超时耗尽 vs. 候选队列全军覆没）。

2. **幻觉与误导罪证表 (`mislead_indices.csv`)**
   * **核心关注 `mislead_index` 列 ($\Delta$)**：只要 $\Delta > 0$，即为确凿的模型幻觉。数值越大，说明模型给错误路线打的分数比专家正确路径还要高得多，偏离越严重。

3. **搜索演化心电图 (`evolution_plots/` 目录)**
   * **图一 (Cumulative Log-Prob)**：展示路径总得分随深度的衰减情况。
   * **图二 (PPL)**：排除深度干扰，展示模型每一步的平均“挣扎/迷茫”程度。
   * *绿线为成功路径，红线为错误分支。*

4. **骤降断崖记录 (`logprob_cliffs.csv`)**
   * 精准记录连续两步得分骤降（$\text{step\_logprob} < -3.0$）的节点位置。暴露出模型知识体系的严重断层，是构建 RL (强化学习) 负样本的绝佳素材。

5. **失败根因诊断书 (`root_cause_report.txt`)**
   * **自信幻觉型**：提取极高分（接近 0）但被 Lean 引擎物理否决的具体 Tactic。
   * **搜索迷茫型**：提取死机前最后 100 步（前沿节点），若其平均得分低于 -10 且 PPL 极高，系统会标记这些“盲目撞大运”的节点供人工复盘。