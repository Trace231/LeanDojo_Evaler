import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)

# ！！关键点！！
# 我们这次使用 LeanDojo 官方 benchmark 所在的、极其稳定的 mathlib4 仓库和对应 commit
# 这个仓库已经被官方跑通了无数遍，绝不会报 ExtractData.lean 的错
url = "https://github.com/leanprover-community/mathlib4"
commit = "29b8c0ab4fbcf006a14ba92416801caee7c71f30"

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1, 
    batch_size=8 
)

agent = HFAgent(trainer=trainer)

print("⏳ 正在拉取官方标准 Mathlib 仓库并执行 Tracing...")
print("（这一步可能需要 5-15 分钟编译，因为这包含了整个高等数学库，请耐心等待）")

# 必须带上 build_deps=True，只有这样模型才能看懂那些复杂的数学符号
agent.setup_github_repository(url=url, commit=commit, build_deps=True)

print("🔥 编译成功！环境上下文已完美加载！")
print("🚀 正在使用 DeepSeek 7B 结合完整 Mathlib 进行暴力搜索...")

os.makedirs("logs/search_trees", exist_ok=True)
agent.prove()
