import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)

url = "https://github.com/leanprover-community/mathlib4"
commit = "29b8c0ab4fbcf006a14ba92416801caee7c71f30"

# ！！！神之一手：重写 HFAgent 强制开启依赖编译 ！！！
class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1, 
    batch_size=8 
)

# 使用我们修改过的、强制带依赖编译的 Agent
agent = FullMathlibAgent(trainer=trainer)

print("⏳ 正在拉取官方标准 Mathlib 仓库并执行 Tracing...")
print("（这一步可能需要 5-15 分钟编译，因为这包含了整个高等数学库，请耐心等待）")

# 这次不需要传 build_deps=True 了，底层会自动获取到 True
agent.setup_github_repository(url=url, commit=commit)

print("🔥 编译成功！环境上下文已完美加载！")
print("🚀 正在使用 DeepSeek 7B 结合完整 Mathlib 进行暴力搜索...")

os.makedirs("logs/search_trees", exist_ok=True)
agent.prove()
