import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)

# 使用官方推荐的、版本绝对兼容的现代 Lean 4 仓库
url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

# 强行覆盖 API，开启依赖编译（解决之前的 Substring.Raw 报错）
class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1, 
    batch_size=8 
)

agent = FullMathlibAgent(trainer=trainer)

print("⏳ [阶段 1] 正在拉取现代 Lean 4 仓库并完整编译 (包含所有数学依赖)...")
# 这一步不仅会编译，还会在你的 raid/ 目录下生成一份全新的、属于这个仓库的 test.json！
agent.setup_github_repository(url=url, commit=commit)

print("🔥 [阶段 2] 环境就绪！启动 DeepSeek 7B 进行启发式搜索...")
os.makedirs("logs/search_trees", exist_ok=True)
agent.prove()
print("✅ 搜索完毕！搜索树已成功保存至 logs/search_trees/")
