import os
import json
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)

# 1. 自动从你的数据集中读取正确的 URL 和 Commit
dataset_path = "data/random/test.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

url = dataset[0]["url"]
commit = dataset[0]["commit"]

print(f"🎯 自动解析到目标仓库: {url}")
print(f"🎯 自动解析到 Commit: {commit}")

# 2. 强制开启依赖编译
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

print("⏳ 正在根据数据集的要求拉取对应仓库并执行 Tracing...")
agent.setup_github_repository(url=url, commit=commit)

print("🔥 编译成功！环境上下文已完美加载！")
print("🚀 正在使用 DeepSeek 7B 进行真实定理搜索...")

os.makedirs("logs/search_trees", exist_ok=True)
agent.prove()
