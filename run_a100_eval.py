# run_a100_eval.py
import os
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

# 指定一个轻量级的 Lean4 测试仓库
url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

# 加载你已经下好的 7B 模型
trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1, 
    batch_size=8 
)

agent = HFAgent(trainer=trainer)
agent.setup_github_repository(url=url, commit=commit)

print("🔥 正在 A100 上执行本地定理证明，并行记录搜索树...")
# 这行代码会触发模型去证明定理，并在后台生成 json 搜索树
agent.prove()
