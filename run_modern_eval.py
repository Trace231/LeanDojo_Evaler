import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

# 1. 定义原始模型 ID
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

logging.basicConfig(level=logging.INFO)

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

def main():
    # 2. 设置训练器（仅提供评估所需的参数配置）
    trainer = SFTTrainer(
        model_name=MODEL_NAME,
        output_dir="outputs-deepseek",
        epochs_per_repo=1, 
        batch_size=8 
    )

    # 3. 实例化 Agent
    agent = FullMathlibAgent(trainer=trainer)

    # --- 核心修复：强制将 checkpoint 指向原始权重 ---
    # 这会覆盖它原本指向 "outputs-deepseek" 的默认逻辑
    agent.checkpoint = MODEL_NAME 

    print("⏳ [阶段 1] 正在检查/构建环境...")
    # 因为你之前已经提取过数据，这一步会快速通过
    agent.setup_github_repository(url=url, commit=commit)

    print("🔥 [阶段 2] 环境就绪！使用原生逻辑启动证明搜索...")
    os.makedirs("logs/search_trees", exist_ok=True)
    
    # 4. 运行证明搜索
    agent.prove()
    print("✅ 任务完成！")

if __name__ == "__main__":
    main()