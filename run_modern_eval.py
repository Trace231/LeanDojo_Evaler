import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

# 原始模型 ID
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

logging.basicConfig(level=logging.INFO)

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

# 继承 Agent 仅为了开启依赖构建，不重写加载逻辑
class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

def main():
    # 1. 正常的初始化
    MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"
    trainer = SFTTrainer(
        model_name=MODEL_NAME,
        output_dir="outputs-deepseek",
        # 即使不训练，这些参数也建议保留以供 Prover 参考
        epochs_per_repo=1,
        batch_size=2 
    )
    
    # 2. 实例化 Agent
    agent = FullMathlibAgent(trainer=trainer)

    # 🌟 [关键一步] 手动覆盖 checkpoint 路径
    # 这会强制 Prover 去加载原始模型，而不是去空的 outputs-deepseek 文件夹寻找
    agent.checkpoint = MODEL_NAME 

    print("⏳ [阶段 1] 正在检查/构建环境...")
    agent.setup_github_repository(url=url, commit=commit)

    print("🔥 [阶段 2] 环境就绪！启动证明搜索...")
    os.makedirs("logs/search_trees", exist_ok=True)
    
    # 3. 此时运行 prove() 就会正常加载 DeepSeek 模型
    agent.prove()

if __name__ == "__main__":
    main()