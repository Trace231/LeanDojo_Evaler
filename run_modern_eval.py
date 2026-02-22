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
    # 1. 设置训练器（仅提供参数配置）
    trainer = SFTTrainer(
        model_name=MODEL_NAME,
        output_dir="outputs-deepseek",
        epochs_per_repo=1, 
        batch_size=8 
    )

    # 2. 实例化 Agent
    agent = FullMathlibAgent(trainer=trainer)

    # --- 核心修正：利用原生逻辑，但修正检查点路径 ---
    # 这一步手动把 Agent 误以为的“输出目录”改回“原始模型 ID”
    # 这样底层调用 from_pretrained(agent.checkpoint) 时就会指向 DeepSeek 官方权重
    agent.checkpoint = MODEL_NAME 

    print("⏳ [阶段 1] 正在检查/构建环境...")
    agent.setup_github_repository(url=url, commit=commit)

    print("🔥 [阶段 2] 环境就绪！使用原生 HFProver 逻辑启动搜索...")
    os.makedirs("logs/search_trees", exist_ok=True)
    
    # 3. 运行原生的证明搜索
    agent.prove()
    print("✅ 任务完成！")

if __name__ == "__main__":
    main()