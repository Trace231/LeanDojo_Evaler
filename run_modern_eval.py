import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

# 强制指定模型名称，防止它去空的 outputs 目录找
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

logging.basicConfig(level=logging.INFO)

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

    # --- 关键覆盖：强制让 Prover 加载原始模型，而不是从 trainer.output_dir 加载 ---
    def _setup_prover(self):
        from lean_dojo_v2.prover.hf_prover import HFProver
        # 这里直接传 MODEL_NAME，绕过 trainer 的输出路径
        self.prover = HFProver(
            MODEL_NAME, 
            indexed_theorems=self.indexed_theorems,
            tactic_state_fixed=True
        )

def main():
    # 既然已经提取过数据了，这里就不再清理 /tmp 了，直接利用之前的成果
    # os.system("rm -rf /tmp/tmp*") 

    trainer = SFTTrainer(
        model_name=MODEL_NAME,
        output_dir="outputs-deepseek",
        epochs_per_repo=1, 
        batch_size=8 
    )

    # 实例化我们修改后的 Agent
    agent = FullMathlibAgent(trainer=trainer)

    print("⏳ [阶段 1] 正在检查/构建环境...")
    # 因为你之前已经跑到了 1613/1614，这一步会非常快（直接命中缓存）
    agent.setup_github_repository(url=url, commit=commit)

    print("🔥 [阶段 2] 环境就绪！启动 DeepSeek 进行证明搜索...")
    os.makedirs("logs/search_trees", exist_ok=True)
    
    # 启动推理
    agent.prove()
    print("✅ 搜索完毕！")

if __name__ == "__main__":
    main()