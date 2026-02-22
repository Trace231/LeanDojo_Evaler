import os
import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

# 强制模型名称
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

logging.basicConfig(level=logging.INFO)

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

    # --- 修复后的 _setup_prover ---
    def _setup_prover(self):
        from lean_dojo_v2.prover.hf_prover import HFProver
        
        # 使用 getattr 安全获取，如果没定义则返回 None
        idx_theorems = getattr(self, "indexed_theorems", None)
        
        print(f"🚀 正在加载模型权重: {MODEL_NAME}")
        self.prover = HFProver(
            MODEL_NAME, 
            indexed_theorems=idx_theorems,
            tactic_state_fixed=True
        )

def main():
    # 之前已经跑完了 1614 文件，千万不要删缓存！
    # os.system("rm -rf /tmp/tmp*") 

    trainer = SFTTrainer(
        model_name=MODEL_NAME,
        output_dir="outputs-deepseek",
        epochs_per_repo=1, 
        batch_size=8 
    )

    agent = FullMathlibAgent(trainer=trainer)

    print("⏳ [阶段 1] 检查环境（数据已提取，将秒过）...")
    agent.setup_github_repository(url=url, commit=commit)

    print("🔥 [阶段 2] 环境就绪！启动 DeepSeek 进行证明搜索...")
    os.makedirs("logs/search_trees", exist_ok=True)
    
    # 启动推理
    agent.prove()
    print("✅ 任务完成！")

if __name__ == "__main__":
    main()