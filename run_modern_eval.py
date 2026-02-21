import os

# --- 核心修复：强制指定 Hugging Face 终点协议 ---
# 如果你在国内使用镜像站，请确保包含 https://
os.environ["HF_ENDPOINT"] = "https://huggingface.co" 
# 或者如果你使用的是镜像站：os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)

# 目标仓库信息
url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True

def main():
    # 每次运行前清理缓存，这是你之前补丁里提到的好习惯
    print("🧹 清理临时缓存...")
    os.system("rm -rf /tmp/tmp*")

    # 注意：如果 DeepSeek 模型已经在本地，model_name 可以直接填本地绝对路径
    # 这样可以绕过所有网络问题
    model_path = "deepseek-ai/DeepSeek-Prover-V2-7B"

    trainer = SFTTrainer(
        model_name=model_path,
        output_dir="outputs-deepseek",
        epochs_per_repo=1, 
        batch_size=8 
    )

    agent = FullMathlibAgent(trainer=trainer)

    print("⏳ [阶段 1] 构建环境并提取数据...")
    # 这一步现在应该能跑通了，因为 ExtractData.lean 已经修好了
    agent.setup_github_repository(url=url, commit=commit)

    print("🔥 [阶段 2] 启动推理与搜索树持久化...")
    os.makedirs("logs/search_trees", exist_ok=True)
    
    # 执行证明搜索
    agent.prove()
    print("✅ 任务完成！")

if __name__ == "__main__":
    main()