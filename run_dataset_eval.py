# run_dataset_eval.py
import os
import json
from pantograph.server import Server
from lean_dojo_v2.prover import HFProver

# 1. 加载已经下好的 DeepSeek 7B
print("🚀 初始化 HFProver...")
prover = HFProver(ckpt_path="deepseek-ai/DeepSeek-Prover-V2-7B")
server = Server()
os.makedirs("logs/search_trees", exist_ok=True)

# 2. 读取你之前抽样的真实数据集 (考卷)
dataset_path = "data/random/test.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"🔥 开始在真实考卷上进行严谨的启发式搜索 (共加载 {len(dataset)} 题)...")

# 为了测试效率，我们让模型先做前 15 道真实定理
for i, item in enumerate(dataset[:15]):
    # 提取定理全名和初始证明状态
    name = item.get("full_name", f"theorem_{i}")
    
    # 获取证明第一步的初始目标 (Goal)
    if not item.get("traced_tactics"):
        continue
    goal = item["traced_tactics"][0]["state_before"]
    
    print(f"\n====================================")
    print(f"[{i+1}/15] 挑战真实定理: {name}")
    print(f"初始目标: {goal}")
    
    try:
        # 启动真实的树搜索！
        # 只要跑进去，不管模型是陷入死胡同、超时还是报错，
        # Claude 写的底层逻辑都会把这棵“挣扎的树”存进 logs/search_trees/
        result, used_tactics = prover.search(server=server, goal=goal, verbose=False)
        print(f"--> 结果: {'✅ 成功' if result else '❌ 失败 (陷入幻觉或超时)'}")
    except Exception as e:
        # 有些极度复杂的 Mathlib 符号如果没有前置 Imports 会被引擎踢出
        # 这种“直接被否决”的节点也会被我们的评估器记录为负面样本
        print(f"--> 引擎驳回/搜索中断: {e}")

print("\n🎉 真实测试跑完！现在你的 logs/search_trees/ 里拥有高质量的数据了！")
