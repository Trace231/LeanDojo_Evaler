# run_minimal_eval.py
import os
from pantograph.server import Server
from lean_dojo_v2.prover import HFProver

print("🚀 步骤 1: 初始化 HFProver (加载 DeepSeek 7B)...")
# 这里会直接使用你缓存好的模型，瞬间加载
prover = HFProver(ckpt_path="deepseek-ai/DeepSeek-Prover-V2-7B")

print("🔌 步骤 2: 启动本地 Pantograph Lean 服务器...")
server = Server()

# 确保输出目录存在，Claude 修改的代码会把 JSON 写到这里
os.makedirs("logs/search_trees", exist_ok=True)

# 我们直接给两道最纯粹的逻辑题，不依赖任何第三方数学库
test_goals = [
    "∀ {p q : Prop}, p ∧ q → q ∧ p",
    "∀ (p q : Prop), p ∨ q → q ∨ p"
]

print("🔥 步骤 3: 开始执行证明搜索 (触发底层树记录逻辑!)...")
for i, goal in enumerate(test_goals):
    print(f"\n正在搜索第 {i+1} 题: {goal}")
    try:
        # 这里就是 Best-First Search 的核心入口！
        # 只要跑了这行，你的 Claude 拦截代码就会疯狂记录节点
        result, used_tactics = prover.search(server=server, goal=goal, verbose=True)
        print(f"--> 结果: {'成功' if result else '失败'}")
    except Exception as e:
        print(f"--> 报错: {e}")

print("\n✅ 搜索完毕！")
