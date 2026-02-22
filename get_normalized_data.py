import json, os
from datasets import load_dataset

# 尝试使用官方机构名下的路径
# 注意：LeanDojo 数据集很大，建议先确认 config 名称（如 "random" 或 "novel_premises"）
try:
    ds = load_dataset("lean-dojo/lean4-tactic-state-comments", split="test")
except Exception as e:
    print(f"尝试失败，正在搜索备选路径... 错误信息: {e}")
    # 如果上面的也不行，可以尝试最基础的 LeanDojo 数据集再进行过滤
    ds = load_dataset("lean-dojo/LeanDojo", "lean4", split="test")

os.makedirs("data/leandojo_benchmark/random", exist_ok=True)
path = "data/leandojo_benchmark/random/test_official.json"

with open(path, "w") as f:
    json.dump(list(ds), f)

print("saved:", path, "size:", len(ds))