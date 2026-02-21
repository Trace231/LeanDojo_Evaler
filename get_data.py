# get_data.py
import json
import os
from datasets import load_dataset

print("🚀 正在通过 HF datasets 稳定拉取数据集...")
# tasksource/leandojo 是 HF 上非常稳定的 LeanDojo 数据集镜像（防 404）
ds = load_dataset("tasksource/leandojo", split="train")

print("🔍 正在清洗数据并提取前 100 条测试用例...")
# 过滤掉那些没有具体策略 (tactic) 的空数据
valid_data = [row for row in ds if row.get("traced_tactics") and len(row["traced_tactics"]) > 0]
sampled_data = valid_data[:100]

# 创建目录并保存为我们需要的格式
os.makedirs("data/random", exist_ok=True)
output_path = "data/random/test.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"✅ 成功！已将 {len(sampled_data)} 条定理保存至 {output_path}")
