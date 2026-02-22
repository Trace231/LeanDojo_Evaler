import json
import os
from datasets import load_dataset

# --- 配置区 ---
# 使用 cat-searcher 的版本，这是目前 Lean 4 比较稳定的 Benchmark 镜像
DATASET_ID = "cat-searcher/leandojo-benchmark-4-random"
SAVE_DIR = "data/leandojo_benchmark/random"
# 定义你想下载的数据切分
SPLITS = ["test", "train", "validation"] 

def download_and_save():
    # 1. 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建目录: {SAVE_DIR}")

    for split in SPLITS:
        print(f"正在下载 {split} 部分...")
        try:
            # 2. 加载数据集
            # 如果你在国内，运行脚本前请先执行 export HF_ENDPOINT=https://hf-mirror.com
            ds = load_dataset(DATASET_ID, split=split)
            
            # 3. 构造保存路径
            file_path = os.path.join(SAVE_DIR, f"{split}_official.json")
            
            # 4. 写入 JSON
            with open(file_path, "w", encoding="utf-8") as f:
                # 转换为列表并保存
                json.dump(list(ds), f, ensure_ascii=False, indent=2)
            
            print(f"✅ {split} 下载并保存成功！路径: {file_path}, 条数: {len(ds)}")
            
        except Exception as e:
            print(f"❌ 下载 {split} 失败: {e}")

if __name__ == "__main__":
    download_and_save()