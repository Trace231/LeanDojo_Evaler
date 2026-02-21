"""
download_benchmark.py
=====================
Download the LeanDojo Lean 4 benchmark from HuggingFace and optionally
sample a smaller subset for quick testing.

Usage
-----
# Download full benchmark (~2000 test theorems)
python download_benchmark.py

# Download and sample 100 theorems for quick testing
python download_benchmark.py --sample 100

# Custom output directory
python download_benchmark.py --output_dir data/my_benchmark --sample 50
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install huggingface_hub:  pip install huggingface_hub")
    sys.exit(1)


# LeanDojo Lean 4 benchmark on HuggingFace
HF_REPO = "kaiyuy/leandojo-lean4-tactic-state-comments"

# Fallback repos (different benchmark versions)
HF_REPO_FALLBACKS = [
    "kaiyuy/leandojo-lean4-v4",
    "kaiyuy/leandojo-lean4",
]

SPLIT_DIRS = ["random", "novel_premises"]
SPLIT_FILES = ["train.json", "val.json", "test.json"]


def try_download(repo_id: str, filename: str, output_dir: Path) -> bool:
    """Attempt to download a file from a HF repo. Returns True on success."""
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        return True
    except Exception:
        return False


def download_benchmark(output_dir: Path) -> bool:
    """Download the LeanDojo benchmark. Tries multiple HF repos."""
    repos_to_try = [HF_REPO] + HF_REPO_FALLBACKS

    for repo in repos_to_try:
        print(f"Trying HuggingFace repo: {repo}")

        # Try downloading test.json from random split
        test_file = "random/test.json"
        success = try_download(repo, test_file, output_dir)

        if success:
            print(f"  Success! Downloading remaining files from {repo}...")
            for split_dir in SPLIT_DIRS:
                for split_file in SPLIT_FILES:
                    fname = f"{split_dir}/{split_file}"
                    if try_download(repo, fname, output_dir):
                        fpath = output_dir / fname
                        if fpath.exists():
                            with open(fpath) as f:
                                data = json.load(f)
                            print(f"  {fname}: {len(data)} theorems")
                    else:
                        print(f"  {fname}: not found (skipping)")
            return True
        else:
            print(f"  Failed. Trying next repo...")

    return False


def sample_dataset(input_path: Path, output_path: Path, n: int, seed: int = 42):
    """Sample n theorems that have traced_tactics from the input JSON."""
    with open(input_path) as f:
        data = json.load(f)

    # Only keep theorems that have at least 1 traced tactic
    valid = [
        t for t in data
        if t.get("traced_tactics") and len(t["traced_tactics"]) > 0
    ]

    print(f"  Source: {len(data)} total, {len(valid)} with tactics")

    random.seed(seed)
    sampled = random.sample(valid, min(n, len(valid)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sampled, f, indent=2)

    # Print summary
    total_tactics = sum(len(t["traced_tactics"]) for t in sampled)
    avg_tactics = total_tactics / len(sampled) if sampled else 0
    print(f"  Sampled: {len(sampled)} theorems, {total_tactics} total tactics "
          f"({avg_tactics:.1f} avg per theorem)")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download LeanDojo benchmark from HuggingFace."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/leandojo_benchmark",
        help="Directory to save the benchmark (default: data/leandojo_benchmark).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N theorems for quick testing (creates a *_sampled.json).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    test_path = output_dir / "random" / "test.json"
    if test_path.exists():
        print(f"Benchmark already exists at {output_dir}")
        with open(test_path) as f:
            data = json.load(f)
        print(f"  random/test.json: {len(data)} theorems")
    else:
        print(f"Downloading LeanDojo benchmark to {output_dir}...")
        ok = download_benchmark(output_dir)
        if not ok:
            print("\nAutomatic download failed. Manual alternative:")
            print("  1. Visit https://huggingface.co/datasets/kaiyuy/leandojo-lean4-tactic-state-comments")
            print("  2. Download random/test.json")
            print(f"  3. Place it at {test_path}")
            sys.exit(1)

    # --- Sample ---
    if args.sample and test_path.exists():
        sampled_path = output_dir / "random" / f"test_sampled_{args.sample}.json"
        print(f"\nSampling {args.sample} theorems...")
        sample_dataset(test_path, sampled_path, args.sample, args.seed)
        print(f"\nYou can now run:")
        print(f"  python extract_gt_logprobs.py \\")
        print(f"      --dataset_path {sampled_path} \\")
        print(f"      --ckpt_path 'deepseek-ai/DeepSeek-Prover-V2-7B' \\")
        print(f"      --output_path logs/gt_logprobs.json \\")
        print(f"      --details_path logs/gt_details.json \\")
        print(f"      --offline")


if __name__ == "__main__":
    main()
