#!/usr/bin/env python3
"""
patch_extract_data.py  —  LeanDojo ExtractData.lean 兼容性补丁
=================================================================
修复 lean_dojo_v2 的 ExtractData.lean 在新版 Lean 4 上的四个编译错误:
  1. unknown constant 'Substring.Raw'       → 替换为 Substring
  2. unknown constant 'String.Pos.Raw'      → 替换为 String.Pos
  3. invalid field 'trimAscii'              → 替换为 .trim
  4. 'sorryAx' is not a structure           → 级联错误，修复 1-3 后消除；
                                               额外添加防御性 deriving 注释

用法:
    python patch_extract_data.py [--target PATH] [--dry-run] [--no-cleanup]

默认目标:
    /opt/miniconda3/envs/leandojo/lib/python3.11/site-packages/lean_dojo_v2/lean_dojo/data_extraction/ExtractData.lean
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── 默认目标路径 ──────────────────────────────────────────────────────────
DEFAULT_TARGET = (
    "/opt/miniconda3/envs/leandojo/lib/python3.11/site-packages/"
    "lean_dojo_v2/lean_dojo/data_extraction/ExtractData.lean"
)


def parse_args():
    p = argparse.ArgumentParser(description="Patch ExtractData.lean for newer Lean 4")
    p.add_argument(
        "--target", "-t",
        default=DEFAULT_TARGET,
        help="Path to ExtractData.lean (default: conda env path)",
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Only show the diff, do not write changes",
    )
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip the /tmp/tmp* cleanup step",
    )
    return p.parse_args()


def backup(path: Path) -> Path:
    """Create a timestamped backup of the target file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(f".lean.bak_{ts}")
    shutil.copy2(path, bak)
    return bak


def apply_patches(src: str) -> str:
    """Apply all four patches and return the modified source."""
    patched = src
    applied = []

    # ── Patch 1: Substring.Raw → Substring ────────────────────────────────
    # 新版 Lean 4 移除了 Substring.Raw, Substring 本身即为顶层类型。
    # 替换类型签名中的 Substring.Raw, 保留 ToJson 实例体 (s.toString 对 Substring 同样合法)
    count = patched.count("Substring.Raw")
    if count:
        patched = patched.replace("Substring.Raw", "Substring")
        applied.append(f"[Patch 1] Substring.Raw → Substring  ({count} occurrences)")

    # ── Patch 2: String.Pos.Raw → String.Pos ──────────────────────────────
    # 旧版 Lean 4 中 String.Pos 是对 String.Pos.Raw (= Nat) 的 wrapper；
    # 新版直接使用 String.Pos, 且 .1 (即 .byteIdx) 访问仍然有效。
    count = patched.count("String.Pos.Raw")
    if count:
        patched = patched.replace("String.Pos.Raw", "String.Pos")
        applied.append(f"[Patch 2] String.Pos.Raw → String.Pos  ({count} occurrences)")

    # ── Patch 2b: 去除重复的 deriving instance ─────────────────────────────
    # 替换后会出现:
    #   instance : ToJson String.Pos where ...       (手写实例, line 13)
    #   deriving instance Lean.ToJson for String.Pos  (line 16)
    # 两者冲突。注释掉 deriving 行, 保留手写实例。
    dup_line = "deriving instance Lean.ToJson for String.Pos"
    if dup_line in patched:
        patched = patched.replace(
            dup_line,
            f"-- [patched] {dup_line}  -- removed: manual instance above suffices",
        )
        applied.append("[Patch 2b] Commented out duplicate deriving instance for String.Pos")

    # ── Patch 3: .trimAscii → .trim ──────────────────────────────────────
    # String.trimAscii 在新版中已更名为 String.trim。
    # 旧代码: s.trimAscii.toString   (trimAscii 返回 Substring, 再 .toString)
    # 新代码: s.trim                  (trim 直接返回 String, .toString 冗余但无害)
    count = patched.count(".trimAscii")
    if count:
        patched = patched.replace(".trimAscii", ".trim")
        applied.append(f"[Patch 3] .trimAscii → .trim  ({count} occurrences)")

    # ── Patch 4: sorryAx 防御 ─────────────────────────────────────────────
    # 报错 "'sorryAx' is not a structure" 是一个 **级联错误**:
    #   当 Patch 1-3 对应的类型/方法无法解析时, Lean elaborator 会插入
    #   sorryAx 占位符。在新版 Lean 4 中 sorryAx 不再是 structure, 因此
    #   这些占位符自身也会触发编译错误。
    #
    # 修复策略: 修复上游的三个类型错误后, elaborator 不再需要插入
    # sorryAx, 此错误自然消失。
    #
    # 作为额外防御, 如果源码中显式引用了 sorryAx (例如用于日志打印
    # 或模式匹配), 我们将其替换为 Lean.Expr.isSorry 检查。
    sorry_pattern = re.compile(r'\bsorryAx\b')
    sorry_matches = sorry_pattern.findall(patched)
    if sorry_matches:
        # 如果有显式的 sorryAx 结构体引用 (e.g., match on sorryAx fields),
        # 注释掉相关行, 因为新版 Lean 不再暴露该结构体。
        new_lines = []
        for line in patched.splitlines():
            if sorry_pattern.search(line):
                new_lines.append(f"-- [patched: sorryAx removed in new Lean 4] {line}")
            else:
                new_lines.append(line)
        patched = "\n".join(new_lines)
        applied.append(
            f"[Patch 4] Commented out {len(sorry_matches)} explicit sorryAx reference(s)"
        )
    else:
        applied.append(
            "[Patch 4] No explicit sorryAx references found — "
            "cascading error will be resolved by Patches 1-3"
        )

    return patched, applied


def show_diff(original: str, patched: str):
    """Print a simple line-by-line diff of changes."""
    orig_lines = original.splitlines()
    new_lines = patched.splitlines()
    max_lines = max(len(orig_lines), len(new_lines))

    print("\n" + "=" * 70)
    print("DIFF (changed lines only)")
    print("=" * 70)

    changes = 0
    for i in range(max_lines):
        old = orig_lines[i] if i < len(orig_lines) else ""
        new = new_lines[i] if i < len(new_lines) else ""
        if old != new:
            changes += 1
            print(f"  L{i+1:>4} - {old}")
            print(f"  L{i+1:>4} + {new}")
            print()

    print(f"Total changed lines: {changes}")
    print("=" * 70)


def cleanup_tmp():
    """Remove /tmp/tmp* leftover caches from crashed LeanDojo runs."""
    import glob

    targets = glob.glob("/tmp/tmp*")
    if not targets:
        print("[Cleanup] No /tmp/tmp* files found. Skipping.")
        return

    print(f"[Cleanup] Removing {len(targets)} /tmp/tmp* entries...")
    try:
        subprocess.run(
            ["rm", "-rf"] + targets,
            check=True,
            timeout=30,
        )
        print("[Cleanup] Done.")
    except subprocess.TimeoutExpired:
        print("[Cleanup] WARNING: rm -rf timed out after 30s.")
    except subprocess.CalledProcessError as e:
        print(f"[Cleanup] WARNING: rm -rf failed: {e}")


def main():
    args = parse_args()
    target = Path(args.target)

    print(f"LeanDojo ExtractData.lean Compatibility Patch")
    print(f"Target: {target}")
    print()

    # ── Read ──────────────────────────────────────────────────────────────
    if not target.exists():
        print(f"ERROR: Target file not found: {target}", file=sys.stderr)
        print(
            "  Hint: use --target to specify the correct path, e.g.:",
            file=sys.stderr,
        )
        print(f"    python {sys.argv[0]} --target ./ExtractData.lean", file=sys.stderr)
        sys.exit(1)

    original = target.read_text(encoding="utf-8")
    print(f"Read {len(original)} bytes ({original.count(chr(10))} lines)")

    # ── Patch ─────────────────────────────────────────────────────────────
    patched, applied = apply_patches(original)

    for msg in applied:
        print(f"  {msg}")

    if original == patched:
        print("\nNo changes needed — file is already patched.")
        if not args.no_cleanup:
            cleanup_tmp()
        return

    # ── Diff ──────────────────────────────────────────────────────────────
    show_diff(original, patched)

    if args.dry_run:
        print("\n[Dry run] No changes written.")
        return

    # ── Backup & Write ────────────────────────────────────────────────────
    bak = backup(target)
    print(f"\nBackup saved: {bak}")

    target.write_text(patched, encoding="utf-8")
    print(f"Patched file written: {target}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if not args.no_cleanup:
        cleanup_tmp()

    print("\nAll done. You can now re-run your LeanDojo pipeline.")


if __name__ == "__main__":
    main()
