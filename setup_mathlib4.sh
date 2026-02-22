#!/usr/bin/env bash
# setup_mathlib4.sh
# =================
# Clone and build mathlib4 with a Lean version matching Pantograph.
#
# This script:
#   1. Detects the Lean version Pantograph expects
#   2. Finds a matching mathlib4 commit (or uses latest)
#   3. Clones mathlib4 to /tmp/mathlib4
#   4. Downloads cached oleans (MUCH faster than compiling from source)
#   5. Builds only if cache download fails
#
# Usage:
#   bash setup_mathlib4.sh
#
# Prerequisites:
#   - elan (Lean version manager): https://github.com/leanprover/elan
#   - lake (comes with Lean)
#   - Python with pantograph installed (to detect version)

set -euo pipefail

MATHLIB_DIR="${MATHLIB_DIR:-/tmp/mathlib4}"
MATHLIB_REPO="https://github.com/leanprover-community/mathlib4.git"

echo "============================================================"
echo "  mathlib4 Setup for Pantograph Evaluation"
echo "============================================================"
echo ""

# ── Step 1: Detect Pantograph's Lean version ──────────────────────────────
echo "[Step 1] Detecting Pantograph's Lean version..."

PANTO_LEAN_VERSION=""
# Try to get the version from pantograph's package info
if python3 -c "import pantograph" 2>/dev/null; then
    # Try to find the lean-toolchain in pantograph's installation
    PANTO_PATH=$(python3 -c "import pantograph; import os; print(os.path.dirname(pantograph.__file__))" 2>/dev/null || true)
    if [ -n "$PANTO_PATH" ]; then
        # Check for lean-toolchain in pantograph's directory
        for tc_path in "$PANTO_PATH/lean-toolchain" "$PANTO_PATH/../lean-toolchain"; do
            if [ -f "$tc_path" ]; then
                PANTO_LEAN_VERSION=$(cat "$tc_path" | tr -d '[:space:]')
                echo "  Found from pantograph lean-toolchain: $PANTO_LEAN_VERSION"
                break
            fi
        done
    fi
fi

if [ -z "$PANTO_LEAN_VERSION" ]; then
    # Fall back to checking the default lean version
    if command -v lean &>/dev/null; then
        LEAN_VERSION_OUTPUT=$(lean --version 2>/dev/null || true)
        echo "  Current Lean version: $LEAN_VERSION_OUTPUT"
        # Extract version like "4.26.0" from "Lean (version 4.26.0, ...)"
        LEAN_VER=$(echo "$LEAN_VERSION_OUTPUT" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [ -n "$LEAN_VER" ]; then
            PANTO_LEAN_VERSION="leanprover/lean4:v${LEAN_VER}"
        fi
    fi
fi

if [ -z "$PANTO_LEAN_VERSION" ]; then
    echo "  WARNING: Could not detect Pantograph's Lean version."
    echo "  Will use mathlib4's latest lean-toolchain."
    echo "  If there's a version mismatch, Pantograph will fail at runtime."
    echo ""
else
    echo "  Pantograph needs: $PANTO_LEAN_VERSION"
    echo ""
fi

# ── Step 2: Clone mathlib4 ───────────────────────────────────────────────
if [ -d "$MATHLIB_DIR/.git" ]; then
    echo "[Step 2] mathlib4 already exists at $MATHLIB_DIR"
    cd "$MATHLIB_DIR"
    git fetch origin --quiet
else
    echo "[Step 2] Cloning mathlib4..."
    git clone --depth=1 "$MATHLIB_REPO" "$MATHLIB_DIR"
    cd "$MATHLIB_DIR"
fi

# ── Step 3: Find matching commit ─────────────────────────────────────────
echo "[Step 3] Checking lean-toolchain compatibility..."

CURRENT_TC=$(cat lean-toolchain 2>/dev/null | tr -d '[:space:]')
echo "  mathlib4 latest lean-toolchain: $CURRENT_TC"

if [ -n "$PANTO_LEAN_VERSION" ] && [ "$CURRENT_TC" != "$PANTO_LEAN_VERSION" ]; then
    echo ""
    echo "  ⚠ VERSION MISMATCH DETECTED!"
    echo "    Pantograph expects: $PANTO_LEAN_VERSION"
    echo "    mathlib4 latest:    $CURRENT_TC"
    echo ""
    echo "  Options:"
    echo "    1. Override mathlib4's lean-toolchain to match Pantograph (risky)"
    echo "    2. Install matching Pantograph for $CURRENT_TC"
    echo "    3. Find an older mathlib4 commit that uses $PANTO_LEAN_VERSION"
    echo ""
    echo "  Attempting option 1: overriding lean-toolchain..."
    echo "$PANTO_LEAN_VERSION" > lean-toolchain
    echo "  Updated lean-toolchain to: $PANTO_LEAN_VERSION"
    echo ""
    echo "  NOTE: If lake build fails, you may need option 2 or 3."
    echo "  To install a matching Pantograph, run:"
    echo "    pip install pantograph  # (check for version matching $CURRENT_TC)"
    echo ""
fi

# ── Step 4: Download cached oleans ────────────────────────────────────────
echo "[Step 4] Downloading cached oleans (this avoids compiling from source)..."

# Install the correct Lean toolchain first
TOOLCHAIN=$(cat lean-toolchain | tr -d '[:space:]')
echo "  Installing Lean toolchain: $TOOLCHAIN"
elan toolchain install "$TOOLCHAIN" 2>/dev/null || true

# Try lake exe cache get (downloads pre-built .olean files)
echo "  Running: lake exe cache get"
if lake exe cache get 2>&1; then
    echo "  Cache download successful!"
else
    echo "  Cache download failed (expected if lean-toolchain was overridden)."
    echo "  Will try full build instead."
fi

# ── Step 5: Build ─────────────────────────────────────────────────────────
echo ""
echo "[Step 5] Building mathlib4..."
echo "  This may take a while if oleans were not cached."
echo "  Running: lake build"
echo ""

if lake build 2>&1; then
    echo ""
    echo "============================================================"
    echo "  SUCCESS! mathlib4 built at: $MATHLIB_DIR"
    echo ""
    echo "  Lean toolchain: $(cat lean-toolchain)"
    echo ""
    echo "  Next steps:"
    echo "    1. Ensure MATHLIB_PROJECT_PATH=\"$MATHLIB_DIR\" in run_dataset_eval.py"
    echo "    2. python download_benchmark.py --sample 50"
    echo "    3. python run_dataset_eval.py"
    echo "    4. python search_analysis_master.py --input_dir logs/search_trees"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "  BUILD FAILED"
    echo ""
    echo "  This likely means a Lean version mismatch."
    echo "  Try one of these fixes:"
    echo ""
    echo "  Fix 1: Use mathlib4's native Lean version"
    echo "    cd $MATHLIB_DIR"
    echo "    git checkout -- lean-toolchain"
    echo "    lake exe cache get && lake build"
    echo "    # Then install matching Pantograph:"
    echo "    pip install pantograph  # check for matching version"
    echo ""
    echo "  Fix 2: Use offline mode (no real verification)"
    echo "    Set MATHLIB_PROJECT_PATH = None in run_dataset_eval.py"
    echo ""
    echo "============================================================"
    exit 1
fi
