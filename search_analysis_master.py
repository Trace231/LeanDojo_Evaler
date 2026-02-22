"""
search_analysis_master.py
=========================
Full-spectrum offline analysis of LeanDojo-v2 proof search trees.

Reads JSON files exported by the instrumented BestFirstSearchProver,
flattens the recursive tree into a DataFrame, and produces:
  1. Basic statistics (expansion counts, failure classification)
  2. Cumulative log-prob & PPL evolution plots
  3. Mislead Index computation (branch / global_A / global_B)
  4. Root-cause classification report with histograms

Usage:
    python search_analysis_master.py --input_dir logs/search_trees --output_dir analysis_output
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Data Parsing
# ---------------------------------------------------------------------------


def _flatten_tree(
    node: dict,
    theorem_name: str,
    parent_id: Optional[int],
    path_from_root: List[int],
    records: List[dict],
) -> int:
    """Recursively flatten a search-tree dict into a list of flat records.

    Returns the node_id assigned to *this* node so that callers can
    reference it as a parent.
    """
    node_id = len(records)
    records.append(
        {
            "theorem_name": theorem_name,
            "node_id": node_id,
            "parent_id": parent_id,
            "node_type": node["node_type"],
            "state_text": node.get("state_text"),
            "tactic": node.get("tactic"),
            "step_logprob": node.get("step_logprob", 0.0),
            "cumulative_logprob": node.get("cumulative_logprob", 0.0),
            "depth": node.get("depth", 0),
            "ppl": node.get("ppl"),
            "order_of_expansion": node.get("order_of_expansion", -1),
            "error_message": node.get("error_message"),
            "path_from_root": list(path_from_root),
        }
    )

    children = node.get("children", [])
    for child in children:
        if isinstance(child, str):
            # "<circular_ref>" sentinel — skip
            continue
        _flatten_tree(
            child,
            theorem_name=theorem_name,
            parent_id=node_id,
            path_from_root=path_from_root + [node_id],
            records=records,
        )

    return node_id


def load_all_trees(input_dir: str) -> Tuple[pd.DataFrame, List[dict]]:
    """Load every ``*.json`` file under *input_dir* and return
    (flattened_df, raw_meta_list).
    """
    records: List[dict] = []
    meta_list: List[dict] = []
    json_dir = Path(input_dir)

    if not json_dir.exists():
        print(f"[ERROR] Input directory does not exist: {json_dir}")
        sys.exit(1)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {json_dir}")
        return pd.DataFrame(), []

    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)
        meta_list.append(
            {
                "theorem_name": data["theorem_name"],
                "status": data["status"],
                "proof": data.get("proof"),
                "total_time": data.get("total_time"),
                "actor_time": data.get("actor_time"),
                "environment_time": data.get("environment_time"),
                "num_total_nodes": data.get("num_total_nodes"),
                "num_searched_nodes": data.get("num_searched_nodes"),
                "analysis": data.get("analysis", {}),
                "file": str(fpath),
            }
        )
        tree = data.get("search_tree")
        if tree:
            _flatten_tree(
                tree,
                theorem_name=data["theorem_name"],
                parent_id=None,
                path_from_root=[],
                records=records,
            )

    df = pd.DataFrame(records)
    print(f"Loaded {len(json_files)} theorems, {len(df)} total nodes.")
    return df, meta_list


# ---------------------------------------------------------------------------
# 2. Mark success paths
# ---------------------------------------------------------------------------


def mark_success_paths(df: pd.DataFrame, meta_list: List[dict]) -> pd.DataFrame:
    """Add a boolean column ``on_success_path`` indicating whether a node
    lies on the shortest proof path (root -> ProofFinishedNode)."""
    df["on_success_path"] = False

    for meta in meta_list:
        if meta["status"] != "Proved":
            continue
        thm = meta["theorem_name"]
        sub = df[df["theorem_name"] == thm]

        # Find all ProofFinishedNode leaves
        pf_nodes = sub[sub["node_type"] == "ProofFinishedNode"]
        if pf_nodes.empty:
            continue

        # Take the one with smallest depth (shortest proof)
        best = pf_nodes.loc[pf_nodes["depth"].idxmin()]
        # Walk the path_from_root back to root
        path_ids = best["path_from_root"]
        if isinstance(path_ids, str):
            path_ids = json.loads(path_ids)
        success_ids = set(path_ids) | {best["node_id"]}

        mask = (df["theorem_name"] == thm) & (df["node_id"].isin(success_ids))
        df.loc[mask, "on_success_path"] = True

    return df


# ---------------------------------------------------------------------------
# 3. Basic statistics
# ---------------------------------------------------------------------------


def basic_statistics(df: pd.DataFrame, meta_list: List[dict], out_dir: Path) -> None:
    meta_df = pd.DataFrame(meta_list)
    proved = meta_df[meta_df["status"] == "Proved"]
    failed = meta_df[meta_df["status"] != "Proved"]

    report_lines = ["=" * 60, "BASIC STATISTICS", "=" * 60]
    report_lines.append(f"Total theorems        : {len(meta_df)}")
    report_lines.append(f"  Proved              : {len(proved)}")
    report_lines.append(f"  Failed / Open       : {len(failed)}")

    if not proved.empty:
        avg_expanded = proved["num_searched_nodes"].mean()
        report_lines.append(
            f"Avg expansions (proved): {avg_expanded:.1f}"
        )
    if not failed.empty:
        avg_expanded_fail = failed["num_searched_nodes"].mean()
        report_lines.append(
            f"Avg expansions (failed): {avg_expanded_fail:.1f}"
        )

    # ----- Failure classification -----
    report_lines.append("")
    report_lines.append("--- Failure Classification ---")

    if not df.empty and not failed.empty:
        failed_thms = set(failed["theorem_name"])
        fail_sub = df[df["theorem_name"].isin(failed_thms)]

        timeout_thms = set()
        all_error_thms = set()

        for thm in failed_thms:
            thm_nodes = fail_sub[fail_sub["theorem_name"] == thm]
            error_nodes = thm_nodes[thm_nodes["node_type"] == "ErrorNode"]
            if error_nodes.empty:
                continue
            has_timeout = error_nodes["error_message"].str.contains(
                "Timeout", case=False, na=False
            ).any()
            if has_timeout:
                timeout_thms.add(thm)
            # Check if ALL explored children are errors (queue exhaustion)
            internal_nodes = thm_nodes[thm_nodes["node_type"] == "InternalNode"]
            explored = internal_nodes[internal_nodes["order_of_expansion"] > 0]
            if not explored.empty:
                all_error_thms.add(thm)

        n_fail = len(failed_thms)
        n_timeout = len(timeout_thms)
        n_exhausted = len(all_error_thms - timeout_thms)
        n_other = n_fail - n_timeout - n_exhausted

        report_lines.append(f"  Timeout             : {n_timeout}  ({100*n_timeout/max(n_fail,1):.1f}%)")
        report_lines.append(f"  Queue exhausted     : {n_exhausted}  ({100*n_exhausted/max(n_fail,1):.1f}%)")
        report_lines.append(f"  Other / resource cap: {n_other}  ({100*n_other/max(n_fail,1):.1f}%)")

    report = "\n".join(report_lines)
    print(report)
    (out_dir / "basic_statistics.txt").write_text(report)


# ---------------------------------------------------------------------------
# 3.5 Search efficiency statistics (from runtime analysis block)
# ---------------------------------------------------------------------------


def efficiency_statistics(meta_list: List[dict], out_dir: Path) -> None:
    rows = []
    for m in meta_list:
        analysis = m.get("analysis", {}) or {}
        if not analysis:
            continue
        rows.append(
            {
                "theorem_name": m["theorem_name"],
                "status": m["status"],
                "blind_search_ratio": analysis.get("blind_search_ratio"),
                "blind_expansions": analysis.get("blind_expansions"),
                "generated_calls": analysis.get("generated_calls"),
                "total_suggestions": analysis.get("total_suggestions"),
                "total_executed_edges": analysis.get("total_executed_edges"),
                "avg_suggestions_per_expansion": analysis.get(
                    "avg_suggestions_per_expansion"
                ),
                "max_depth_reached": analysis.get("max_depth_reached"),
                "queue_peak_size": analysis.get("queue_peak_size"),
                "avg_step_logprob_all": analysis.get("avg_step_logprob_all"),
                "avg_step_logprob_success": analysis.get("avg_step_logprob_success"),
                "avg_step_logprob_error": analysis.get("avg_step_logprob_error"),
            }
        )

    if not rows:
        msg = "No runtime analysis block found in input JSONs; skipping efficiency report."
        print(msg)
        (out_dir / "efficiency_statistics.txt").write_text(msg)
        return

    eff = pd.DataFrame(rows)
    eff.to_csv(out_dir / "efficiency_per_theorem.csv", index=False)

    proved = eff[eff["status"] == "Proved"]
    failed = eff[eff["status"] != "Proved"]

    def _mean_str(frame: pd.DataFrame, col: str) -> str:
        if frame.empty or col not in frame:
            return "N/A"
        val = frame[col].dropna()
        return f"{val.mean():.4f}" if not val.empty else "N/A"

    lines = ["=" * 60, "SEARCH EFFICIENCY STATISTICS", "=" * 60]
    lines.append(f"Theorems with analysis block: {len(eff)}")
    lines.append(f"  Proved: {len(proved)}")
    lines.append(f"  Failed/Open: {len(failed)}")
    lines.append("")
    lines.append("--- Blind Search Ratio ---")
    lines.append(f"  Mean (all):    {_mean_str(eff, 'blind_search_ratio')}")
    lines.append(f"  Mean (proved): {_mean_str(proved, 'blind_search_ratio')}")
    lines.append(f"  Mean (failed): {_mean_str(failed, 'blind_search_ratio')}")
    lines.append("")
    lines.append("--- Search Width/Depth ---")
    lines.append(f"  Peak queue size (all): {_mean_str(eff, 'queue_peak_size')}")
    lines.append(f"  Max depth (all):       {_mean_str(eff, 'max_depth_reached')}")
    lines.append(f"  Executed edges (all):  {_mean_str(eff, 'total_executed_edges')}")
    lines.append(
        "  Avg suggestions / expansion (all): "
        f"{_mean_str(eff, 'avg_suggestions_per_expansion')}"
    )
    lines.append("")
    lines.append("--- Calibration Proxy (step logprob) ---")
    lines.append(f"  Avg step logprob (all):     {_mean_str(eff, 'avg_step_logprob_all')}")
    lines.append(
        f"  Avg step logprob (success): {_mean_str(eff, 'avg_step_logprob_success')}"
    )
    lines.append(
        f"  Avg step logprob (error):   {_mean_str(eff, 'avg_step_logprob_error')}"
    )

    report = "\n".join(lines)
    print(report)
    (out_dir / "efficiency_statistics.txt").write_text(report)


# ---------------------------------------------------------------------------
# 4. Cumulative Log-Prob & PPL Evolution Plots
# ---------------------------------------------------------------------------


def _collect_paths(df: pd.DataFrame, theorem_name: str) -> List[pd.DataFrame]:
    """Return a list of DataFrames, one per root-to-leaf path."""
    sub = df[df["theorem_name"] == theorem_name].copy()
    leaves = sub[
        (sub["node_type"].isin(["ProofFinishedNode", "ErrorNode"]))
        | (sub["order_of_expansion"] == -1)  # unexplored InternalNode
    ]

    paths = []
    for _, leaf in leaves.iterrows():
        path_ids = leaf["path_from_root"]
        if isinstance(path_ids, str):
            path_ids = json.loads(path_ids)
        all_ids = list(path_ids) + [leaf["node_id"]]
        path_df = sub[sub["node_id"].isin(all_ids)].sort_values("depth")
        if not path_df.empty:
            paths.append(path_df)
    return paths


def plot_evolution(df: pd.DataFrame, meta_list: List[dict], out_dir: Path) -> None:
    """Plot cumulative log-prob and PPL vs depth for each theorem."""
    plot_dir = out_dir / "evolution_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cliff_records = []

    for meta in meta_list:
        thm = meta["theorem_name"]
        paths = _collect_paths(df, thm)
        if not paths:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{thm}  [{meta['status']}]", fontsize=10)

        for path_df in paths:
            is_success = path_df["on_success_path"].any()
            color = "green" if is_success else "red"
            alpha = 1.0 if is_success else 0.25
            lw = 2.0 if is_success else 0.8

            depths = path_df["depth"].values
            cum_lps = path_df["cumulative_logprob"].values
            ppls = path_df["ppl"].values

            ax1.plot(depths, cum_lps, color=color, alpha=alpha, linewidth=lw)
            valid = ~pd.isna(ppls)
            if valid.any():
                ax2.plot(
                    depths[valid],
                    np.array(ppls[valid], dtype=float),
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                )

            # Cliff detection: consecutive step_logprob drop > 3.0
            step_lps = path_df["step_logprob"].values
            for i in range(1, len(step_lps)):
                if step_lps[i] < -3.0 and (i < 2 or step_lps[i - 1] < -3.0):
                    cliff_records.append(
                        {
                            "theorem_name": thm,
                            "depth": int(depths[i]),
                            "step_logprob": float(step_lps[i]),
                            "cumulative_logprob": float(cum_lps[i]),
                            "tactic": path_df.iloc[i]["tactic"],
                            "on_success_path": bool(is_success),
                        }
                    )

        ax1.set_xlabel("Depth")
        ax1.set_ylabel("Cumulative Log-Prob")
        ax1.set_title("Cumulative Log-Prob vs Depth")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Depth")
        ax2.set_ylabel("PPL")
        ax2.set_title("Perplexity vs Depth")
        ax2.grid(True, alpha=0.3)

        safe_name = thm.replace("/", "_").replace("\\", "_").replace(".", "_")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=150)
        plt.close(fig)

    # Cliff report
    if cliff_records:
        cliff_df = pd.DataFrame(cliff_records)
        cliff_df.to_csv(out_dir / "logprob_cliffs.csv", index=False)
        print(f"Detected {len(cliff_df)} log-prob cliff events → logprob_cliffs.csv")

    # Aggregate evolution plot (all theorems)
    _plot_aggregate_evolution(df, meta_list, out_dir)


def _plot_aggregate_evolution(
    df: pd.DataFrame, meta_list: List[dict], out_dir: Path
) -> None:
    """Aggregate depth-vs-cumulative-logprob across all proved theorems to build
    the 'average success curve' (used later for Mislead Index global_B)."""
    proved_thms = {m["theorem_name"] for m in meta_list if m["status"] == "Proved"}
    success_nodes = df[
        (df["theorem_name"].isin(proved_thms)) & (df["on_success_path"])
    ]

    if success_nodes.empty:
        return

    grouped = success_nodes.groupby("depth")["cumulative_logprob"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped.index, grouped.values, "g-o", markersize=3, label="Avg success path")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean Cumulative Log-Prob")
    ax.set_title("Average Success-Path Log-Prob by Depth (all proved theorems)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_success_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Mislead Index
# ---------------------------------------------------------------------------


def load_ground_truth(gt_path: Optional[str]) -> Dict[str, float]:
    """Load an external ground-truth cumulative-logprob dictionary.

    Expected format — JSON dict mapping theorem_name -> cumulative_logprob of
    the teacher-forced correct proof path.  If unavailable, returns empty dict.

    To populate this file, run teacher-forcing scoring on expert tactic
    sequences and record the sum of per-step log-probs.
    """
    if gt_path is None or not Path(gt_path).exists():
        return {}
    with open(gt_path) as f:
        return json.load(f)


def compute_mislead_indices(
    df: pd.DataFrame,
    meta_list: List[dict],
    gt: Dict[str, float],
    out_dir: Path,
) -> None:
    report_lines = ["=" * 60, "MISLEAD INDEX REPORT", "=" * 60]
    records = []

    proved_thms = {m["theorem_name"] for m in meta_list if m["status"] == "Proved"}
    failed_thms = {m["theorem_name"] for m in meta_list if m["status"] != "Proved"}

    # ----- Per-theorem success-path avg logprob by depth (for global_B) -----
    success_nodes = df[
        (df["theorem_name"].isin(proved_thms)) & (df["on_success_path"])
    ]
    avg_success_by_depth: Dict[int, float] = {}
    if not success_nodes.empty:
        avg_success_by_depth = (
            success_nodes.groupby("depth")["cumulative_logprob"].mean().to_dict()
        )

    # ===== Mislead Index_branch (for proved theorems) =====
    report_lines.append("\n--- Mislead Index (branch) — proved theorems ---")
    report_lines.append("  Formula: Delta = max_fail_cum_lp - success_cum_lp")
    report_lines.append("  Delta > 0  =>  model scored a WRONG branch higher than the proof (hallucination)")
    report_lines.append("  Delta ~ 0  =>  fail branches scored similarly to the proof")
    report_lines.append("  Delta < 0  =>  model correctly preferred the proof path")
    for thm in sorted(proved_thms):
        sub = df[df["theorem_name"] == thm]
        success_path = sub[sub["on_success_path"]]
        fail_branches = sub[(~sub["on_success_path"]) & (sub["node_type"] != "InternalNode")]

        if success_path.empty or fail_branches.empty:
            continue

        success_cum_lp = success_path["cumulative_logprob"].iloc[-1]
        max_fail_cum_lp = fail_branches["cumulative_logprob"].max()

        mi_branch = max_fail_cum_lp - success_cum_lp

        # Find worst offender at same depth
        worst_row = fail_branches.loc[fail_branches["cumulative_logprob"].idxmax()]

        records.append(
            {
                "theorem_name": thm,
                "index_type": "branch",
                "mislead_index": mi_branch,
                "worst_fail_depth": int(worst_row["depth"]),
                "worst_fail_tactic": worst_row.get("tactic"),
                "worst_fail_cum_lp": float(max_fail_cum_lp),
                "success_cum_lp": float(success_cum_lp),
            }
        )
        report_lines.append(
            f"  {thm}: MI_branch={mi_branch:+.3f}  "
            f"(fail_max={max_fail_cum_lp:.3f}, success={success_cum_lp:.3f})"
        )

    # ===== Mislead Index_global_A (for failed theorems, using GT) =====
    report_lines.append("\n--- Mislead Index (global_A) — failed theorems vs Ground Truth ---")
    report_lines.append("  Formula: Delta = max_fail_cum_lp - gt_cum_lp")
    report_lines.append("  Delta > 0  =>  model scored wrong paths higher than the GT proof")
    if not gt:
        report_lines.append("  [No ground-truth file provided; skipping global_A.]")
    else:
        for thm in sorted(failed_thms):
            if thm not in gt:
                continue
            gt_cum_lp = gt[thm]
            sub = df[(df["theorem_name"] == thm)]
            if sub.empty:
                continue
            max_fail_cum_lp = sub["cumulative_logprob"].max()

            mi_global_a = max_fail_cum_lp - gt_cum_lp

            records.append(
                {
                    "theorem_name": thm,
                    "index_type": "global_A",
                    "mislead_index": mi_global_a,
                    "worst_fail_depth": -1,
                    "worst_fail_tactic": None,
                    "worst_fail_cum_lp": float(max_fail_cum_lp),
                    "success_cum_lp": float(gt_cum_lp),
                }
            )
            report_lines.append(
                f"  {thm}: MI_global_A={mi_global_a:+.3f}  "
                f"(fail_max={max_fail_cum_lp:.3f}, GT={gt_cum_lp:.3f})"
            )

    # ===== Mislead Index_global_B (for failed theorems, using avg success curve) =====
    report_lines.append(
        "\n--- Mislead Index (global_B) — failed theorems vs avg success curve ---"
    )
    report_lines.append("  Formula: Delta = fail_cum_lp_at_depth_d - avg_success_cum_lp_at_depth_d")
    report_lines.append("  Delta > 0  =>  model scored wrong path higher than avg proved path at same depth")
    if not avg_success_by_depth:
        report_lines.append("  [No proved theorems available to build baseline; skipping.]")
    else:
        for thm in sorted(failed_thms):
            sub = df[(df["theorem_name"] == thm) & (df["depth"] > 0)]
            if sub.empty:
                continue

            mi_vals = []
            for _, row in sub.iterrows():
                d = int(row["depth"])
                if d in avg_success_by_depth:
                    mi_vals.append(
                        row["cumulative_logprob"] - avg_success_by_depth[d]
                    )

            if mi_vals:
                mi_global_b = max(mi_vals)
                records.append(
                    {
                        "theorem_name": thm,
                        "index_type": "global_B",
                        "mislead_index": mi_global_b,
                        "worst_fail_depth": -1,
                        "worst_fail_tactic": None,
                        "worst_fail_cum_lp": float(sub["cumulative_logprob"].max()),
                        "success_cum_lp": None,
                    }
                )
                report_lines.append(f"  {thm}: MI_global_B={mi_global_b:+.3f}")

    report = "\n".join(report_lines)
    print(report)
    (out_dir / "mislead_index_report.txt").write_text(report)

    if records:
        mi_df = pd.DataFrame(records)
        mi_df.to_csv(out_dir / "mislead_indices.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Root-Cause Classification Report & Histograms
# ---------------------------------------------------------------------------


def root_cause_analysis(
    df: pd.DataFrame, meta_list: List[dict], out_dir: Path
) -> None:
    report_lines = ["=" * 60, "ROOT-CAUSE CLASSIFICATION", "=" * 60]

    # ---- 6a. Distribution histograms (all nodes) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cum_lps = df["cumulative_logprob"].dropna()
    if not cum_lps.empty:
        ax1.hist(cum_lps, bins=60, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Cumulative Log-Prob")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Cumulative Log-Prob (all nodes)")

    ppls = df["ppl"].dropna()
    if not ppls.empty:
        # Clip extreme PPL for visualization
        ppls_clipped = ppls.clip(upper=ppls.quantile(0.99))
        ax2.hist(ppls_clipped, bins=60, color="salmon", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("PPL")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of PPL (all nodes, clipped at 99th pct)")

    fig.tight_layout()
    fig.savefig(out_dir / "distribution_histograms.png", dpi=150)
    plt.close(fig)

    # ---- 6b. Confident-hallucination errors ----
    report_lines.append("\n--- Confident-Hallucination Errors ---")
    report_lines.append("(ErrorNode with cumulative_logprob > -1.0, i.e. near 0)")

    error_nodes = df[df["node_type"] == "ErrorNode"].copy()
    if not error_nodes.empty:
        confident_errors = error_nodes[error_nodes["cumulative_logprob"] > -1.0]
        confident_errors = confident_errors.sort_values(
            "cumulative_logprob", ascending=False
        )

        report_lines.append(f"  Found {len(confident_errors)} confident-hallucination errors")
        for _, row in confident_errors.head(20).iterrows():
            report_lines.append(
                f"  [{row['theorem_name']}] depth={row['depth']} "
                f"cum_lp={row['cumulative_logprob']:.3f} "
                f"tactic=\"{row['tactic']}\" "
                f"error=\"{row['error_message']}\""
            )
    else:
        report_lines.append("  No error nodes found.")

    # ---- 6c. Search-confusion diagnosis ----
    report_lines.append("\n--- Search-Confusion Diagnosis ---")
    report_lines.append(
        "(Last 100 expanded nodes with avg cumulative_logprob < -10.0)"
    )

    failed_thms = {m["theorem_name"] for m in meta_list if m["status"] != "Proved"}
    confusion_thms = []

    for thm in sorted(failed_thms):
        sub = df[
            (df["theorem_name"] == thm)
            & (df["node_type"] == "InternalNode")
            & (df["order_of_expansion"] > 0)
        ].sort_values("order_of_expansion", ascending=False)

        frontier = sub.head(100)
        if frontier.empty:
            continue

        avg_lp = frontier["cumulative_logprob"].mean()
        avg_ppl = frontier["ppl"].dropna().mean() if not frontier["ppl"].dropna().empty else None

        if avg_lp < -10.0:
            confusion_thms.append(thm)
            ppl_str = f"{avg_ppl:.1f}" if avg_ppl is not None else "N/A"
            report_lines.append(
                f"  {thm}: frontier_avg_cum_lp={avg_lp:.3f}, frontier_avg_ppl={ppl_str}"
            )

    report_lines.append(
        f"\n  Total search-confusion theorems: {len(confusion_thms)} / {len(failed_thms)}"
    )

    # ---- 6d. Frontier scatter plot ----
    if not df.empty:
        explored = df[
            (df["node_type"] == "InternalNode") & (df["order_of_expansion"] > 0)
        ]
        if not explored.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            proved_set = {m["theorem_name"] for m in meta_list if m["status"] == "Proved"}
            for thm, grp in explored.groupby("theorem_name"):
                color = "green" if thm in proved_set else "red"
                ax.scatter(
                    grp["order_of_expansion"],
                    grp["cumulative_logprob"],
                    c=color,
                    alpha=0.15,
                    s=8,
                    edgecolors="none",
                )
            ax.set_xlabel("Expansion Order")
            ax.set_ylabel("Cumulative Log-Prob")
            ax.set_title("Expansion Order vs Cumulative Log-Prob")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "expansion_scatter.png", dpi=150)
            plt.close(fig)

    report = "\n".join(report_lines)
    print(report)
    (out_dir / "root_cause_report.txt").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Offline analysis of LeanDojo-v2 search trees."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="logs/search_trees",
        help="Directory containing exported search-tree JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_output",
        help="Directory for analysis outputs (plots, CSVs, reports).",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Optional JSON file mapping theorem_name -> GT cumulative logprob "
        "(for Mislead Index global_A).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    df, meta_list = load_all_trees(args.input_dir)
    if df.empty:
        print("No data to analyse. Exiting.")
        return

    # --- Mark success paths ---
    df = mark_success_paths(df, meta_list)

    # --- Save flattened data ---
    df.to_csv(out_dir / "all_nodes.csv", index=False)
    print(f"Flattened node table saved to {out_dir / 'all_nodes.csv'}")

    # --- Run analyses ---
    basic_statistics(df, meta_list, out_dir)
    efficiency_statistics(meta_list, out_dir)
    plot_evolution(df, meta_list, out_dir)

    gt = load_ground_truth(args.ground_truth)
    compute_mislead_indices(df, meta_list, gt, out_dir)
    root_cause_analysis(df, meta_list, out_dir)

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
