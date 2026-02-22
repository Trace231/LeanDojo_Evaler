"""
run_dataset_eval.py
===================
Instrumented search evaluation on the LeanDojo benchmark.

Uses the pre-downloaded benchmark (JSON with traced theorems) and a bare
Pantograph server.  For each theorem it runs a depth-limited search,
computes per-step log-probabilities, and exports search trees in the
exact JSON format expected by search_analysis_master.py.

Usage:
    # 1. Download the benchmark first
    python download_benchmark.py --sample 50

    # 2. Run evaluation
    python run_dataset_eval.py

    # 3. Analyse
    python search_analysis_master.py --input_dir logs/search_trees
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Pantograph imports (only needed if a live server is available)
# ---------------------------------------------------------------------------
try:
    from pantograph.server import Server, ServerError, TacticFailure
    HAS_PANTOGRAPH = True
except ImportError:
    HAS_PANTOGRAPH = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

DATASET_PATH = "data/leandojo_benchmark/random/test.json"
SAMPLED_DATASET_PATH = "data/leandojo_benchmark/random/test_sampled_50.json"

LOG_DIR = Path("logs/search_trees")

MAX_STEPS = 100            # Max search iterations per theorem
MAX_TRIALS_PER_GOAL = 5    # Max tactic attempts per goal before backtracking
NUM_TACTIC_CANDIDATES = 5  # Tactics to sample per step
MAX_THEOREMS = 50          # How many theorems to evaluate (0 = all)


# ---------------------------------------------------------------------------
# Search tree data structure (mirrors BestFirstSearchProver's export format)
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    """A node in the search tree, serialisable to the JSON format
    expected by search_analysis_master.py."""
    node_type: str                          # InternalNode / ErrorNode / ProofFinishedNode
    state_text: Optional[str] = None
    tactic: Optional[str] = None            # Tactic that LED to this node
    step_logprob: float = 0.0
    cumulative_logprob: float = 0.0
    depth: int = 0
    ppl: Optional[float] = None
    order_of_expansion: int = -1            # -1 = not yet expanded
    error_message: Optional[str] = None
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "node_type": self.node_type,
            "state_text": self.state_text,
            "tactic": self.tactic,
            "step_logprob": self.step_logprob,
            "cumulative_logprob": self.cumulative_logprob,
            "depth": self.depth,
            "ppl": self.ppl,
            "order_of_expansion": self.order_of_expansion,
            "error_message": self.error_message,
            "children": [c.to_dict() for c in self.children],
        }
        return d


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------
def compute_step_logprob(
    model, tokenizer, prompt: str, tactic: str, device: torch.device,
) -> float:
    """Compute the average per-token log-probability of *tactic* given *prompt*.

    This is equivalent to teacher-forced scoring: we concatenate
    prompt + tactic, run a forward pass, and extract the log-probs of
    only the tactic tokens.
    """
    full_text = prompt + tactic
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tactic_len = len(full_ids) - len(prompt_ids)

    if tactic_len <= 0:
        return 0.0

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Log-softmax over vocab
    log_probs = torch.log_softmax(logits, dim=-1)

    # Extract log-probs for tactic tokens
    # Token at position i predicts token i+1
    tactic_start = len(prompt_ids)
    total_lp = 0.0
    for i in range(tactic_len):
        token_idx = full_ids[tactic_start + i]
        pos = tactic_start + i - 1  # logit at pos predicts token at pos+1
        if pos >= 0:
            total_lp += log_probs[pos, token_idx].item()

    avg_lp = total_lp / tactic_len
    return avg_lp


# ---------------------------------------------------------------------------
# Tactic generation with log-probs
# ---------------------------------------------------------------------------
def build_prompt(goal_str: str) -> str:
    return (
        "### System:\n"
        "You are a Lean 4 tactic generator. Given a goal state, "
        "output exactly ONE Lean tactic that advances or solves the goal.\n"
        "Rules:\n"
        "- Output only the tactic text; no prose, quotes, or code fences.\n"
        "- Single line only; no `by` blocks.\n"
        "- Never use `sorry` or `admit`.\n"
        "### User:\n"
        f"{goal_str}\n\n"
        "### Assistant:\n"
    )


def generate_tactics(
    model, tokenizer, goal_str: str, device: torch.device, n: int = 5,
) -> List[dict]:
    """Generate *n* tactic candidates and score each one.

    Returns a list of dicts: [{"tactic": str, "step_logprob": float}, ...]
    sorted by descending log-prob (best first).
    """
    prompt = build_prompt(goal_str)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            num_return_sequences=n,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    seen = set()
    candidates = []
    for text in generated_texts:
        tactic = text[len(prompt):].strip().split("\n")[0].split("<;>")[0].strip()
        if not tactic or tactic == "sorry" or tactic in seen:
            continue
        seen.add(tactic)
        lp = compute_step_logprob(model, tokenizer, prompt, tactic, device)
        candidates.append({"tactic": tactic, "step_logprob": lp})

    # Sort by log-prob descending (best first)
    candidates.sort(key=lambda x: x["step_logprob"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Instrumented search (offline mode — no Pantograph)
# ---------------------------------------------------------------------------
def search_offline(
    model,
    tokenizer,
    device: torch.device,
    theorem_name: str,
    initial_goal: str,
) -> dict:
    """Run search WITHOUT a Pantograph server.

    Since we have no live Lean verifier, every tactic attempt is recorded
    as 'tried but unverified'.  The tree still captures the model's
    generation behaviour — which tactics it proposes, their log-probs,
    and how confident it is — which is exactly what the hallucination
    analysis needs.
    """
    t0 = time.time()
    expansion_counter = 0

    root = TreeNode(
        node_type="InternalNode",
        state_text=initial_goal,
        depth=0,
        order_of_expansion=0,
    )
    expansion_counter += 1

    # BFS-like expansion: expand nodes level by level
    frontier = [root]

    max_depth = 6  # Keep trees manageable in offline mode
    total_nodes = 1

    for depth in range(1, max_depth + 1):
        next_frontier = []
        for parent in frontier:
            if parent.node_type != "InternalNode":
                continue

            goal_str = parent.state_text or initial_goal
            candidates = generate_tactics(
                model, tokenizer, goal_str, device, n=NUM_TACTIC_CANDIDATES
            )

            if not candidates:
                # No tactic generated — dead end
                err_node = TreeNode(
                    node_type="ErrorNode",
                    tactic="<no_tactic_generated>",
                    step_logprob=0.0,
                    cumulative_logprob=parent.cumulative_logprob,
                    depth=depth,
                    ppl=None,
                    error_message="Model failed to generate any valid tactic",
                )
                parent.children.append(err_node)
                total_nodes += 1
                continue

            for cand in candidates:
                tactic = cand["tactic"]
                step_lp = cand["step_logprob"]
                cum_lp = parent.cumulative_logprob + step_lp
                ppl = math.exp(-cum_lp / depth) if cum_lp < 0 else None

                child = TreeNode(
                    node_type="InternalNode",
                    state_text=f"[after '{tactic}' on goal]",
                    tactic=tactic,
                    step_logprob=step_lp,
                    cumulative_logprob=cum_lp,
                    depth=depth,
                    ppl=ppl,
                    order_of_expansion=expansion_counter,
                )
                expansion_counter += 1
                parent.children.append(child)
                next_frontier.append(child)
                total_nodes += 1

        frontier = next_frontier
        if not frontier:
            break

    total_time = time.time() - t0

    return {
        "theorem_name": theorem_name,
        "theorem_statement": initial_goal,
        "file_path": "benchmark_dataset",
        "status": "Failed",  # Offline mode cannot verify proofs
        "proof": None,
        "total_time": total_time,
        "actor_time": total_time * 0.8,  # Approximate split
        "environment_time": total_time * 0.2,
        "num_total_nodes": total_nodes,
        "num_searched_nodes": expansion_counter,
        "search_tree": root.to_dict(),
    }


# ---------------------------------------------------------------------------
# Instrumented search (online mode — with Pantograph)
# ---------------------------------------------------------------------------
def search_online(
    model,
    tokenizer,
    device: torch.device,
    server: "Server",
    theorem_name: str,
    initial_goal: str,
) -> dict:
    """Run search WITH a live Pantograph server for tactic verification.

    This matches the BaseProver.search() logic but builds a proper tree
    with log-probs for export.
    """
    t0 = time.time()
    actor_time = 0.0
    env_time = 0.0
    expansion_counter = 0

    # Start goal in Pantograph
    try:
        goal_state = server.goal_start(initial_goal)
    except Exception as e:
        # Cannot even start the goal — create a minimal error tree
        root = TreeNode(
            node_type="ErrorNode",
            state_text=initial_goal,
            error_message=f"goal_start failed: {e}",
        )
        return {
            "theorem_name": theorem_name,
            "theorem_statement": initial_goal,
            "file_path": "benchmark_dataset",
            "status": "Failed",
            "proof": None,
            "total_time": time.time() - t0,
            "actor_time": 0.0,
            "environment_time": 0.0,
            "num_total_nodes": 1,
            "num_searched_nodes": 0,
            "search_tree": root.to_dict(),
        }

    root = TreeNode(
        node_type="InternalNode",
        state_text=str(goal_state),
        depth=0,
        order_of_expansion=0,
    )
    expansion_counter += 1

    # Stack-based DFS (matching BaseProver.search)
    stack = [(goal_state, root, 0)]  # (pantograph_state, tree_node, trials)
    status = "Failed"
    proof_tactics = None
    total_nodes = 1

    for i_step in range(MAX_STEPS):
        if not stack:
            break

        gs, current_node, trials = stack[-1]

        # Check if solved
        if len(gs.goals) == 0:
            current_node.node_type = "ProofFinishedNode"
            status = "Proved"
            # Collect proof path
            proof_tactics = _extract_proof_path(root)
            break

        if trials >= MAX_TRIALS_PER_GOAL:
            stack.pop()
            continue

        # Generate tactic candidates
        goal_str = str(gs)
        t_actor = time.time()
        candidates = generate_tactics(
            model, tokenizer, goal_str, device, n=NUM_TACTIC_CANDIDATES
        )
        actor_time += time.time() - t_actor

        if not candidates:
            stack.pop()
            continue

        # Try the next untried candidate
        cand = candidates[min(trials, len(candidates) - 1)]
        stack[-1] = (gs, current_node, trials + 1)

        tactic = cand["tactic"]
        step_lp = cand["step_logprob"]
        cum_lp = current_node.cumulative_logprob + step_lp
        depth = current_node.depth + 1
        ppl = math.exp(-cum_lp / depth) if cum_lp < 0 else None

        # Try applying the tactic in Pantograph
        t_env = time.time()
        try:
            next_gs = server.goal_tactic(gs, tactic)
            env_time += time.time() - t_env

            child = TreeNode(
                node_type="InternalNode",
                state_text=str(next_gs),
                tactic=tactic,
                step_logprob=step_lp,
                cumulative_logprob=cum_lp,
                depth=depth,
                ppl=ppl,
                order_of_expansion=expansion_counter,
            )
            expansion_counter += 1
            current_node.children.append(child)
            total_nodes += 1
            stack.append((next_gs, child, 0))

        except TacticFailure as e:
            env_time += time.time() - t_env
            err_child = TreeNode(
                node_type="ErrorNode",
                tactic=tactic,
                step_logprob=step_lp,
                cumulative_logprob=cum_lp,
                depth=depth,
                ppl=ppl,
                order_of_expansion=expansion_counter,
                error_message=str(e),
            )
            expansion_counter += 1
            current_node.children.append(err_child)
            total_nodes += 1

        except (ServerError, Exception) as e:
            env_time += time.time() - t_env
            err_child = TreeNode(
                node_type="ErrorNode",
                tactic=tactic,
                step_logprob=step_lp,
                cumulative_logprob=cum_lp,
                depth=depth,
                ppl=ppl,
                order_of_expansion=expansion_counter,
                error_message=str(e),
            )
            expansion_counter += 1
            current_node.children.append(err_child)
            total_nodes += 1

    total_time = time.time() - t0

    return {
        "theorem_name": theorem_name,
        "theorem_statement": initial_goal,
        "file_path": "benchmark_dataset",
        "status": status,
        "proof": proof_tactics,
        "total_time": total_time,
        "actor_time": actor_time,
        "environment_time": env_time,
        "num_total_nodes": total_nodes,
        "num_searched_nodes": expansion_counter,
        "search_tree": root.to_dict(),
    }


def _extract_proof_path(root: TreeNode) -> Optional[List[str]]:
    """Walk from root to the first ProofFinishedNode and collect tactics."""
    if root.node_type == "ProofFinishedNode":
        return [root.tactic] if root.tactic else []
    for child in root.children:
        sub = _extract_proof_path(child)
        if sub is not None:
            prefix = [child.tactic] if child.tactic else []
            return prefix + sub
    return None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_tree(tree_dict: dict, log_dir: Path) -> Path:
    """Write a search tree dict to logs/search_trees/{name}.json."""
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[/\\. ]', '_', tree_dict["theorem_name"])
    out_path = log_dir / f"{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(tree_dict, f, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Find dataset ----
    dataset_path = None
    for candidate in [SAMPLED_DATASET_PATH, DATASET_PATH, "data/random/test.json"]:
        if Path(candidate).exists():
            dataset_path = candidate
            break

    if dataset_path is None:
        print("No benchmark dataset found. Run first:")
        print("  python download_benchmark.py --sample 50")
        return

    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} theorems from {dataset_path}")

    # ---- Limit ----
    if MAX_THEOREMS > 0:
        dataset = dataset[:MAX_THEOREMS]

    # ---- Load model ----
    print(f"Loading model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # ---- Try to start Pantograph server ----
    server = None
    if HAS_PANTOGRAPH:
        try:
            server = Server()
            print("Pantograph server started (online mode)")
        except Exception as e:
            print(f"Pantograph unavailable ({e}), falling back to offline mode")
            server = None
    else:
        print("Pantograph not installed, using offline mode")

    mode = "online" if server else "offline"
    print(f"Mode: {mode}")
    print(f"Search trees will be saved to: {LOG_DIR}/")
    print("=" * 60)

    # ---- Evaluate ----
    stats = {"proved": 0, "failed": 0, "error": 0}

    for i, item in enumerate(dataset):
        name = item.get("full_name", f"theorem_{i}")
        traced = item.get("traced_tactics", [])
        if not traced:
            continue
        goal = traced[0].get("state_before", "")
        if not goal:
            continue

        print(f"\n[{i+1}/{len(dataset)}] {name}")
        print(f"  Goal: {goal[:120]}{'...' if len(goal) > 120 else ''}")

        try:
            if server is not None:
                tree_dict = search_online(
                    model, tokenizer, device, server, name, goal,
                )
            else:
                tree_dict = search_offline(
                    model, tokenizer, device, name, goal,
                )

            out_path = export_tree(tree_dict, LOG_DIR)
            status = tree_dict["status"]
            n_nodes = tree_dict["num_total_nodes"]
            t = tree_dict["total_time"]

            if status == "Proved":
                stats["proved"] += 1
                print(f"  --> Proved  ({n_nodes} nodes, {t:.1f}s)  -> {out_path}")
            else:
                stats["failed"] += 1
                print(f"  --> Failed  ({n_nodes} nodes, {t:.1f}s)  -> {out_path}")

        except Exception as e:
            stats["error"] += 1
            print(f"  --> Error: {e}")
            # Still export a minimal error tree
            err_tree = {
                "theorem_name": name,
                "theorem_statement": goal,
                "file_path": "benchmark_dataset",
                "status": "Failed",
                "proof": None,
                "total_time": 0.0,
                "actor_time": 0.0,
                "environment_time": 0.0,
                "num_total_nodes": 1,
                "num_searched_nodes": 0,
                "search_tree": TreeNode(
                    node_type="ErrorNode",
                    state_text=goal,
                    error_message=str(e),
                ).to_dict(),
            }
            export_tree(err_tree, LOG_DIR)

    # ---- Summary ----
    total = stats["proved"] + stats["failed"] + stats["error"]
    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE  ({mode} mode)")
    print(f"  Total:   {total}")
    print(f"  Proved:  {stats['proved']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Error:   {stats['error']}")
    print(f"\nSearch trees saved to: {LOG_DIR}/")
    print(f"Next step: python search_analysis_master.py --input_dir {LOG_DIR}")


if __name__ == "__main__":
    main()
