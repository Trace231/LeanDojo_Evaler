"""
run_dataset_eval.py
===================
Instrumented search evaluation on the LeanDojo benchmark with REAL
Pantograph verification against a built mathlib4 project.

For each theorem the script:
  1. Starts a Pantograph server with the theorem's module imported
  2. Uses env_inspect to get the theorem's type
  3. Runs DFS search: generate tactics -> verify with Pantograph -> build tree
  4. Computes per-step log-probabilities (teacher-forced scoring)
  5. Exports search trees in the JSON format expected by search_analysis_master.py

Prerequisites
-------------
  # 1. Set up mathlib4 (MUST match Pantograph's Lean version)
  bash setup_mathlib4.sh

  # 2. Download the benchmark
  python download_benchmark.py --sample 50

  # 3. Run evaluation
  python run_dataset_eval.py

  # 4. Analyse
  python search_analysis_master.py --input_dir logs/search_trees
"""

import json
import math
import os
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Pantograph imports
# ---------------------------------------------------------------------------
try:
    from pantograph.server import Server, ServerError, TacticFailure
    HAS_PANTOGRAPH = True
except ImportError:
    HAS_PANTOGRAPH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

DATASET_PATH = "data/leandojo_benchmark/random/test.json"
SAMPLED_DATASET_PATH = "data/leandojo_benchmark/random/test_sampled_50.json"

# Path to a built mathlib4 project (must match Pantograph's Lean version).
# Run `bash setup_mathlib4.sh` to set this up.
# Set to None to use offline mode (no real tactic verification).
MATHLIB_PROJECT_PATH = "/tmp/mathlib4"

LOG_DIR = Path("logs/search_trees")

MAX_STEPS = 100            # Max search iterations per theorem
MAX_TRIALS_PER_GOAL = 5    # Max tactic attempts per goal before backtracking
NUM_TACTIC_CANDIDATES = 5  # Tactics to sample per step
MAX_THEOREMS = 50          # How many theorems to evaluate (0 = all)


# ---------------------------------------------------------------------------
# Search tree data structure
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    """A node in the search tree, serialisable to the JSON format
    expected by search_analysis_master.py."""
    node_type: str                          # InternalNode / ErrorNode / ProofFinishedNode
    state_text: Optional[str] = None
    tactic: Optional[str] = None
    step_logprob: float = 0.0
    cumulative_logprob: float = 0.0
    depth: int = 0
    ppl: Optional[float] = None
    order_of_expansion: int = -1
    error_message: Optional[str] = None
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
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


# ---------------------------------------------------------------------------
# Tactic cleaning — prevents "theorem X := by ..." hallucinations
# ---------------------------------------------------------------------------
_DECL_RE = re.compile(
    r'^(theorem|lemma|def|example|private\s+theorem|private\s+lemma)\s+'
)
_BY_RE = re.compile(r'\bby\b\s*(.*)', re.DOTALL)
_FENCE_RE = re.compile(r'^```\w*\s*')
_FENCE_END_RE = re.compile(r'\s*```$')


def clean_tactic(raw: str) -> str:
    """Post-process a model-generated tactic to extract a pure tactic.

    Handles common failure modes:
      - Model outputs 'theorem Foo : ... := by simp' instead of just 'simp'
      - Model wraps output in code fences
      - Model outputs quotes around the tactic
    """
    # Take first line, strip combinators
    tactic = raw.strip().split("\n")[0].split("<;>")[0].strip()

    # Strip code fences
    tactic = _FENCE_RE.sub('', tactic)
    tactic = _FENCE_END_RE.sub('', tactic)

    # Strip wrapping quotes/backticks
    if len(tactic) >= 2 and tactic[0] == tactic[-1] and tactic[0] in '`"\'':
        tactic = tactic[1:-1]

    # If the model output is a theorem/lemma declaration, extract the tactic after 'by'
    if _DECL_RE.match(tactic):
        by_match = _BY_RE.search(tactic)
        if by_match:
            tactic = by_match.group(1).strip()
        else:
            # Declaration without 'by' — not a usable tactic
            return ""

    # Remove leading 'by ' if present (we just want the tactic itself)
    if tactic.startswith("by "):
        tactic = tactic[3:].strip()

    # Final cleanup
    tactic = tactic.strip()

    # Reject known-bad outputs
    if tactic in ("sorry", "admit", ""):
        return ""

    return tactic


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------
def compute_step_logprob(
    model, tokenizer, prompt: str, tactic: str, device: torch.device,
) -> float:
    """Compute the average per-token log-probability of *tactic* given *prompt*.

    Teacher-forced scoring: concatenate prompt + tactic, forward pass,
    extract log-probs of tactic tokens only.
    """
    full_text = prompt + tactic
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tactic_len = len(full_ids) - len(prompt_ids)

    if tactic_len <= 0:
        return 0.0

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (seq_len, vocab_size)

    log_probs = torch.log_softmax(logits, dim=-1)

    tactic_start = len(prompt_ids)
    total_lp = 0.0
    for i in range(tactic_len):
        token_idx = full_ids[tactic_start + i]
        pos = tactic_start + i - 1
        if pos >= 0:
            total_lp += log_probs[pos, token_idx].item()

    return total_lp / tactic_len


# ---------------------------------------------------------------------------
# Tactic generation with log-probs
# ---------------------------------------------------------------------------
def build_prompt(goal_str: str) -> str:
    """Build a prompt that strongly constrains the model to output a pure tactic."""
    return (
        "### System:\n"
        "You are a Lean 4 tactic generator.\n"
        "Given a proof goal state, output exactly ONE tactic that makes progress.\n"
        "CRITICAL RULES:\n"
        "- Output ONLY the tactic text (e.g., 'simp', 'ring', 'exact h').\n"
        "- Do NOT output theorem/lemma/def declarations.\n"
        "- Do NOT wrap in code fences, quotes, or markdown.\n"
        "- Do NOT write 'by' before the tactic.\n"
        "- Single line only. No semicolons, no multi-step combinator.\n"
        "- Never use 'sorry' or 'admit'.\n"
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
        raw = text[len(prompt):]
        tactic = clean_tactic(raw)
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        lp = compute_step_logprob(model, tokenizer, prompt, tactic, device)
        candidates.append({"tactic": tactic, "step_logprob": lp})

    candidates.sort(key=lambda x: x["step_logprob"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Pantograph server management
# ---------------------------------------------------------------------------
_server_cache: Dict[str, "Server"] = {}


def get_server_for_theorem(
    file_path: str, mathlib_path: str
) -> Optional["Server"]:
    """Create (or reuse) a Pantograph server for the given theorem's module.

    The server imports the module that contains the theorem so that
    env_inspect can find the theorem and goal_tactic can verify tactics
    against the full Mathlib environment.
    """
    if not HAS_PANTOGRAPH:
        return None

    # Convert file path to import name:
    #   "Mathlib/Analysis/BoxIntegral/Box/Basic.lean" -> "Mathlib.Analysis.BoxIntegral.Box.Basic"
    import_name = file_path.replace(".lean", "").replace("/", ".")

    # Check cache
    if import_name in _server_cache:
        return _server_cache[import_name]

    project = Path(mathlib_path)
    if not project.exists():
        logger.warning(f"Mathlib project not found at {mathlib_path}")
        return None

    try:
        server = Server(
            imports=["Init", import_name],
            project_path=project,
        )
        _server_cache[import_name] = server
        return server
    except Exception as e:
        logger.warning(f"Server init failed for {import_name}: {e}")
        return None


def setup_goal(server: "Server", full_name: str) -> Tuple[object, str]:
    """Use env_inspect to get the theorem's type, then start a goal.

    Returns (goal_state_obj, initial_goal_str).
    Raises on failure.
    """
    info = server.env_inspect(full_name)
    goal_type = info["type"]["pp"]
    goal_state = server.goal_start(goal_type)
    initial_goal_str = str(goal_state)
    return goal_state, initial_goal_str


# ---------------------------------------------------------------------------
# Instrumented search (offline mode - no Pantograph)
# ---------------------------------------------------------------------------
def search_offline(
    model,
    tokenizer,
    device: torch.device,
    theorem_name: str,
    initial_goal: str,
    file_path: str = "benchmark_dataset",
) -> dict:
    """Run search WITHOUT a Pantograph server.

    Uses the benchmark's state_before as the goal text for tactic generation.
    Cannot verify tactics, but captures model behaviour (tactics, log-probs,
    confidence) which is what the hallucination analysis needs.
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
    max_depth = 6
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
        "file_path": file_path,
        "status": "Failed",  # Offline cannot verify
        "proof": None,
        "total_time": total_time,
        "actor_time": total_time * 0.8,
        "environment_time": total_time * 0.2,
        "num_total_nodes": total_nodes,
        "num_searched_nodes": expansion_counter,
        "search_tree": root.to_dict(),
    }


# ---------------------------------------------------------------------------
# Instrumented search (online mode - with Pantograph + mathlib4)
# ---------------------------------------------------------------------------
def search_online(
    model,
    tokenizer,
    device: torch.device,
    server: "Server",
    theorem_name: str,
    goal_state_obj,
    initial_goal_str: str,
    file_path: str = "benchmark_dataset",
) -> dict:
    """DFS search with REAL Pantograph verification + full tree tracking.

    Identical logic to run_modern_eval.py's instrumented_search().
    """
    t0 = time.time()
    actor_time = 0.0
    env_time = 0.0
    expansion_counter = 0

    root = TreeNode(
        node_type="InternalNode",
        state_text=initial_goal_str,
        depth=0,
        order_of_expansion=0,
    )
    expansion_counter += 1
    total_nodes = 1

    # Stack: (pantograph_goal_state, tree_node, trial_index)
    stack = [(goal_state_obj, root, 0)]
    status = "Failed"
    proof_tactics = None

    for i_step in range(MAX_STEPS):
        if not stack:
            break

        gs, current_node, trials = stack[-1]

        # Check solved
        if len(gs.goals) == 0:
            current_node.node_type = "ProofFinishedNode"
            status = "Proved"
            proof_tactics = _extract_proof_path(root)
            break

        if trials >= MAX_TRIALS_PER_GOAL:
            stack.pop()
            continue

        goal_str = str(gs)

        # Generate tactic candidates
        t_a = time.time()
        candidates = generate_tactics(
            model, tokenizer, goal_str, device, n=NUM_TACTIC_CANDIDATES,
        )
        actor_time += time.time() - t_a

        if not candidates:
            stack.pop()
            continue

        cand = candidates[min(trials, len(candidates) - 1)]
        stack[-1] = (gs, current_node, trials + 1)

        tactic = cand["tactic"]
        step_lp = cand["step_logprob"]
        cum_lp = current_node.cumulative_logprob + step_lp
        depth = current_node.depth + 1
        ppl = math.exp(-cum_lp / depth) if cum_lp < 0 else None

        # Apply tactic in Pantograph (REAL verification)
        t_e = time.time()
        try:
            next_gs = server.goal_tactic(gs, tactic)
            env_time += time.time() - t_e

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
            env_time += time.time() - t_e
            err = TreeNode(
                node_type="ErrorNode",
                tactic=tactic,
                step_logprob=step_lp,
                cumulative_logprob=cum_lp,
                depth=depth, ppl=ppl,
                order_of_expansion=expansion_counter,
                error_message=str(e),
            )
            expansion_counter += 1
            current_node.children.append(err)
            total_nodes += 1

        except (ServerError, Exception) as e:
            env_time += time.time() - t_e
            err = TreeNode(
                node_type="ErrorNode",
                tactic=tactic,
                step_logprob=step_lp,
                cumulative_logprob=cum_lp,
                depth=depth, ppl=ppl,
                order_of_expansion=expansion_counter,
                error_message=str(e),
            )
            expansion_counter += 1
            current_node.children.append(err)
            total_nodes += 1

    total_time = time.time() - t0

    return {
        "theorem_name": theorem_name,
        "theorem_statement": initial_goal_str,
        "file_path": file_path,
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Model loaded on {device}")

    # ---- Check mathlib4 project path ----
    use_online = False
    mathlib_path = MATHLIB_PROJECT_PATH
    if mathlib_path and Path(mathlib_path).exists() and HAS_PANTOGRAPH:
        print(f"Mathlib4 project found at: {mathlib_path}")
        use_online = True
    elif mathlib_path and not Path(mathlib_path).exists():
        print(f"WARNING: MATHLIB_PROJECT_PATH={mathlib_path} does not exist.")
        print("  Run `bash setup_mathlib4.sh` to set up mathlib4.")
        print("  Falling back to OFFLINE mode.")
    elif not HAS_PANTOGRAPH:
        print("WARNING: Pantograph not installed. Using OFFLINE mode.")
    else:
        print("MATHLIB_PROJECT_PATH not set. Using OFFLINE mode.")

    mode = "online" if use_online else "offline"
    print(f"Mode: {mode}")
    print(f"Search trees will be saved to: {LOG_DIR}/")
    print("=" * 60)

    # ---- Evaluate ----
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"proved": 0, "failed": 0, "error": 0, "skipped": 0}

    for i, item in enumerate(dataset):
        name = item.get("full_name", f"theorem_{i}")
        file_path = item.get("file_path", "unknown")
        traced = item.get("traced_tactics", [])
        if not traced:
            stats["skipped"] += 1
            continue
        benchmark_goal = traced[0].get("state_before", "")
        if not benchmark_goal:
            stats["skipped"] += 1
            continue

        print(f"\n[{i+1}/{len(dataset)}] {name}")
        print(f"  File: {file_path}")

        try:
            if use_online:
                # --- Online mode: real Pantograph verification ---
                server = get_server_for_theorem(file_path, mathlib_path)
                if server is None:
                    print("  Server init failed, falling back to offline for this theorem")
                    tree_dict = search_offline(
                        model, tokenizer, device, name, benchmark_goal, file_path,
                    )
                else:
                    # Use env_inspect to get the theorem's type for goal_start
                    try:
                        goal_state, initial_goal_str = setup_goal(server, name)
                        print(f"  Goal: {initial_goal_str[:120]}...")

                        tree_dict = search_online(
                            model, tokenizer, device,
                            server, name,
                            goal_state, initial_goal_str,
                            file_path,
                        )
                    except Exception as e:
                        print(f"  env_inspect/goal_start failed: {e}")
                        print(f"  Falling back to offline (using benchmark state_before)")
                        print(f"  Goal: {benchmark_goal[:120]}...")
                        tree_dict = search_offline(
                            model, tokenizer, device, name, benchmark_goal, file_path,
                        )
            else:
                # --- Offline mode ---
                print(f"  Goal: {benchmark_goal[:120]}...")
                tree_dict = search_offline(
                    model, tokenizer, device, name, benchmark_goal, file_path,
                )

            out_path = export_tree(tree_dict, LOG_DIR)
            status = tree_dict["status"]
            n_nodes = tree_dict["num_total_nodes"]
            t = tree_dict["total_time"]

            if status == "Proved":
                stats["proved"] += 1
                print(f"  --> Proved  ({n_nodes} nodes, {t:.1f}s)")
                if tree_dict["proof"]:
                    for tac in tree_dict["proof"]:
                        print(f"      {tac}")
            else:
                stats["failed"] += 1
                print(f"  --> Failed  ({n_nodes} nodes, {t:.1f}s)")

            print(f"  Tree: {out_path}")

        except Exception as e:
            stats["error"] += 1
            print(f"  --> Error: {e}")
            import traceback
            traceback.print_exc()
            err_tree = {
                "theorem_name": name,
                "theorem_statement": benchmark_goal,
                "file_path": file_path,
                "status": "Failed",
                "proof": None,
                "total_time": 0.0,
                "actor_time": 0.0,
                "environment_time": 0.0,
                "num_total_nodes": 1,
                "num_searched_nodes": 0,
                "search_tree": TreeNode(
                    node_type="ErrorNode",
                    state_text=benchmark_goal,
                    error_message=str(e),
                ).to_dict(),
            }
            export_tree(err_tree, LOG_DIR)

    # ---- Summary ----
    total = stats["proved"] + stats["failed"] + stats["error"]
    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE  ({mode} mode)")
    print(f"  Total:    {total}")
    print(f"  Proved:   {stats['proved']}")
    print(f"  Failed:   {stats['failed']}")
    print(f"  Error:    {stats['error']}")
    print(f"  Skipped:  {stats['skipped']}")
    if total > 0:
        print(f"  Pass@1:   {stats['proved']/total*100:.1f}%")
    print(f"\nSearch trees saved to: {LOG_DIR}/")
    print(f"Next step: python search_analysis_master.py --input_dir {LOG_DIR}")


if __name__ == "__main__":
    main()
