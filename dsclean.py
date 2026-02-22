"""
run_modern_eval.py
==================
Full-pipeline evaluation for DeepSeek-Prover-V2 on Lean 4 standard benchmarks.
Includes text sanitization to bridge DeepSeek's output format with Pantograph.
"""

import json
import math
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from pantograph import Server
from pantograph.server import ServerError, TacticFailure

from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"

# 切换为标准的 miniF2F Lean 4 测试集
URL = "https://github.com/amirlfe/miniF2F-lean4"
COMMIT = "0351f045bb62768560ca9142b451d08e5c184cf4" # 请确保这是你本地能够成功 clone 并编译的 commit

LOCAL_PROJECT_PATH = None # 如果使用上面的远程仓库，设为 None 即可

LOG_DIR = Path("logs/search_trees")
MAX_STEPS = 100
MAX_TRIALS_PER_GOAL = 5
NUM_TACTIC_CANDIDATES = 5


# ── Search tree data structure ────────────────────────────────────────────
@dataclass
class TreeNode:
    node_type: str
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


# ── Text Sanitization (核心修复点) ─────────────────────────────────────────
def clean_tactic_output(raw_text: str) -> str:
    """清理模型输出，使其成为纯净的 Lean 4 Tactic"""
    text = raw_text.strip()
    
    # 1. 移除可能存在的 Markdown 代码块标记 (```lean ... ```)
    text = re.sub(r'```lean\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    # 2. 移除行内代码的反引号 (`tactic`)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # 3. 如果模型输出了定理声明，截取 `:= by` 之后的内容
    if ':= by' in text:
        text = text.split(':= by')[-1]
    
    # 4. 只取第一行非空的命令 (防止模型一次性输出多行证明)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return ""
    
    tactic = lines[0]
    
    # 5. 过滤掉无意义或作弊的 tactic
    if tactic in ["sorry", "admit"]:
        return ""
        
    return tactic


# ── Log-prob scoring ──────────────────────────────────────────────────────
def compute_step_logprob(model, tokenizer, prompt: str, tactic: str, device) -> float:
    full_text = prompt + tactic
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tactic_len = len(full_ids) - len(prompt_ids)
    if tactic_len <= 0:
        return 0.0

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    tactic_start = len(prompt_ids)
    total_lp = 0.0
    for i in range(tactic_len):
        token_idx = full_ids[tactic_start + i]
        pos = tactic_start + i - 1
        if pos >= 0:
            total_lp += log_probs[pos, token_idx].item()

    return total_lp / tactic_len


def build_prompt(goal_str: str) -> str:
    # 强化了格式要求的 Prompt
    return (
        "### System:\n"
        "You are an expert Lean 4 tactic generator. Given the current tactic state, "
        "output exactly ONE valid Lean 4 tactic to advance the proof.\n"
        "CRITICAL RULES:\n"
        "- DO NOT output the theorem declaration or `:= by`.\n"
        "- DO NOT wrap the tactic in markdown backticks.\n"
        "- Output ONLY the raw tactic text on a single line.\n"
        "### User:\n"
        f"{goal_str}\n\n"
        "### Assistant:\n"
    )


def generate_tactics_with_logprobs(model, tokenizer, goal_str: str, device, n: int = 5):
    prompt = build_prompt(goal_str)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

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

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    seen = set()
    candidates = []
    
    for text in texts:
        raw_output = text[len(prompt):]
        # 调用我们新增的清理函数
        tactic = clean_tactic_output(raw_output)
        
        if not tactic or tactic in seen:
            continue
            
        seen.add(tactic)
        lp = compute_step_logprob(model, tokenizer, prompt, tactic, device)
        candidates.append({"tactic": tactic, "step_logprob": lp})

    candidates.sort(key=lambda x: x["step_logprob"], reverse=True)
    return candidates


# ── Instrumented search (uses REAL Pantograph server with project context) ─
def instrumented_search(
    model, tokenizer, device,
    server: Server,
    theorem_name: str,
    goal_state_obj, 
    initial_goal_str: str,
) -> dict:
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

    stack = [(goal_state_obj, root, 0)]
    status = "Failed"
    proof_tactics = None

    for i_step in range(MAX_STEPS):
        if not stack:
            break

        gs, current_node, trials = stack[-1]

        if len(gs.goals) == 0:
            current_node.node_type = "ProofFinishedNode"
            status = "Proved"
            proof_tactics = _extract_proof_path(root)
            break

        if trials >= MAX_TRIALS_PER_GOAL:
            stack.pop()
            continue

        goal_str = str(gs)

        t_a = time.time()
        candidates = generate_tactics_with_logprobs(
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
        "file_path": "minif2f",
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
    if root.node_type == "ProofFinishedNode":
        return [root.tactic] if root.tactic else []
    for child in root.children:
        sub = _extract_proof_path(child)
        if sub is not None:
            prefix = [child.tactic] if child.tactic else []
            return prefix + sub
    return None


def export_tree(tree_dict: dict, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[/\\. ]', '_', tree_dict["theorem_name"])
    out = log_dir / f"{safe}.json"
    with open(out, "w") as f:
        json.dump(tree_dict, f, indent=2, default=str)
    return out


class FullMathlibAgent(HFAgent):
    def _get_build_deps(self) -> bool:
        return True


def main():
    trainer = SFTTrainer(
        model_name=MODEL_NAME,
        output_dir="outputs-deepseek",
        epochs_per_repo=1,
        batch_size=2,
    )
    agent = FullMathlibAgent(trainer=trainer)
    agent.output_dir = MODEL_NAME

    print(f"Stage 1: Setting up repository {URL}...")
    agent.setup_github_repository(url=URL, commit=COMMIT)

    print("Stage 1: Initializing prover...")
    sorry_theorems = agent.initialize_prover()

    if not sorry_theorems:
        print("No sorry theorems found.")
        return

    print(f"Found {len(sorry_theorems)} sorry theorems to prove")

    model = agent.prover.model
    tokenizer = agent.prover.tokenizer
    device = agent.prover.device

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"proved": 0, "failed": 0, "error": 0}

    for i, (theorem, repo) in enumerate(sorry_theorems):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sorry_theorems)}] {theorem.full_name}")

        if LOCAL_PROJECT_PATH:
            project_path = Path(LOCAL_PROJECT_PATH)
        else:
            project_path = get_traced_repo_path(repo, build_deps=True)
            
        try:
            server = Server(
                imports=["Init", str(theorem.file_path).replace(".lean", "")],
                project_path=project_path,
            )
        except Exception as e:
            print(f"  Server init failed: {e}")
            stats["error"] += 1
            continue

        try:
            goal_type = server.env_inspect(theorem.full_name)["type"]["pp"]
            goal_state = server.goal_start(goal_type)
            initial_goal_str = str(goal_state)
        except Exception as e:
            print(f"  goal_start failed: {e}")
            stats["error"] += 1
            continue

        print(f"  Goal: {initial_goal_str[:120]}...")

        tree_dict = instrumented_search(
            model, tokenizer, device,
            server, theorem.full_name,
            goal_state, initial_goal_str,
        )

        out_path = export_tree(tree_dict, LOG_DIR)
        s = tree_dict["status"]
        n = tree_dict["num_total_nodes"]
        t = tree_dict["total_time"]

        if s == "Proved":
            stats["proved"] += 1
            print(f"  --> Proved  ({n} nodes, {t:.1f}s)")
            if tree_dict["proof"]:
                for tac in tree_dict["proof"]:
                    print(f"      {tac}")
        else:
            stats["failed"] += 1
            print(f"  --> Failed  ({n} nodes, {t:.1f}s)")

        print(f"  Tree saved: {out_path}")

    total = stats["proved"] + stats["failed"] + stats["error"]
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"  Total:   {total}")
    print(f"  Proved:  {stats['proved']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Error:   {stats['error']}")
    print(f"\nSearch trees: {LOG_DIR}/")


if __name__ == "__main__":
    main()