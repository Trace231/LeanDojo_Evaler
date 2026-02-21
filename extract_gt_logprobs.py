"""
extract_gt_logprobs.py
======================
Teacher-Forcing scoring script for LeanDojo-v2.

Computes the Ground Truth cumulative log-probability of expert (human-written)
tactic sequences by running the ReProver model in teacher-forcing mode:
for each proof step the model scores the *known correct* tactic rather than
generating its own.

Output: a JSON dict  { theorem_full_name: gt_cumulative_logprob, ... }
that can be fed to ``search_analysis_master.py --ground_truth``.

Usage
-----
# Dojo mode (default) — live Lean interaction, most accurate states
python extract_gt_logprobs.py \
    --dataset_path  data/random/test.json \
    --ckpt_path     checkpoints/reprover.ckpt \
    --output_path   logs/gt_logprobs.json \
    --timeout       300

# Offline mode — uses state_before from dataset directly, no Lean needed
python extract_gt_logprobs.py \
    --dataset_path  data/random/test.json \
    --ckpt_path     checkpoints/reprover.ckpt \
    --output_path   logs/gt_logprobs.json \
    --offline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from lean_dojo_v2.lean_agent.common import format_state, format_augmented_state
from lean_dojo_v2.lean_agent.generator.model import RetrievalAugmentedGenerator
from lean_dojo_v2.lean_dojo import (
    Dojo,
    DojoCrashError,
    DojoInitError,
    DojoTacticTimeoutError,
    LeanError,
    LeanGitRepo,
    Pos,
    ProofFinished,
    ProofGivenUp,
    TacticState,
    Theorem,
)
from lean_dojo_v2.utils.common import zip_strict
from lean_dojo_v2.utils.constants import remove_marks


# ---------------------------------------------------------------------------
# Core: score a single tactic given a state
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_tactic(
    model: RetrievalAugmentedGenerator,
    state_text: str,
    tactic_text: str,
) -> float:
    """Compute log P(tactic | state) via a single teacher-forcing forward pass.

    Returns the *sum* of per-token log-probabilities (not the average),
    which is the quantity that gets accumulated into cumulative_logprob
    during best-first search.
    """
    tokenizer = model.tokenizer
    device = model.device

    # --- Encode state (encoder input) ---
    enc = tokenizer(
        state_text,
        padding=False,
        max_length=model.max_inp_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    state_ids = enc.input_ids.to(device)
    state_mask = enc.attention_mask.to(device)

    # --- Encode target tactic (decoder labels) ---
    dec = tokenizer(
        tactic_text,
        padding=False,
        max_length=model.max_oup_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    tactic_ids = dec.input_ids.to(device)  # (1, tgt_len)

    # Mask padding tokens with -100 (ignored in loss)
    label_ids = tactic_ids.clone()
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    # --- Forward pass ---
    output = model.generator(
        input_ids=state_ids,
        attention_mask=state_mask,
        labels=label_ids,
    )
    logits = output.logits  # (1, tgt_len, vocab_size)

    # --- Gather log-probs at target positions ---
    log_probs = F.log_softmax(logits, dim=-1)  # (1, tgt_len, vocab)

    # Replace -100 with 0 for gathering (masked out below)
    gather_ids = label_ids.clone()
    gather_ids[gather_ids == -100] = 0
    token_log_probs = log_probs.gather(2, gather_ids.unsqueeze(2)).squeeze(2)  # (1, tgt_len)

    # Sum only over real (non-padding) tokens
    mask = (label_ids != -100).float()
    total_log_prob = (token_log_probs * mask).sum().item()

    return total_log_prob


# ---------------------------------------------------------------------------
# Dojo mode: live Lean interaction
# ---------------------------------------------------------------------------


def score_theorem_dojo(
    model: RetrievalAugmentedGenerator,
    thm_data: dict,
    timeout: int,
) -> Optional[Tuple[str, float, List[dict]]]:
    """Score one theorem using Dojo for live state tracking.

    Returns (full_name, cumulative_logprob, per_step_details) or None on failure.
    """
    full_name = thm_data["full_name"]
    traced_tactics = thm_data.get("traced_tactics", [])

    if not traced_tactics:
        logger.warning(f"[{full_name}] No traced tactics, skipping.")
        return None

    # Filter out trivial steps
    tactics_to_score = [
        t for t in traced_tactics
        if t.get("state_before", "") != "no goals" and "·" not in t.get("tactic", "")
    ]
    if not tactics_to_score:
        logger.warning(f"[{full_name}] No valid tactics after filtering, skipping.")
        return None

    # Build Theorem object for Dojo
    repo = LeanGitRepo(thm_data["url"], thm_data["commit"])
    start = Pos(*thm_data["start"])
    end = Pos(*thm_data["end"])
    thm = Theorem(repo, thm_data["file_path"], full_name, start, end)

    cumulative_logprob = 0.0
    step_details = []

    try:
        with Dojo(thm, timeout) as (dojo, init_state):
            state = init_state

            for i, tac_data in enumerate(tactics_to_score):
                tactic_str = tac_data["tactic"]

                # Get pretty-printed state
                if isinstance(state, TacticState):
                    state_text = format_state(state.pp)
                else:
                    state_text = format_state(str(state))

                # Score the expert tactic
                step_lp = score_tactic(model, state_text, tactic_str)
                cumulative_logprob += step_lp

                step_details.append({
                    "step": i,
                    "tactic": tactic_str,
                    "step_logprob": step_lp,
                    "cumulative_logprob": cumulative_logprob,
                })

                # Execute the tactic to advance Lean state
                response = dojo.run_tac(state, tactic_str)

                if isinstance(response, ProofFinished):
                    logger.debug(f"[{full_name}] Proof finished at step {i}.")
                    break
                elif isinstance(response, TacticState):
                    state = response
                elif isinstance(response, LeanError):
                    logger.warning(
                        f"[{full_name}] LeanError at step {i}: "
                        f"{getattr(response, 'error', str(response))}"
                    )
                    break
                else:
                    logger.warning(
                        f"[{full_name}] Unexpected response at step {i}: {type(response)}"
                    )
                    break

    except DojoInitError as e:
        logger.warning(f"[{full_name}] DojoInitError: {e}")
        return None
    except DojoCrashError as e:
        logger.warning(f"[{full_name}] DojoCrashError: {e}")
        return None
    except Exception as e:
        logger.warning(f"[{full_name}] Unexpected error: {e}")
        return None

    return full_name, cumulative_logprob, step_details


# ---------------------------------------------------------------------------
# Offline mode: use state_before from dataset (no Lean required)
# ---------------------------------------------------------------------------


def score_theorem_offline(
    model: RetrievalAugmentedGenerator,
    thm_data: dict,
) -> Optional[Tuple[str, float, List[dict]]]:
    """Score one theorem using pre-recorded state_before from the dataset."""
    full_name = thm_data["full_name"]
    traced_tactics = thm_data.get("traced_tactics", [])

    if not traced_tactics:
        return None

    tactics_to_score = [
        t for t in traced_tactics
        if t.get("state_before", "") != "no goals" and "·" not in t.get("tactic", "")
    ]
    if not tactics_to_score:
        return None

    cumulative_logprob = 0.0
    step_details = []

    for i, tac_data in enumerate(tactics_to_score):
        tactic_str = tac_data["tactic"]
        state_text = format_state(tac_data["state_before"])

        try:
            step_lp = score_tactic(model, state_text, tactic_str)
        except Exception as e:
            logger.warning(f"[{full_name}] Scoring error at step {i}: {e}")
            return None

        cumulative_logprob += step_lp
        step_details.append({
            "step": i,
            "tactic": tactic_str,
            "step_logprob": step_lp,
            "cumulative_logprob": cumulative_logprob,
        })

    return full_name, cumulative_logprob, step_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute Ground Truth cumulative log-probs via teacher forcing."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to test.json (LeanDojo export with traced_tactics).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to ReProver / RetrievalAugmentedGenerator checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/gt_logprobs.json",
        help="Output JSON path (default: logs/gt_logprobs.json).",
    )
    parser.add_argument(
        "--details_path",
        type=str,
        default=None,
        help="Optional path to save per-step scoring details as JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Dojo timeout per theorem in seconds (default: 300).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use state_before from dataset instead of live Dojo interaction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--ret_ckpt_path",
        type=str,
        default=None,
        help="Optional retriever checkpoint (for premise-augmented states).",
    )
    parser.add_argument(
        "--max_theorems",
        type=int,
        default=None,
        help="Limit the number of theorems to process (for debugging).",
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load model ---
    logger.info(f"Loading model from {args.ckpt_path}")
    config = {
        "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
        "lr": 1e-3,
        "warmup_steps": 1000,
        "num_beams": 5,
        "eval_num_retrieved": 10,
        "eval_num_workers": 1,
        "eval_num_gpus": 1,
        "eval_num_theorems": 100,
        "max_inp_seq_len": 512,
        "max_oup_seq_len": 128,
        "ret_ckpt_path": args.ret_ckpt_path,
    }
    model = RetrievalAugmentedGenerator.load(
        args.ckpt_path, device=device, freeze=True, config=config
    )
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # --- Load dataset ---
    logger.info(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path) as f:
        theorems = json.load(f)
    if args.max_theorems:
        theorems = theorems[: args.max_theorems]
    logger.info(f"Loaded {len(theorems)} theorems.")

    # --- Score loop ---
    results: Dict[str, float] = {}
    all_details: Dict[str, List[dict]] = {}
    n_success = 0
    n_skipped = 0

    mode_label = "offline" if args.offline else "Dojo"
    logger.info(f"Starting teacher-forcing scoring ({mode_label} mode)...")

    for thm_data in tqdm(theorems, desc="Scoring"):
        full_name = thm_data.get("full_name", "unknown")

        if args.offline:
            result = score_theorem_offline(model, thm_data)
        else:
            result = score_theorem_dojo(model, thm_data, args.timeout)

        if result is None:
            n_skipped += 1
            continue

        name, cum_lp, details = result
        results[name] = cum_lp
        all_details[name] = details
        n_success += 1

    # --- Export ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"GT log-probs saved to {output_path}")

    if args.details_path:
        details_path = Path(args.details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with open(details_path, "w") as f:
            json.dump(all_details, f, indent=2)
        logger.info(f"Per-step details saved to {details_path}")

    # --- Summary ---
    logger.info(
        f"Done. Scored {n_success} theorems, skipped {n_skipped}."
    )
    if results:
        vals = list(results.values())
        logger.info(
            f"Cumulative log-prob stats:  "
            f"mean={sum(vals)/len(vals):.3f}  "
            f"min={min(vals):.3f}  "
            f"max={max(vals):.3f}"
        )


if __name__ == "__main__":
    main()
