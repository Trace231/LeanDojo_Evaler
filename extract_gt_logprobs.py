"""
extract_gt_logprobs.py
======================
Teacher-Forcing scoring: compute Ground Truth cumulative log-probability
of expert tactic sequences.

Supports two model backends:
  - **Causal LM** (decoder-only): DeepSeek-Prover-V2, LLaMA, etc.
  - **Seq2Seq**  (encoder-decoder): ReProver / ByT5-based models

Usage
-----
# DeepSeek-Prover-V2 (causal LM, offline, no Lean required)
python extract_gt_logprobs.py \\
    --dataset_path data/leandojo_benchmark/random/test.json \\
    --ckpt_path "deepseek-ai/DeepSeek-Prover-V2-7B" \\
    --output_path logs/gt_logprobs.json \\
    --details_path logs/gt_details.json \\
    --offline

# ReProver checkpoint (seq2seq T5, offline)
python extract_gt_logprobs.py \\
    --dataset_path data/random/test.json \\
    --ckpt_path checkpoints/reprover.ckpt \\
    --model_type seq2seq \\
    --output_path logs/gt_logprobs.json \\
    --offline

# Dojo mode (live Lean interaction, any model)
python extract_gt_logprobs.py \\
    --dataset_path data/random/test.json \\
    --ckpt_path "deepseek-ai/DeepSeek-Prover-V2-7B" \\
    --output_path logs/gt_logprobs.json \\
    --timeout 300
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Utility: inline format_state so we don't depend on lean_dojo_v2 imports
# ---------------------------------------------------------------------------

def format_state(s: str) -> str:
    """Strip the 'N goals' prefix that LeanDojo prepends to tactic states."""
    m = re.match(r"\d+ goals?\n", s)
    if m is not None:
        return s[m.end():].strip()
    return s


# ---------------------------------------------------------------------------
# Lazy Dojo imports (only when --offline is NOT set)
# ---------------------------------------------------------------------------

_dojo_imports_loaded = False
Dojo = None
TacticState = None
ProofFinished = None
LeanError = None
LeanGitRepo = None
Pos = None
Theorem = None


def _load_dojo_imports():
    """Try importing Dojo types from lean_dojo (upstream) or lean_dojo_v2."""
    global _dojo_imports_loaded
    global Dojo, TacticState, ProofFinished, LeanError
    global LeanGitRepo, Pos, Theorem

    if _dojo_imports_loaded:
        return

    # Try the original lean_dojo package first
    try:
        from lean_dojo import (
            Dojo as _Dojo,
            TacticState as _TS,
            ProofFinished as _PF,
            LeanError as _LE,
            LeanGitRepo as _LGR,
            Pos as _Pos,
            Theorem as _Thm,
        )
        Dojo, TacticState, ProofFinished = _Dojo, _TS, _PF
        LeanError, LeanGitRepo, Pos, Theorem = _LE, _LGR, _Pos, _Thm
        _dojo_imports_loaded = True
        logger.info("Dojo imported from 'lean_dojo'")
        return
    except ImportError:
        pass

    # Fallback: lean_dojo_v2.lean_dojo
    try:
        from lean_dojo_v2.lean_dojo import (
            Dojo as _Dojo,
            TacticState as _TS,
            ProofFinished as _PF,
            LeanError as _LE,
            LeanGitRepo as _LGR,
            Pos as _Pos,
            Theorem as _Thm,
        )
        Dojo, TacticState, ProofFinished = _Dojo, _TS, _PF
        LeanError, LeanGitRepo, Pos, Theorem = _LE, _LGR, _Pos, _Thm
        _dojo_imports_loaded = True
        logger.info("Dojo imported from 'lean_dojo_v2.lean_dojo'")
        return
    except ImportError:
        pass

    logger.error(
        "Cannot import Dojo. Install lean_dojo or use --offline mode."
    )
    sys.exit(1)


# ===================================================================
# Model Backends
# ===================================================================


class CausalLMScorer:
    """Teacher-forcing scorer for decoder-only models (DeepSeek, LLaMA, …)."""

    def __init__(self, model_name_or_path: str, device: torch.device,
                 dtype: torch.dtype = torch.float16,
                 prompt_template: Optional[str] = None):
        logger.info(f"Loading causal LM: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, trust_remote_code=True,
            device_map=device if str(device) != "cpu" else None,
        )
        if str(device) == "cpu" or self.model.device == torch.device("cpu"):
            self.model = self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prompt template: {state} and {tactic} are placeholders
        self.prompt_template = prompt_template or (
            "Complete the following Lean 4 tactic proof.\n\n"
            "State:\n{state}\n\n"
            "Tactic:\n{tactic}"
        )
        # The "context" part is everything before {tactic}
        self._context_template = self.prompt_template.split("{tactic}")[0]

        logger.info(f"CausalLMScorer ready on {self.device}")

    @torch.no_grad()
    def score(self, state_text: str, tactic_text: str) -> float:
        """Return sum of log P(tactic_token_i | context) — the step_logprob."""
        context = self._context_template.format(state=state_text)
        full_text = self.prompt_template.format(
            state=state_text, tactic=tactic_text,
        )

        # Tokenize context and full text separately to find the split point
        ctx_ids = self.tokenizer.encode(context, add_special_tokens=True)
        full_enc = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True,
        )
        full_ids = full_enc.input_ids.to(self.device)  # (1, L)

        n_ctx = len(ctx_ids)  # number of context tokens (state + prompt)
        n_full = full_ids.shape[1]

        if n_ctx >= n_full:
            # Tactic produced no extra tokens (degenerate case)
            return 0.0

        logits = self.model(full_ids).logits  # (1, L, V)

        # Causal LM: logits[t] predicts token[t+1]
        # We want log P(token[n_ctx], token[n_ctx+1], …, token[n_full-1])
        # In shifted space: positions (n_ctx - 1) .. (n_full - 2)
        shift_logits = logits[:, :-1, :]              # (1, L-1, V)
        shift_labels = full_ids[:, 1:]                 # (1, L-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(
            2, shift_labels.unsqueeze(2)
        ).squeeze(2)  # (1, L-1)

        # Sum only over tactic tokens
        tactic_lps = token_lps[:, n_ctx - 1:]
        return tactic_lps.sum().item()


class Seq2SeqScorer:
    """Teacher-forcing scorer for encoder-decoder models (ReProver / ByT5)."""

    def __init__(self, ckpt_path: str, device: torch.device,
                 ret_ckpt_path: Optional[str] = None):
        logger.info(f"Loading seq2seq model from checkpoint: {ckpt_path}")
        try:
            from lean_dojo_v2.lean_agent.generator.model import (
                RetrievalAugmentedGenerator,
            )
        except ImportError:
            logger.error(
                "Cannot import RetrievalAugmentedGenerator. "
                "Make sure lean_dojo_v2 is installed for seq2seq mode."
            )
            sys.exit(1)

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
            "ret_ckpt_path": ret_ckpt_path,
        }
        self.model = RetrievalAugmentedGenerator.load(
            ckpt_path, device=device, freeze=True, config=config,
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.device = device
        logger.info("Seq2SeqScorer ready.")

    @torch.no_grad()
    def score(self, state_text: str, tactic_text: str) -> float:
        """Return sum of log P(tactic_token_i | state)."""
        enc = self.tokenizer(
            state_text, padding=False,
            max_length=self.model.max_inp_seq_len,
            truncation=True, return_tensors="pt",
        )
        state_ids = enc.input_ids.to(self.device)
        state_mask = enc.attention_mask.to(self.device)

        dec = self.tokenizer(
            tactic_text, padding=False,
            max_length=self.model.max_oup_seq_len,
            truncation=True, return_tensors="pt",
        )
        label_ids = dec.input_ids.to(self.device)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        output = self.model.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=label_ids,
        )
        logits = output.logits
        log_probs = F.log_softmax(logits, dim=-1)

        gather_ids = label_ids.clone()
        gather_ids[gather_ids == -100] = 0
        token_lps = log_probs.gather(
            2, gather_ids.unsqueeze(2)
        ).squeeze(2)

        mask = (label_ids != -100).float()
        return (token_lps * mask).sum().item()


# ===================================================================
# Scoring drivers (offline / Dojo)
# ===================================================================


def score_theorem_offline(
    scorer, thm_data: dict,
) -> Optional[Tuple[str, float, List[dict]]]:
    """Score using pre-recorded state_before from the dataset."""
    full_name = thm_data["full_name"]
    traced_tactics = thm_data.get("traced_tactics", [])
    if not traced_tactics:
        return None

    tactics_to_score = [
        t for t in traced_tactics
        if t.get("state_before", "") not in ("", "no goals")
        and "·" not in t.get("tactic", "")
    ]
    if not tactics_to_score:
        return None

    cumulative_logprob = 0.0
    step_details = []

    for i, tac_data in enumerate(tactics_to_score):
        tactic_str = tac_data["tactic"]
        state_text = format_state(tac_data["state_before"])

        try:
            step_lp = scorer.score(state_text, tactic_str)
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


def score_theorem_dojo(
    scorer, thm_data: dict, timeout: int,
) -> Optional[Tuple[str, float, List[dict]]]:
    """Score using Dojo for live Lean state tracking."""
    _load_dojo_imports()

    full_name = thm_data["full_name"]
    traced_tactics = thm_data.get("traced_tactics", [])
    if not traced_tactics:
        logger.warning(f"[{full_name}] No traced tactics, skipping.")
        return None

    tactics_to_score = [
        t for t in traced_tactics
        if t.get("state_before", "") not in ("", "no goals")
        and "·" not in t.get("tactic", "")
    ]
    if not tactics_to_score:
        return None

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

                if hasattr(state, "pp"):
                    state_text = format_state(state.pp)
                else:
                    state_text = format_state(str(state))

                step_lp = scorer.score(state_text, tactic_str)
                cumulative_logprob += step_lp

                step_details.append({
                    "step": i,
                    "tactic": tactic_str,
                    "step_logprob": step_lp,
                    "cumulative_logprob": cumulative_logprob,
                })

                response = dojo.run_tac(state, tactic_str)

                if isinstance(response, ProofFinished):
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
                        f"[{full_name}] Unexpected response at step {i}: "
                        f"{type(response)}"
                    )
                    break

    except Exception as e:
        logger.warning(f"[{full_name}] Dojo error: {e}")
        return None

    return full_name, cumulative_logprob, step_details


# ===================================================================
# Model type detection
# ===================================================================


def detect_model_type(ckpt_path: str) -> str:
    """Heuristic: .ckpt → seq2seq (ReProver);  HF model ID → causal_lm."""
    if ckpt_path.endswith(".ckpt"):
        return "seq2seq"
    return "causal_lm"


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compute GT cumulative log-probs via teacher forcing.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to test.json (LeanDojo export with traced_tactics).",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="HuggingFace model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--model_type", type=str, default="auto",
        choices=["auto", "causal_lm", "seq2seq"],
        help="Model architecture (default: auto-detect).",
    )
    parser.add_argument(
        "--output_path", type=str, default="logs/gt_logprobs.json",
    )
    parser.add_argument(
        "--details_path", type=str, default=None,
        help="Optional path for per-step scoring details.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--offline", action="store_true",
        help="Use state_before from dataset (no Lean/Dojo required).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Weight dtype for causal LM (default: float16).",
    )
    parser.add_argument(
        "--ret_ckpt_path", type=str, default=None,
        help="Retriever checkpoint (seq2seq mode only).",
    )
    parser.add_argument(
        "--prompt_template", type=str, default=None,
        help="Custom prompt template with {state} and {tactic} placeholders.",
    )
    parser.add_argument("--max_theorems", type=int, default=None)
    args = parser.parse_args()

    # --- Device & dtype ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # --- Detect model type ---
    model_type = args.model_type
    if model_type == "auto":
        model_type = detect_model_type(args.ckpt_path)
    logger.info(f"Model type: {model_type} | Device: {device}")

    # --- Build scorer ---
    if model_type == "causal_lm":
        scorer = CausalLMScorer(
            args.ckpt_path, device,
            dtype=dtype_map[args.dtype],
            prompt_template=args.prompt_template,
        )
    else:
        scorer = Seq2SeqScorer(
            args.ckpt_path, device,
            ret_ckpt_path=args.ret_ckpt_path,
        )

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
    logger.info(f"Scoring mode: {mode_label}")

    for thm_data in tqdm(theorems, desc="Scoring"):
        if args.offline:
            result = score_theorem_offline(scorer, thm_data)
        else:
            result = score_theorem_dojo(scorer, thm_data, args.timeout)

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
    logger.info(f"GT log-probs saved → {output_path}")

    if args.details_path:
        dp = Path(args.details_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "w") as f:
            json.dump(all_details, f, indent=2)
        logger.info(f"Per-step details saved → {dp}")

    # --- Summary ---
    logger.info(f"Done. Scored {n_success}, skipped {n_skipped}.")
    if results:
        vals = list(results.values())
        logger.info(
            f"Stats:  mean={sum(vals)/len(vals):.3f}  "
            f"min={min(vals):.3f}  max={max(vals):.3f}"
        )


if __name__ == "__main__":
    main()
