"""
Dataset evaluation via LeanDojo-v2 BestFirstSearchProver.

This script follows framework constraints strictly:
1) Reuses LeanDojo-v2 BestFirstSearchProver (no custom search logic).
2) Uses a Generator-compatible interface (TacticGenerator).
3) Exports search trees via upstream proof_search exporter.
4) Optionally emits realtime event logs by setting analysis_event_dir.
"""

import argparse
import importlib.util
import inspect
import json
import math
import pkgutil
import re
import sys
import types
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lean_dojo_v2.lean_dojo import LeanGitRepo, Pos, Theorem


def _patch_lean_dojo_exports() -> None:
    """
    Some lean_dojo_v2 builds expose tracing symbols only, while BFSP modules expect
    runtime Dojo symbols on `lean_dojo_v2.lean_dojo`. Patch them from `lean_dojo`.
    """
    import lean_dojo_v2.lean_dojo as ld2

    required = [
        "Dojo",
        "DojoCrashError",
        "DojoInitError",
        "DojoTacticTimeoutError",
        "LeanError",
        "ProofFinished",
        "ProofGivenUp",
        "TacticState",
        "LeanGitRepo",
        "Pos",
        "Theorem",
    ]
    missing = [name for name in required if not hasattr(ld2, name)]
    if not missing:
        return

    try:
        import lean_dojo as ld
    except Exception as ex:
        raise RuntimeError(
            "Current lean_dojo_v2 package lacks runtime Dojo symbols needed by "
            "BestFirstSearchProver, and fallback import from `lean_dojo` failed. "
            "Please install a compatible lean_dojo runtime package."
        ) from ex

    resolved = {}
    # First try top-level lean_dojo exports.
    for name in missing:
        if hasattr(ld, name):
            resolved[name] = getattr(ld, name)

    # Then recursively scan submodules for unresolved symbols.
    unresolved = [name for name in missing if name not in resolved]
    if unresolved and hasattr(ld, "__path__"):
        for mod_info in pkgutil.walk_packages(ld.__path__, prefix=f"{ld.__name__}."):
            if not unresolved:
                break
            mod_name = mod_info.name
            try:
                mod = __import__(mod_name, fromlist=["*"])
            except Exception:
                continue
            for name in list(unresolved):
                if hasattr(mod, name):
                    resolved[name] = getattr(mod, name)
                    unresolved.remove(name)

    for name, obj in resolved.items():
        setattr(ld2, name, obj)

    still_missing = [name for name in missing if not hasattr(ld2, name)]
    if still_missing:
        raise RuntimeError(
            "Could not patch required Dojo symbols for BFSP: "
            + ", ".join(still_missing)
        )


def _load_best_first_search_prover():
    """Load BestFirstSearchProver with a fallback for broken prover __init__.py."""
    _patch_lean_dojo_exports()
    try:
        from lean_dojo_v2.lean_agent.prover.proof_search import BestFirstSearchProver

        return BestFirstSearchProver
    except ModuleNotFoundError:
        import lean_dojo_v2

        pkg_root = Path(lean_dojo_v2.__file__).resolve().parent
        prover_dir = pkg_root / "lean_agent" / "prover"
        proof_search_path = prover_dir / "proof_search.py"
        search_tree_path = prover_dir / "search_tree.py"

        # Build a synthetic package to bypass broken prover/__init__.py.
        prover_pkg_name = "lean_dojo_v2.lean_agent.prover"
        if prover_pkg_name not in sys.modules:
            prover_pkg = types.ModuleType(prover_pkg_name)
            prover_pkg.__path__ = [str(prover_dir)]  # type: ignore[attr-defined]
            sys.modules[prover_pkg_name] = prover_pkg

        # Preload search_tree so proof_search can import it without touching __init__.py.
        search_tree_mod_name = "lean_dojo_v2.lean_agent.prover.search_tree"
        if search_tree_mod_name not in sys.modules:
            st_spec = importlib.util.spec_from_file_location(
                search_tree_mod_name, str(search_tree_path)
            )
            if st_spec is None or st_spec.loader is None:
                raise RuntimeError(f"Cannot load search_tree.py from {search_tree_path}")
            st_mod = importlib.util.module_from_spec(st_spec)
            sys.modules[search_tree_mod_name] = st_mod
            st_spec.loader.exec_module(st_mod)

        proof_mod_name = "lean_dojo_v2.lean_agent.prover.proof_search"
        if proof_mod_name not in sys.modules:
            ps_spec = importlib.util.spec_from_file_location(
                proof_mod_name, str(proof_search_path)
            )
            if ps_spec is None or ps_spec.loader is None:
                raise RuntimeError(
                    f"Cannot load proof_search.py from {proof_search_path}"
                )
            ps_mod = importlib.util.module_from_spec(ps_spec)
            sys.modules[proof_mod_name] = ps_mod
            ps_spec.loader.exec_module(ps_mod)

        return sys.modules[proof_mod_name].BestFirstSearchProver


_DECL_RE = re.compile(
    r"^(theorem|lemma|def|example|private\s+theorem|private\s+lemma)\s+"
)
_BY_RE = re.compile(r"\bby\b\s*(.*)", re.DOTALL)


def _clean_tactic(raw: str) -> str:
    tactic = raw.strip().split("\n")[0].split("<;>")[0].strip()
    tactic = re.sub(r"^```\w*\s*", "", tactic)
    tactic = re.sub(r"\s*```$", "", tactic)
    if len(tactic) >= 2 and tactic[0] == tactic[-1] and tactic[0] in "`\"'":
        tactic = tactic[1:-1]
    if _DECL_RE.match(tactic):
        m = _BY_RE.search(tactic)
        if not m:
            return ""
        tactic = m.group(1).strip()
    if tactic.startswith("by "):
        tactic = tactic[3:].strip()
    if tactic in ("", "sorry", "admit"):
        return ""
    return tactic


def _build_prompt(goal_str: str) -> str:
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


def _step_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    tactic: str,
    device: torch.device,
) -> float:
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


class DeepSeekGenerator:
    """Generator adapter compatible with LeanDojo's TacticGenerator interface."""

    def __init__(
        self,
        model_name: str,
        num_return_sequences: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 64,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = _build_prompt(state)
        n = max(num_samples, self.num_return_sequences)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        seen = set()
        pairs: List[Tuple[str, float]] = []
        for text in texts:
            tactic = _clean_tactic(text[len(prompt) :])
            if not tactic or tactic in seen:
                continue
            seen.add(tactic)
            lp = _step_logprob(self.model, self.tokenizer, prompt, tactic, self.device)
            pairs.append((tactic, lp))

        # BestFirstSearchProver expects better tactics first (larger logprob first).
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:num_samples]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, fp, thm, pos, num_samples)
            for s, fp, thm, pos in zip(state, file_path, theorem_full_name, theorem_pos)
        ]


def _parse_pos(item: dict) -> Pos:
    start = item.get("start")
    if isinstance(start, str):
        try:
            return Pos.from_str(start)
        except Exception:
            pass
    if isinstance(start, (list, tuple)) and len(start) == 2:
        return Pos(int(start[0]), int(start[1]))
    if isinstance(start, dict) and "line_nb" in start and "column_nb" in start:
        return Pos(int(start["line_nb"]), int(start["column_nb"]))
    return Pos(1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to benchmark JSON. If omitted, common paths are auto-detected.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-Prover-V2-7B",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max_expansions", type=int, default=256)
    parser.add_argument("--num_sampled_tactics", type=int, default=5)
    parser.add_argument("--max_theorems", type=int, default=50)
    parser.add_argument(
        "--analysis_event_dir",
        type=str,
        default="logs/search_events",
        help="Directory for realtime JSONL events and per-theorem summaries.",
    )
    args = parser.parse_args()

    candidate_paths = []
    if args.dataset_path:
        candidate_paths.append(Path(args.dataset_path))
    candidate_paths.extend(
        [
            Path("data/leandojo_benchmark/random/test_sampled_50.json"),
            Path("data/leandojo_benchmark/random/test.json"),
            Path("data/random/test_sampled_50.json"),
            Path("data/random/test.json"),
        ]
    )

    data_path = None
    for p in candidate_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        tried = "\n".join(f"  - {p}" for p in candidate_paths)
        raise FileNotFoundError(
            "Dataset not found. Tried:\n"
            f"{tried}\n"
            "Please pass --dataset_path explicitly."
        )

    print(f"Using dataset: {data_path}")
    with open(data_path) as f:
        dataset = json.load(f)
    if args.max_theorems > 0:
        dataset = dataset[: args.max_theorems]

    BestFirstSearchProver = _load_best_first_search_prover()

    gen = DeepSeekGenerator(
        model_name=args.model_name,
        num_return_sequences=max(2, args.num_sampled_tactics),
    )
    prover_kwargs = dict(
        tac_gen=gen,
        timeout=args.timeout,
        max_expansions=args.max_expansions,
        num_sampled_tactics=args.num_sampled_tactics,
        debug=False,
    )
    sig = inspect.signature(BestFirstSearchProver.__init__)
    if "analysis_event_dir" in sig.parameters:
        prover_kwargs["analysis_event_dir"] = args.analysis_event_dir
    prover = BestFirstSearchProver(**prover_kwargs)

    stats = {"proved": 0, "failed": 0, "init_error": 0}
    for idx, item in enumerate(dataset, start=1):
        url = item["url"]
        commit = item["commit"]
        file_path = item["file_path"]
        full_name = item["full_name"]
        pos = _parse_pos(item)

        repo = LeanGitRepo(url, commit)
        thm = Theorem(repo=repo, file_path=Path(file_path), full_name=full_name)

        result = prover.search(repo=repo, thm=thm, pos=pos)
        if result is None:
            stats["init_error"] += 1
            print(f"[{idx}/{len(dataset)}] INIT_ERROR {full_name}")
            continue

        if result.status.value == "Proved":
            stats["proved"] += 1
        else:
            stats["failed"] += 1
        print(
            f"[{idx}/{len(dataset)}] {result.status.value:<6} {full_name} "
            f"(expanded={result.num_searched_nodes}, total_nodes={result.num_total_nodes}, "
            f"time={result.total_time:.2f}s)"
        )

    total = sum(stats.values())
    pass_rate = (stats["proved"] / total * 100.0) if total > 0 else 0.0
    print("=" * 72)
    print(
        f"Done. total={total}, proved={stats['proved']}, failed={stats['failed']}, "
        f"init_error={stats['init_error']}, pass@1={pass_rate:.2f}%"
    )
    print("Search trees: logs/search_trees")
    print(f"Realtime events: {args.analysis_event_dir}")


if __name__ == "__main__":
    main()

