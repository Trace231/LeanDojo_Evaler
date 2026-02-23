"""
Dataset evaluation via LeanDojo-v2 BestFirstSearchProver.

This script follows framework constraints strictly:
1) Reuses LeanDojo-v2 BestFirstSearchProver (no custom search logic).
2) Uses a Generator-compatible interface (TacticGenerator).
3) Exports search trees via upstream proof_search exporter.
4) Optionally emits realtime event logs by setting analysis_event_dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import pkgutil
import re
import site
import sys
import types
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LeanGitRepo = None
Pos = None
Theorem = None


def _bootstrap_local_lean_dojo_v2() -> None:
    """
    Prefer current repo source tree over broken site-packages installations.
    """
    repo_root = Path(__file__).resolve().parent
    if not (repo_root / "lean_agent").exists() or not (repo_root / "lean_dojo").exists():
        return
    pkg_name = "lean_dojo_v2"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(repo_root)]  # type: ignore[attr-defined]
    pkg.__file__ = str(repo_root / "__init__.py")
    sys.modules[pkg_name] = pkg


def _load_core_symbols() -> None:
    global LeanGitRepo, Pos, Theorem
    # Always prefer local source tree first when available.
    _bootstrap_local_lean_dojo_v2()
    from lean_dojo_v2.lean_dojo import LeanGitRepo as _LeanGitRepo, Pos as _Pos, Theorem as _Theorem

    LeanGitRepo = _LeanGitRepo
    Pos = _Pos
    Theorem = _Theorem


def _refresh_core_symbols_from_patched_ld2() -> None:
    """After export patching, rebind symbols to the same runtime used by Dojo."""
    global LeanGitRepo, Pos, Theorem
    import lean_dojo_v2.lean_dojo as ld2

    LeanGitRepo = ld2.LeanGitRepo
    Pos = ld2.Pos
    Theorem = ld2.Theorem


def _make_pos_like(pos_data, PosCls):
    if isinstance(pos_data, str):
        try:
            return PosCls.from_str(pos_data)
        except Exception:
            pass
    if isinstance(pos_data, dict) and "line_nb" in pos_data and "column_nb" in pos_data:
        return PosCls(int(pos_data["line_nb"]), int(pos_data["column_nb"]))
    if isinstance(pos_data, (list, tuple)) and len(pos_data) == 2:
        return PosCls(int(pos_data[0]), int(pos_data[1]))
    return PosCls(1, 1)


def _build_theorem_like(item: dict, LeanGitRepoCls, PosCls, TheoremCls):
    repo = LeanGitRepoCls(item["url"], item["commit"])
    file_path = Path(item["file_path"])
    full_name = item["full_name"]
    start = _make_pos_like(item.get("start"), PosCls)
    end = _make_pos_like(item.get("end"), PosCls)

    sig = inspect.signature(TheoremCls)
    kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name == "repo":
            kwargs[name] = repo
        elif name == "file_path":
            kwargs[name] = file_path
        elif name == "full_name":
            kwargs[name] = full_name
        elif name == "start":
            kwargs[name] = start
        elif name == "end":
            kwargs[name] = end
        elif name == "url":
            kwargs[name] = item["url"]
        elif name == "commit":
            kwargs[name] = item["commit"]
        elif name == "theorem_statement":
            kwargs[name] = item.get("theorem_statement")
    return TheoremCls(**kwargs), repo, start


def _cache_root_dir() -> Path:
    try:
        from lean_dojo_v2.utils.constants import CACHE_DIR

        return Path(CACHE_DIR)
    except Exception:
        return Path.home() / ".cache" / "lean_dojo"


def _repair_repo_cache_layout(repo) -> bool:
    """
    Fix common remote-cache layout mismatch:
    expected: <cache>/<dirname>/<repo.name>_d
    got:      <cache>/<dirname>/<repo.name>
    """
    try:
        dirname = repo.get_cache_dirname()
    except Exception:
        return False

    root = _cache_root_dir()
    base = root / dirname
    normal = base / repo.name
    with_deps = base / f"{repo.name}_d"

    if not base.exists():
        return False
    if not normal.exists() and not with_deps.exists():
        return False

    import shutil

    # If _d path exists as a symlink, replace it with a real directory copy.
    if with_deps.is_symlink():
        target = with_deps.resolve()
        with_deps.unlink()
        shutil.copytree(target, with_deps)
        print(f"[cache-fix] replaced symlink with real dir copy: {with_deps}")
        return True

    # If _d path already exists as a real directory, leave it unchanged.
    if with_deps.exists():
        return False

    # Create a real directory copy instead of symlink to avoid subpath checks failing.
    shutil.copytree(normal, with_deps)
    print(f"[cache-fix] copied dir: {normal} -> {with_deps}")
    return True


def _import_site_package(pkg_name: str):
    """Load a package directly from site-packages to avoid local shadowing."""
    for base in site.getsitepackages():
        pkg_dir = Path(base) / pkg_name
        init_py = pkg_dir / "__init__.py"
        if not init_py.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(init_py),
            submodule_search_locations=[str(pkg_dir)],
        )
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        # Ensure package-relative imports (e.g. from .foo import bar) resolve.
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)
        return mod
    return None


def _patch_lean_dojo_exports() -> None:
    def _is_valid_symbol(name: str, obj) -> bool:
        if obj is None:
            return False
        if name == "Dojo":
            return callable(obj)
        if name in {
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
        }:
            return isinstance(obj, type)
        return True

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
    missing = [
        name
        for name in required
        if not hasattr(ld2, name) or not _is_valid_symbol(name, getattr(ld2, name))
    ]
    if not missing:
        return

    candidate_pkgs = []
    import lean_dojo_v2 as ld2_root
    ld = None
    try:
        import lean_dojo as ld

        repo_root = Path(__file__).resolve().parent
        ld_file = Path(getattr(ld, "__file__", "")).resolve()
        if str(ld_file).startswith(str(repo_root)):
            # Local repository's lean_dojo package shadows installed runtime package.
            ld = _import_site_package("lean_dojo")
        if ld is not None:
            # Prefer lean_dojo runtime symbols when available.
            candidate_pkgs.append(ld)
    except Exception:
        ld = _import_site_package("lean_dojo")
        if ld is not None:
            candidate_pkgs.append(ld)
    candidate_pkgs.append(ld2_root)

    resolved = {}

    # 1) Check top-level exports of candidate packages.
    for pkg in candidate_pkgs:
        for name in missing:
            if name not in resolved and hasattr(pkg, name):
                cand = getattr(pkg, name)
                if _is_valid_symbol(name, cand):
                    resolved[name] = cand

    # 2) Recursively scan submodules for unresolved symbols.
    unresolved = [name for name in missing if name not in resolved]
    for pkg in candidate_pkgs:
        if not unresolved:
            break
        if not hasattr(pkg, "__path__"):
            continue
        for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
            if not unresolved:
                break
            mod_name = mod_info.name
            try:
                mod = __import__(mod_name, fromlist=["*"])
            except Exception:
                continue
            for name in list(unresolved):
                if hasattr(mod, name):
                    cand = getattr(mod, name)
                    if _is_valid_symbol(name, cand):
                        resolved[name] = cand
                        unresolved.remove(name)

    for name, obj in resolved.items():
        if _is_valid_symbol(name, obj):
            setattr(ld2, name, obj)

    # Enforce runtime consistency: if Dojo comes from lean_dojo, use the same
    # package's LeanGitRepo/Pos/Theorem to avoid cross-package type mismatch.
    if ld is not None:
        for sym in ("LeanGitRepo", "Pos", "Theorem"):
            if hasattr(ld, sym):
                obj = getattr(ld, sym)
                if _is_valid_symbol(sym, obj):
                    setattr(ld2, sym, obj)

    still_missing = [
        name
        for name in missing
        if not hasattr(ld2, name) or not _is_valid_symbol(name, getattr(ld2, name))
    ]
    if still_missing:
        # Version-compatibility aliases for known renamed/removed symbols.
        alias_candidates = {
            "DojoCrashError": ["DojoHardTimeoutError", "DojoTimeoutError"],
            "DojoInitError": ["DojoError"],
            "DojoTacticTimeoutError": ["DojoHardTimeoutError", "DojoTimeoutError"],
            "ProofGivenUp": ["ProofGivenUpError"],
        }
        for target in list(still_missing):
            for alias in alias_candidates.get(target, []):
                if hasattr(ld2, alias):
                    alias_obj = getattr(ld2, alias)
                    if _is_valid_symbol(target, alias_obj):
                        setattr(ld2, target, alias_obj)
                        break
            if hasattr(ld2, target) and _is_valid_symbol(target, getattr(ld2, target)):
                still_missing.remove(target)

    if still_missing:
        # Last-resort placeholders so BFSP modules can import. This keeps runtime
        # compatible across partially broken package versions.
        if "DojoCrashError" in still_missing and not hasattr(ld2, "DojoCrashError"):
            class DojoCrashError(Exception):
                pass
            setattr(ld2, "DojoCrashError", DojoCrashError)
            still_missing.remove("DojoCrashError")

        if "DojoInitError" in still_missing and not hasattr(ld2, "DojoInitError"):
            class DojoInitError(Exception):
                pass
            setattr(ld2, "DojoInitError", DojoInitError)
            still_missing.remove("DojoInitError")

        if (
            "DojoTacticTimeoutError" in still_missing
            and not hasattr(ld2, "DojoTacticTimeoutError")
        ):
            class DojoTacticTimeoutError(Exception):
                pass
            setattr(ld2, "DojoTacticTimeoutError", DojoTacticTimeoutError)
            still_missing.remove("DojoTacticTimeoutError")

        if "ProofGivenUp" in still_missing and not hasattr(ld2, "ProofGivenUp"):
            class ProofGivenUp:
                pass
            setattr(ld2, "ProofGivenUp", ProofGivenUp)
            still_missing.remove("ProofGivenUp")

    still_missing = [
        name
        for name in missing
        if not hasattr(ld2, name) or not _is_valid_symbol(name, getattr(ld2, name))
    ]
    if still_missing:
        pkg_names = [pkg.__name__ for pkg in candidate_pkgs]
        raise RuntimeError(
            "Could not patch required Dojo symbols for BFSP: "
            + ", ".join(still_missing)
            + ". Scanned packages: "
            + ", ".join(pkg_names)
        )


def _load_best_first_search_prover():
    """Load BestFirstSearchProver with a fallback for broken prover __init__.py."""
    _patch_lean_dojo_exports()
    _refresh_core_symbols_from_patched_ld2()
    try:
        from lean_dojo_v2.lean_agent.prover.proof_search import BestFirstSearchProver

        return BestFirstSearchProver
    except ModuleNotFoundError:
        import lean_dojo_v2

        pkg_root = Path(lean_dojo_v2.__file__).resolve().parent
        lean_agent_dir = pkg_root / "lean_agent"
        generator_dir = lean_agent_dir / "generator"
        prover_dir = pkg_root / "lean_agent" / "prover"
        proof_search_path = prover_dir / "proof_search.py"
        search_tree_path = prover_dir / "search_tree.py"

        # Build synthetic parent packages to avoid importing broken __init__.py files.
        lean_agent_pkg_name = "lean_dojo_v2.lean_agent"
        lean_agent_pkg = types.ModuleType(lean_agent_pkg_name)
        lean_agent_pkg.__path__ = [str(lean_agent_dir)]  # type: ignore[attr-defined]
        sys.modules[lean_agent_pkg_name] = lean_agent_pkg

        generator_pkg_name = "lean_dojo_v2.lean_agent.generator"
        generator_pkg = types.ModuleType(generator_pkg_name)
        generator_pkg.__path__ = [str(generator_dir)]  # type: ignore[attr-defined]
        sys.modules[generator_pkg_name] = generator_pkg

        # Build a synthetic package to bypass broken prover/__init__.py.
        prover_pkg_name = "lean_dojo_v2.lean_agent.prover"
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
_COMMENT_PREFIX_RE = re.compile(r"^\s*--")


def _clean_tactic(raw: str) -> str:
    tactic = raw.strip().split("\n")[0].strip()
    if _COMMENT_PREFIX_RE.match(tactic):
        return ""
    tactic = re.sub(r"^```\w*\s*", "", tactic)
    tactic = re.sub(r"\s*```$", "", tactic)
    tactic = re.sub(r"\s*--.*$", "", tactic).strip()
    if len(tactic) >= 2 and tactic[0] == tactic[-1] and tactic[0] in "`\"'":
        tactic = tactic[1:-1]
    if _DECL_RE.match(tactic):
        m = _BY_RE.search(tactic)
        if not m:
            return ""
        tactic = m.group(1).strip()
    if tactic.startswith("by "):
        tactic = tactic[3:].strip()
    if tactic.endswith(" by"):
        tactic = tactic[:-3].strip()
    if tactic in ("", "sorry", "admit"):
        return ""
    if len(tactic) > 220:
        return ""
    return tactic


def _build_prompt(
    goal_str: str,
    theorem_full_name: str,
    file_path: str,
    tactic_history: List[str] | None = None,
) -> str:
    history_block = ""
    if tactic_history:
        clipped = tactic_history[-12:]
        history_lines = "\n".join(f"- {t}" for t in clipped)
        history_block = (
            "Context:\n"
            f"- theorem: {theorem_full_name}\n"
            f"- file: {file_path}\n"
            "- previous tactics:\n"
            f"{history_lines}\n\n"
        )

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
        "- Prefer short executable tactics over long guessed expressions.\n"
        "- Avoid inventing constants/lemmas that may be out of scope.\n"
        "### User:\n"
        f"{history_block}"
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
        dtype: str = "bf16",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.05,
        use_fallback_tactics: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Use fp32/fp16/bf16.")
        self.dtype = dtype
        torch_dtype = dtype_map[dtype]
        if self.device.type != "cuda":
            # Keep CPU path stable.
            torch_dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.use_fallback_tactics = use_fallback_tactics
        self.fallback_tactics = [
            "simp",
            "aesop",
            "simp_all",
            "rfl",
            "assumption",
            "tauto",
        ]

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
        tactic_history: List[str] | None = None,
    ) -> List[Tuple[str, float]]:
        prompt = _build_prompt(
            goal_str=state,
            theorem_full_name=theorem_full_name,
            file_path=file_path,
            tactic_history=tactic_history,
        )
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
                repetition_penalty=self.repetition_penalty,
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

        # Keep BFS alive when model outputs collapse to unusable tactics.
        if self.use_fallback_tactics:
            for tac in self.fallback_tactics:
                if tac in seen:
                    continue
                seen.add(tac)
                lp = _step_logprob(self.model, self.tokenizer, prompt, tac, self.device)
                pairs.append((tac, lp - 0.15))

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
    return _make_pos_like(item.get("start"), Pos)


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
    parser.add_argument("--num_sampled_tactics", type=int, default=8)
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=0,
        help="Raw number of model samples before dedup/filter. "
        "0 means auto (max(8, 2 * num_sampled_tactics)).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--disable_fallback_tactics",
        action="store_true",
        help="Disable built-in safe fallback tactics (simp/aesop/etc.).",
    )
    parser.add_argument("--max_theorems", type=int, default=50)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Model weight dtype. bf16 is recommended on A100.",
    )
    parser.add_argument(
        "--analysis_event_dir",
        type=str,
        default="logs/search_events",
        help="Directory for realtime JSONL events and per-theorem summaries.",
    )
    args = parser.parse_args()
    _load_core_symbols()

    candidate_paths = []
    if args.dataset_path:
        explicit = Path(args.dataset_path)
        if not explicit.exists():
            raise FileNotFoundError(
                f"--dataset_path not found: {explicit}\n"
                "Please pass a valid absolute path or run from the expected project root."
            )
        candidate_paths.append(explicit)
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
        num_return_sequences=(
            args.num_return_sequences
            if args.num_return_sequences > 0
            else max(8, args.num_sampled_tactics * 2)
        ),
        dtype=args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        use_fallback_tactics=not args.disable_fallback_tactics,
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
        full_name = item["full_name"]
        thm, repo, pos = _build_theorem_like(item, LeanGitRepo, Pos, Theorem)

        repaired_and_retried = False
        try:
            result = prover.search(repo=repo, thm=thm, pos=pos)
        except AssertionError as ex:
            if _repair_repo_cache_layout(repo):
                repaired_and_retried = True
                try:
                    result = prover.search(repo=repo, thm=thm, pos=pos)
                except Exception as ex2:
                    stats["init_error"] += 1
                    print(
                        f"[{idx}/{len(dataset)}] INIT_ERROR {full_name} "
                        f"(Dojo assertion after cache-fix: {type(ex2).__name__}: {ex2})"
                    )
                    continue
            else:
                stats["init_error"] += 1
                print(
                    f"[{idx}/{len(dataset)}] INIT_ERROR {full_name} "
                    f"(Dojo assertion: {ex})"
                )
                continue
        except Exception as ex:
            stats["init_error"] += 1
            print(f"[{idx}/{len(dataset)}] INIT_ERROR {full_name} ({type(ex).__name__}: {ex})")
            continue
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
            f"time={result.total_time:.2f}s"
            + (", cache_fixed=1" if repaired_and_retried else "")
            + ")"
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

