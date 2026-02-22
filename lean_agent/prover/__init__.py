"""
Prover module for LeanAgent.

This module provides theorem proving functionality.
"""

from .proof_search import BestFirstSearchProver, DistributedProver, SearchResult, Status

try:
    from .evaluate import evaluate
except ModuleNotFoundError:
    # Some distributions do not ship evaluate.py; keep core prover imports usable.
    evaluate = None

__all__ = [
    "evaluate",
    "DistributedProver",
    "SearchResult",
    "Status",
    "BestFirstSearchProver",
]
