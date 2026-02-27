"""
mambank/retrieval/query.py

QueryEngine — multi-level HGNS coarse-to-fine retrieval.

Retrieval Strategy
------------------
Mirrors the HGNS base-k traversal: start at the coarsest level,
narrow candidates, refine only where needed.

    Step 1 — Level 2 (coarse): search all entries, return top candidate_k.
    Step 2 — Level 1 (mid):    re-rank the candidate_k using level-1 sims.
    Step 3 — Level 0 (fine):   final re-ranking of top_k using full vectors.

This is O(n) at L2, O(candidate_k) at L1 and L0 — very fast when n is
large because the expensive full-vector comparison only touches a small
candidate set.

The engine also supports:
  - Single-level search (bypass the drill-down, e.g. for benchmarking).
  - Score threshold filtering (discard low-confidence results).
  - Result deduplication by ptr_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mambank.retrieval.index import VectorIndex, SearchResult
from mambank.core.registry import Registry
from mambank.core.pointer import PointerRecord
from mambank.hgns.hierarchy import HGNSHierarchy


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class RecallResult:
    """
    A single recall result: the full PointerRecord at each HGNS level,
    plus the similarity scores used to rank it.

    Attributes
    ----------
    ptr_level2 : PointerRecord
        The Level-2 (coarse) pointer — used for initial candidate selection.
    ptr_level1 : Optional[PointerRecord]
        The Level-1 (mid) pointer — used for refinement. None if level-1
        search was skipped.
    ptr_level0 : Optional[PointerRecord]
        The Level-0 (full) pointer — highest fidelity. None if level-0
        search was skipped.
    score_l2 : float
        Cosine similarity at Level-2 (coarse).
    score_l1 : float
        Cosine similarity at Level-1 (None → not computed).
    score_l0 : float
        Cosine similarity at Level-0 (None → not computed).
    final_score : float
        The score used for final ranking (highest available level).
    source_text_hash : str
        Hash of the original text chunk (for display / dedup).
    rank : int
        Final rank in the result list (0 = best match).
    """
    ptr_level2: PointerRecord
    ptr_level1: Optional[PointerRecord]
    ptr_level0: Optional[PointerRecord]
    score_l2: float
    score_l1: Optional[float]
    score_l0: Optional[float]
    final_score: float
    source_text_hash: str
    rank: int = 0

    @property
    def best_ptr(self) -> PointerRecord:
        """Return the finest-resolution pointer available."""
        return self.ptr_level0 or self.ptr_level1 or self.ptr_level2

    def __repr__(self) -> str:
        return (
            f"RecallResult(rank={self.rank}, score={self.final_score:.4f}, "
            f"ptr={self.best_ptr.ptr_id[:12]}...)"
        )


# ------------------------------------------------------------------
# QueryEngine
# ------------------------------------------------------------------

class QueryEngine:
    """
    Multi-level HGNS retrieval engine.

    Manages three VectorIndex instances (one per HGNS level) and
    orchestrates the coarse-to-fine drill-down query strategy.

    Parameters
    ----------
    hierarchy : HGNSHierarchy
        Used to compress query embeddings for each level.
    registry : Registry
        Used to look up PointerRecords from ptr_ids returned by the index.
    dims : dict
        {"level0": int, "level1": int, "level2": int} — embedding dimensions.
        Obtained from HGNSHierarchy.dims_for(full_dim).
    candidate_multiplier : int
        Level-2 returns top_k * candidate_multiplier candidates for
        refinement at Level-1. Default 5.
        e.g. top_k=3, multiplier=5 → L2 returns 15 candidates → L1 re-ranks
        → top 3 go to L0.
    min_score : float
        Results with final_score below this threshold are discarded.
        Default 0.0 (no filtering).
    """

    def __init__(
        self,
        hierarchy: HGNSHierarchy,
        registry: Registry,
        dims: Dict[str, int],
        candidate_multiplier: int = 5,
        min_score: float = 0.0,
    ):
        self.hierarchy = hierarchy
        self.registry = registry
        self.candidate_multiplier = candidate_multiplier
        self.min_score = min_score

        # One index per HGNS level
        self.indexes: Dict[int, VectorIndex] = {
            0: VectorIndex(dim=dims["level0"], hgns_level=0),
            1: VectorIndex(dim=dims["level1"], hgns_level=1),
            2: VectorIndex(dim=dims["level2"], hgns_level=2),
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(
        self,
        ptr_id_l0: str,
        ptr_id_l1: str,
        ptr_id_l2: str,
        emb_l0: np.ndarray,
        emb_l1: np.ndarray,
        emb_l2: np.ndarray,
    ) -> None:
        """
        Add a new activation at all three levels to the search indexes.

        Called by MemBank.ingest() after writing to the buffer.

        Parameters
        ----------
        ptr_id_l{0,1,2} : str
            Content-addressed pointer IDs for each level.
        emb_l{0,1,2} : np.ndarray
            Activation embeddings at each HGNS level.
        """
        self.indexes[0].add(emb_l0, ptr_id_l0)
        self.indexes[1].add(emb_l1, ptr_id_l1)
        self.indexes[2].add(emb_l2, ptr_id_l2)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        levels: Optional[List[int]] = None,
    ) -> List[RecallResult]:
        """
        Run a multi-level HGNS coarse-to-fine query.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (full_dim,)
            Mean-pooled hidden state of the query text.
        top_k : int
            Number of final results to return.
        levels : list of int, optional
            Which levels to include in the drill-down.
            Default [2, 1, 0] (full coarse-to-fine).
            Use [2] for coarse-only (fastest).
            Use [0] for exact full-dim search.

        Returns
        -------
        List[RecallResult]
            Ranked by final_score descending. Length <= top_k.
        """
        if levels is None:
            levels = [2, 1, 0]

        # Compress query to each requested level
        ha = self.hierarchy.compress(query_embedding, source_text="__query__", model_id="__query__")
        query_by_level = {0: ha.level0, 1: ha.level1, 2: ha.level2}

        # --- Level 2: coarse initial search ---
        if 2 not in levels:
            # Direct Level 0 or 1 search — skip L2
            start_level = min(levels)
            candidate_k = top_k
        else:
            start_level = 2
            # Cast wider net: more candidates for refinement
            candidate_k = top_k * self.candidate_multiplier

        # Initial search at starting level
        raw_results = self.indexes[start_level].search(
            query_by_level[start_level], top_k=candidate_k
        )

        if not raw_results:
            return []

        # Collect candidate ptr_ids and build initial score map
        # key: source_text_hash (groups pointers from same text across levels)
        # value: best score seen so far for this source
        # We use source_text_hash to link L2/L1/L0 pointers for the same chunk

        # Step: resolve L2 ptr_ids → PointerRecords → group by source_text_hash
        candidates: Dict[str, dict] = {}  # source_text_hash → data
        for r in raw_results:
            ptr = self.registry.get(r.ptr_id)
            if ptr is None or not ptr.is_alive:
                continue
            sth = ptr.source_text_hash
            if sth not in candidates:
                candidates[sth] = {
                    "ptr_l2": ptr if start_level == 2 else None,
                    "ptr_l1": ptr if start_level == 1 else None,
                    "ptr_l0": ptr if start_level == 0 else None,
                    "score_l2": r.score if start_level == 2 else None,
                    "score_l1": r.score if start_level == 1 else None,
                    "score_l0": r.score if start_level == 0 else None,
                }
            else:
                # Keep best score per level
                key = f"score_l{start_level}"
                if candidates[sth][key] is None or r.score > candidates[sth][key]:
                    candidates[sth][key] = r.score
                    candidates[sth][f"ptr_l{start_level}"] = ptr

        if not candidates:
            return []

        # --- Level 1 refinement ---
        if 1 in levels and start_level == 2:
            # Find Level-1 pointers for the same source_text_hashes
            for sth, data in candidates.items():
                ptrs_l1 = self._find_ptr_for_source_at_level(sth, level=1)
                if ptrs_l1:
                    ptr_l1 = ptrs_l1[0]
                    emb_l1 = None
                    # Score against Level-1 query
                    from mambank.core.buffer import MemMapBuffer  # Avoid circular at module level
                    # We don't have the buffer here — score via ptr metadata only
                    # Full buffer dereference happens in MemBank.recall()
                    # Here we use the registry's stored embedding via a score proxy:
                    # re-search L1 index for this specific ptr
                    score_l1 = self._score_ptr_against_query(
                        ptr_l1, query_by_level[1], level=1
                    )
                    data["ptr_l1"] = ptr_l1
                    data["score_l1"] = score_l1

        # --- Level 0 refinement ---
        if 0 in levels and len(levels) > 1:
            # Sort by best score so far, take top_k for L0 refinement
            sorted_candidates = sorted(
                candidates.items(),
                key=lambda x: max(
                    v for v in [x[1]["score_l2"], x[1]["score_l1"]]
                    if v is not None
                ),
                reverse=True,
            )[:top_k]

            for sth, data in sorted_candidates:
                ptrs_l0 = self._find_ptr_for_source_at_level(sth, level=0)
                if ptrs_l0:
                    ptr_l0 = ptrs_l0[0]
                    score_l0 = self._score_ptr_against_query(
                        ptr_l0, query_by_level[0], level=0
                    )
                    candidates[sth]["ptr_l0"] = ptr_l0
                    candidates[sth]["score_l0"] = score_l0

        # --- Build final RecallResults ---
        results = []
        for sth, data in candidates.items():
            # Final score = finest available level
            final_score = (
                data["score_l0"]
                if data["score_l0"] is not None
                else data["score_l1"]
                if data["score_l1"] is not None
                else data["score_l2"]
                if data["score_l2"] is not None
                else 0.0
            )

            if final_score < self.min_score:
                continue

            results.append(RecallResult(
                ptr_level2=data["ptr_l2"] or data["ptr_l1"] or data["ptr_l0"],
                ptr_level1=data["ptr_l1"],
                ptr_level0=data["ptr_l0"],
                score_l2=data["score_l2"] or 0.0,
                score_l1=data["score_l1"],
                score_l0=data["score_l0"],
                final_score=final_score,
                source_text_hash=sth,
            ))

        # Sort and assign ranks
        results.sort(key=lambda r: r.final_score, reverse=True)
        for rank, r in enumerate(results[:top_k]):
            r.rank = rank

        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_ptr_for_source_at_level(
        self, source_text_hash: str, level: int
    ) -> List[PointerRecord]:
        """Find pointers for a given source text hash at a specific level."""
        all_by_text = self.registry.get_by_text(source_text_hash)
        return [p for p in all_by_text if p.hgns_level == level and p.is_alive]

    def _score_ptr_against_query(
        self,
        ptr: PointerRecord,
        query_emb: np.ndarray,
        level: int,
    ) -> float:
        """
        Score a pointer against the query by searching its level index.

        We search for the query and find the ptr_id in results.
        If the ptr_id isn't in top results, we return a small positive value
        (it was reachable via a higher level so deserves a non-zero score).
        """
        results = self.indexes[level].search(query_emb, top_k=50)
        for r in results:
            if r.ptr_id == ptr.ptr_id:
                return r.score
        # Not found in top-50 — return a small penalty score
        return 0.01

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            f"level{lv}_size": idx.size
            for lv, idx in self.indexes.items()
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"QueryEngine("
            f"L0={s['level0_size']}, "
            f"L1={s['level1_size']}, "
            f"L2={s['level2_size']} entries)"
        )
