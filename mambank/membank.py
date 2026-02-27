"""
mambank/membank.py

MemBank — the main public API.

This is the only class end users need to import. It orchestrates:
  Adapter → HGNSHierarchy → Buffer + Registry → QueryEngine

Usage
-----
    from mambank import MemBank
    from mambank.adapters.mock_adapter import MockAdapter

    bank = MemBank(adapter=MockAdapter(hidden_dim=256))

    # Ingest text — compresses to 3 HGNS levels, stores pointers
    bank.ingest("The HGNS framework introduces recursive sub-steps.")
    bank.ingest("Quantum chaos is tamed through deterministic iteration.")

    # Recall — multi-level coarse-to-fine retrieval
    results = bank.recall("HGNS butterfly effect", top_k=3)
    for r in results:
        print(r.rank, r.final_score, r.source_text_hash[:12])

    # Integrate — ingest + recall + augmented prompt in one call
    augmented = bank.integrate("How does HGNS tame chaos?")

    # Stats
    print(bank.stats())

Recursive Population
--------------------
MemBank supports auto-population: every call to ingest() or integrate()
adds the new text to the store. In integrate(), the recalled context AND
the new query are both ingested — the store grows with every interaction,
exactly like the recursive DB population described in the MemBank design.

Thread Safety
-------------
ingest() and recall() are individually thread-safe (buffer and registry
use locks internally). Concurrent ingest + recall is safe. Concurrent
ingest calls are safe. However, integrate() is NOT atomic — if two threads
call integrate() simultaneously the behaviour is correct but the augmented
prompts may not see each other's ingestions.

Persistence
-----------
The buffer is a memmap file that survives process restarts. The registry
is SQLite (persistent if given a file path). The FAISS/numpy indexes are
in-memory and are rebuilt from the registry on startup — call
MemBank.rebuild_indexes() after opening an existing store.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from mambank.adapters.base import ModelAdapter
from mambank.core.buffer import MemMapBuffer
from mambank.core.pointer import hash_text, make_pointer, PointerRecord
from mambank.core.registry import Registry
from mambank.hgns.hierarchy import HGNSHierarchy, HierarchicalActivation
from mambank.retrieval.index import VectorIndex
from mambank.retrieval.query import QueryEngine, RecallResult


class MemBank:
    """
    Pointer-based neural activation memory for open-source LLMs.

    Parameters
    ----------
    adapter : ModelAdapter
        Any concrete adapter (MockAdapter, HuggingFaceAdapter, etc.).
    buffer_path : str or Path, optional
        Path for the persistent memmap activation buffer.
        Defaults to "./mambank_buffer.mmap".
    registry_path : str or Path, optional
        Path for the SQLite registry.
        Use ":memory:" for a non-persistent in-memory registry (tests/demos).
        Defaults to "./mambank_registry.db".
    buffer_capacity_bytes : int
        Initial buffer file size. Grows automatically. Default 128 MB.
    hgns_k : int
        HGNS base for hierarchy compression. Default 10.
    hgns_gradient_levels : int
        Gradient recursion depth. Default 4.
    candidate_multiplier : int
        Level-2 over-fetch factor for refinement. Default 5.
    min_recall_score : float
        Discard results below this cosine similarity. Default 0.0.
    auto_validate_adapter : bool
        Run adapter.validate() on construction. Default True.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        buffer_path: Union[str, Path] = "./mambank_buffer.mmap",
        registry_path: Union[str, Path] = ":memory:",
        buffer_capacity_bytes: int = 128 * 1024 * 1024,
        hgns_k: int = 10,
        hgns_gradient_levels: int = 4,
        candidate_multiplier: int = 5,
        min_recall_score: float = 0.0,
        auto_validate_adapter: bool = True,
    ):
        self.adapter = adapter
        self._ingest_lock = threading.Lock()

        # Validate adapter
        if auto_validate_adapter:
            adapter.validate()

        # Initialise HGNS hierarchy
        self.hierarchy = HGNSHierarchy(
            k=hgns_k,
            num_gradient_levels=hgns_gradient_levels,
        )

        # Determine compression dims from adapter's hidden_dim
        self._dims = self.hierarchy.dims_for(adapter.hidden_dim)

        # Initialise storage
        self.buffer = MemMapBuffer(
            path=buffer_path,
            initial_capacity_bytes=buffer_capacity_bytes,
        )
        self.registry = Registry(path=str(registry_path))

        # Initialise query engine
        self.query_engine = QueryEngine(
            hierarchy=self.hierarchy,
            registry=self.registry,
            dims=self._dims,
            candidate_multiplier=candidate_multiplier,
            min_score=min_recall_score,
        )

        # Telemetry
        self._ingest_count = 0
        self._recall_count = 0
        self._dedup_count = 0
        self._created_at = time.time()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def ingest(self, text: str, metadata: Optional[dict] = None) -> dict:
        """
        Encode `text`, compress to 3 HGNS levels, store pointers.

        This is the recursive population step: every new piece of text
        is automatically encoded and added to the activation store.

        Deduplication is automatic: if the same text has been ingested
        before (same content hash), the buffer is NOT written again —
        the registry ref_count is incremented instead.

        Parameters
        ----------
        text : str
            Any text chunk: a sentence, paragraph, or conversation turn.
        metadata : dict, optional
            Extra annotations stored on the pointer records.
            e.g. {"conversation_id": "...", "turn": 3}

        Returns
        -------
        dict
            Summary: {
                "ptr_ids": {"l0": str, "l1": str, "l2": str},
                "is_new": bool,       # False if deduplicated
                "dims": dict,         # embedding dims per level
                "ingest_ms": float,
            }
        """
        t0 = time.perf_counter()
        meta = metadata or {}

        with self._ingest_lock:
            # 1. Encode text → full activation
            raw_activation = self.adapter.encode(text)

            # 2. HGNS compress → all three levels
            ha: HierarchicalActivation = self.hierarchy.compress(
                raw_activation,
                source_text=text,
                model_id=self.adapter.model_id(),
            )

            # 3. Store each level — dedup via registry
            ptr_ids = {}
            is_new_any = False

            for level, emb in [(0, ha.level0), (1, ha.level1), (2, ha.level2)]:
                level_meta = {**meta, "hgns_level": level}
                if level == 1:
                    # Store selected indices for aligned similarity at retrieval time
                    level_meta["level1_indices"] = ha.level1_indices.tolist()

                raw_bytes = emb.tobytes()

                # Check registry first — if already present, skip buffer write
                ptr_id_candidate = self._compute_ptr_id(raw_bytes)

                if self.registry.exists(ptr_id_candidate):
                    # Dedup: increment ref_count, no buffer write
                    dummy_ptr = make_pointer(
                        raw_bytes, 0, emb.shape, str(emb.dtype),
                        self.adapter.model_id(), level, text, level_meta,
                    )
                    self.registry.put(dummy_ptr)  # increments ref_count
                    ptr_ids[f"l{level}"] = ptr_id_candidate
                    self._dedup_count += 1
                else:
                    # New activation — write to buffer
                    ptr = self.buffer.write(
                        emb,
                        model_id=self.adapter.model_id(),
                        hgns_level=level,
                        source_text=text,
                        metadata=level_meta,
                    )
                    self.registry.put(ptr)
                    ptr_ids[f"l{level}"] = ptr.ptr_id
                    is_new_any = True

                    # Add to search index
                    self.query_engine.indexes[level].add(emb, ptr_ids[f"l{level}"])

            self._ingest_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "ptr_ids": ptr_ids,
            "is_new": is_new_any,
            "dims": {f"level{i}": self._dims[f"level{i}"] for i in range(3)},
            "ingest_ms": round(elapsed_ms, 3),
        }

    def recall(
        self,
        query: str,
        top_k: int = 5,
        levels: Optional[List[int]] = None,
    ) -> List[RecallResult]:
        """
        Find the top-k most relevant stored activations for `query`.

        Uses the HGNS coarse-to-fine drill-down:
          L2 → candidate set → L1 refinement → L0 final ranking.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int
            Number of results to return.
        levels : list of int, optional
            Which HGNS levels to include. Default [2, 1, 0] (full drill-down).

        Returns
        -------
        List[RecallResult]
            Sorted by final_score descending.
        """
        t0 = time.perf_counter()
        query_emb = self.adapter.encode(query)
        results = self.query_engine.search(query_emb, top_k=top_k, levels=levels)
        self._recall_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Attach timing to results for instrumentation
        for r in results:
            r.rank = results.index(r)
        return results

    def integrate(
        self,
        text: str,
        top_k: int = 3,
        recall_first: bool = True,
    ) -> dict:
        """
        Full pipeline: recall relevant context + ingest new text.

        This is the "active working memory" mode: every new message
        retrieves relevant past context AND adds itself to the store
        for future recall. The store grows recursively with every call.

        Parameters
        ----------
        text : str
            New input text (e.g., a user message or conversation turn).
        top_k : int
            Number of recalled context items to return.
        recall_first : bool
            If True (default), recall before ingesting `text` — so the
            query doesn't retrieve itself. If False, ingest first.

        Returns
        -------
        dict
            {
                "query": str,
                "recalled": List[RecallResult],
                "augmented_prompt": str,   # query + recalled context
                "ingest_result": dict,     # from ingest()
            }
        """
        if recall_first:
            recalled = self.recall(text, top_k=top_k)
            ingest_result = self.ingest(text)
        else:
            ingest_result = self.ingest(text)
            recalled = self.recall(text, top_k=top_k)

        augmented = self._build_augmented_prompt(text, recalled)

        return {
            "query": text,
            "recalled": recalled,
            "augmented_prompt": augmented,
            "ingest_result": ingest_result,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def rebuild_indexes(self) -> int:
        """
        Rebuild all three search indexes from the registry + buffer.

        Call this after opening an existing (persisted) MemBank store
        to restore search capability. The indexes are in-memory only
        and are not persisted to disk.

        Returns
        -------
        int
            Number of pointer records reindexed.
        """
        count = 0
        for level in [0, 1, 2]:
            ptrs = self.registry.get_by_level(level)
            for ptr in ptrs:
                if not ptr.is_alive:
                    continue
                try:
                    emb = self.buffer.deref(ptr)
                    self.query_engine.indexes[level].add(emb, ptr.ptr_id)
                    count += 1
                except (ValueError, Exception):
                    # Corrupt pointer — skip
                    pass
        return count

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def gc(self) -> dict:
        """
        Collect dead pointers (ref_count <= 0).

        Currently marks them as deleted in the registry. Buffer slot
        reclamation (defragmentation) is not implemented in the PoC
        — dead slots are simply abandoned. This is acceptable because
        the buffer grows by doubling and dead entries are rare in normal use.

        Returns
        -------
        dict
            {"collected": int, "registry_deleted": int}
        """
        candidates = self.registry.gc_candidates()
        for ptr in candidates:
            self.registry.delete(ptr.ptr_id)
        return {
            "collected": len(candidates),
            "registry_deleted": len(candidates),
        }

    # ------------------------------------------------------------------
    # Stats & introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        buf_stats = self.buffer.stats()
        reg_stats = self.registry.stats()
        idx_stats = self.query_engine.stats()
        return {
            "adapter": repr(self.adapter),
            "model_id": self.adapter.model_id(),
            "dims": self._dims,
            "ingest_count": self._ingest_count,
            "recall_count": self._recall_count,
            "dedup_count": self._dedup_count,
            "uptime_seconds": round(time.time() - self._created_at, 1),
            "buffer": buf_stats,
            "registry": reg_stats,
            "indexes": idx_stats,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ptr_id(raw_bytes: bytes) -> str:
        from mambank.core.pointer import hash_activation
        return hash_activation(raw_bytes)

    @staticmethod
    def _build_augmented_prompt(query: str, recalled: List[RecallResult]) -> str:
        """
        Build a context-augmented prompt string.

        Format:
            [MemBank Context]
            [1] (score=0.9231) <source_text_hash[:12]>
            [2] (score=0.8874) <source_text_hash[:12]>
            ...
            [Query]
            <original query text>
        """
        if not recalled:
            return f"[Query]\n{query}"

        lines = ["[MemBank Context]"]
        for r in recalled:
            lines.append(
                f"[{r.rank + 1}] (score={r.final_score:.4f}) "
                f"chunk:{r.source_text_hash[:12]}"
            )
        lines.append(f"\n[Query]\n{query}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close buffer and registry."""
        self.buffer.close()
        self.registry.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"MemBank(adapter={self.adapter.model_id()!r}, "
            f"ingested={self._ingest_count}, "
            f"recalled={self._recall_count})"
        )
