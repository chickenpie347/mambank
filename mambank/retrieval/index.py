"""
mambank/retrieval/index.py

VectorIndex — FAISS-backed approximate nearest neighbour search,
with a pure-numpy brute-force fallback when FAISS is not installed.

Architecture
------------
One VectorIndex instance per HGNS level. The three indexes are:

    Level 2 (coarsest, smallest dim) — queried first, widest net
    Level 1 (mid, medium dim)        — refinement pass
    Level 0 (full, largest dim)      — final precise ranking

The index stores (embedding, ptr_id) pairs. On search it returns
a ranked list of ptr_ids — callers then dereference via the Registry
and Buffer to get the actual PointerRecords and activations.

FAISS Index Selection
---------------------
We use IndexFlatIP (inner product / cosine after L2-normalisation)
for exact search on small-to-medium corpora (<100k entries).

For larger corpora the upgrade path is IndexIVFFlat (inverted file
with 100–1000 centroids) — the API is identical, only build() changes.
We expose a `use_ivf` flag to opt in to this at construction time.

Pure-numpy Fallback
-------------------
When faiss-cpu is not installed (e.g., CI, lightweight deployment),
we fall back to brute-force cosine search via matrix multiplication.
This is O(n) and suitable for up to ~10k entries. Above that, the
performance degradation should prompt the user to install faiss-cpu.
"""

from __future__ import annotations

import threading
import warnings
from typing import List, Optional, Tuple

import numpy as np

# Try to import FAISS; fall back gracefully
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

class SearchResult:
    """A single result from a vector search."""
    __slots__ = ("ptr_id", "score", "hgns_level", "rank")

    def __init__(self, ptr_id: str, score: float, hgns_level: int, rank: int):
        self.ptr_id = ptr_id
        self.score = score
        self.hgns_level = hgns_level
        self.rank = rank

    def __repr__(self) -> str:
        return (
            f"SearchResult(ptr_id={self.ptr_id[:12]}..., "
            f"score={self.score:.4f}, level={self.hgns_level}, rank={self.rank})"
        )


# ------------------------------------------------------------------
# VectorIndex
# ------------------------------------------------------------------

class VectorIndex:
    """
    Single-level vector index for activation embeddings.

    Parameters
    ----------
    dim : int
        Embedding dimensionality for this level.
    hgns_level : int
        Which HGNS level this index serves (0/1/2).
    use_ivf : bool
        If True and FAISS is available, use IVF index for large corpora.
        Requires calling build() after adding sufficient entries (~256+ for IVF).
    nlist : int
        Number of IVF centroids (only relevant if use_ivf=True). Default 100.
    """

    def __init__(
        self,
        dim: int,
        hgns_level: int,
        use_ivf: bool = False,
        nlist: int = 100,
    ):
        self.dim = dim
        self.hgns_level = hgns_level
        self.use_ivf = use_ivf
        self.nlist = nlist

        self._lock = threading.RLock()
        self._ptr_ids: List[str] = []          # Maps integer index → ptr_id
        self._embeddings: List[np.ndarray] = [] # For numpy fallback

        self._faiss_index = None
        self._built = False

        if _FAISS_AVAILABLE:
            self._init_faiss()
        else:
            warnings.warn(
                "faiss-cpu not found. Using brute-force numpy fallback. "
                "Install with: pip install faiss-cpu",
                RuntimeWarning,
                stacklevel=2,
            )

    def _init_faiss(self) -> None:
        """Initialise the FAISS index (flat IP for exact search)."""
        if self.use_ivf:
            # IVF for large corpora — needs training
            quantiser = faiss.IndexFlatIP(self.dim)
            self._faiss_index = faiss.IndexIVFFlat(
                quantiser, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            # Flat exact search — no training needed, always ready
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self._built = True

    # ------------------------------------------------------------------
    # Add / build
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, ptr_id: str) -> None:
        """
        Add an embedding to the index.

        The embedding is L2-normalised before insertion so that inner
        product = cosine similarity.

        Parameters
        ----------
        embedding : np.ndarray, shape (dim,)
        ptr_id : str
            Content-addressed pointer ID for this embedding.
        """
        if embedding.shape != (self.dim,):
            raise ValueError(
                f"Expected embedding of shape ({self.dim},), "
                f"got {embedding.shape}."
            )

        normed = self._normalise(embedding)

        with self._lock:
            self._ptr_ids.append(ptr_id)

            if _FAISS_AVAILABLE and self._faiss_index is not None:
                vec = normed.astype(np.float32).reshape(1, -1)
                if self._built:
                    self._faiss_index.add(vec)
                else:
                    # IVF not yet trained — buffer in _embeddings
                    self._embeddings.append(normed)
            else:
                self._embeddings.append(normed)

    def build(self) -> None:
        """
        Train and build the IVF index from buffered embeddings.

        Only required when use_ivf=True. For flat indexes this is a no-op.
        Call after adding at least nlist * 10 embeddings for good clustering.
        """
        if not self.use_ivf or self._built:
            return

        if not _FAISS_AVAILABLE:
            self._built = True
            return

        with self._lock:
            if len(self._embeddings) < self.nlist:
                warnings.warn(
                    f"IVF index has only {len(self._embeddings)} entries "
                    f"but nlist={self.nlist}. Training may be poor.",
                    RuntimeWarning,
                )
            matrix = np.stack(self._embeddings).astype(np.float32)
            self._faiss_index.train(matrix)
            self._faiss_index.add(matrix)
            self._embeddings.clear()  # Free memory — FAISS holds a copy
            self._built = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Find the top-k most similar embeddings to `query`.

        Parameters
        ----------
        query : np.ndarray, shape (dim,)
        top_k : int

        Returns
        -------
        List[SearchResult]
            Ranked by descending cosine similarity (score 1.0 = identical).
        """
        if query.shape != (self.dim,):
            raise ValueError(
                f"Query shape {query.shape} doesn't match index dim ({self.dim},)."
            )

        if self.size == 0:
            return []

        normed_query = self._normalise(query)
        actual_k = min(top_k, self.size)

        with self._lock:
            if _FAISS_AVAILABLE and self._faiss_index is not None and self._built:
                results = self._faiss_search(normed_query, actual_k)
            else:
                results = self._numpy_search(normed_query, actual_k)

        return results

    def _faiss_search(self, normed_query: np.ndarray, k: int) -> List[SearchResult]:
        vec = normed_query.astype(np.float32).reshape(1, -1)
        scores, indices = self._faiss_index.search(vec, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            results.append(SearchResult(
                ptr_id=self._ptr_ids[idx],
                score=float(score),
                hgns_level=self.hgns_level,
                rank=rank,
            ))
        return results

    def _numpy_search(self, normed_query: np.ndarray, k: int) -> List[SearchResult]:
        if not self._embeddings:
            return []
        matrix = np.stack(self._embeddings).astype(np.float32)
        scores = matrix @ normed_query.astype(np.float32)
        top_indices = np.argsort(scores)[-k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                ptr_id=self._ptr_ids[idx],
                score=float(scores[idx]),
                hgns_level=self.hgns_level,
                rank=rank,
            ))
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return np.zeros_like(v, dtype=np.float32)
        return (v / norm).astype(np.float32)

    @property
    def size(self) -> int:
        """Number of embeddings currently indexed."""
        with self._lock:
            if _FAISS_AVAILABLE and self._faiss_index is not None and self._built:
                return self._faiss_index.ntotal
            return len(self._embeddings)

    @property
    def backend(self) -> str:
        return "faiss" if (_FAISS_AVAILABLE and self._faiss_index is not None) else "numpy"

    def __repr__(self) -> str:
        return (
            f"VectorIndex(level={self.hgns_level}, dim={self.dim}, "
            f"size={self.size}, backend={self.backend!r})"
        )
