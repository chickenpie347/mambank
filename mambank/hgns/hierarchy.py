"""
mambank/hgns/hierarchy.py

HGNSHierarchy — three-level activation compression pipeline.

Architecture
------------
This is the central HGNS component. It takes a raw activation from the
model adapter (Level 0, full resolution) and produces two compressed
variants:

    Level 0: Full activation        e.g. 768-dim   (stored, queried last)
    Level 1: Sentence-level         e.g. 256-dim   (50% HGNS compression + mean-pool)
    Level 2: Topic/paragraph-level  e.g.  64-dim   (fast coarse search)

Retrieval always starts at Level 2 (cheapest), drills to Level 1 for
candidates, and dereferences Level 0 only for the final top-k results.
This mirrors the HGNS base-k traversal: coarse integer part first,
fine sub-steps only when precision is required.

Compression Strategy
--------------------
We use two complementary techniques, chosen to match the model-agnostic
constraint (no fixed projection matrices that would tie us to one dim):

1. HGNS saliency selection (from gradient.py):
   Select the top-N dimensions by multi-level gradient magnitude.
   Content-aware: different activations → different selected dims.
   Used for Level 1 compression.

2. Structured mean-pooling with HGNS weighting:
   Split the activation into k equal segments, weight each segment's
   mean by its HGNS gradient contribution (1/k^l decay), concatenate.
   Produces a fixed-size output regardless of input dim.
   Used for Level 2 compression.

Both are deterministic and parameter-free — no training required,
no matrices to store, works on any embedding dimension.

Index Tracking
--------------
Because saliency selection chooses different dims per activation, we
store `selected_indices` alongside each Level-1 pointer. This allows
aligned cosine similarity at retrieval time (compare only shared dims).
Level 2 uses pooling so no index tracking is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from mambank.hgns.gradient import (
    hgns_compress_with_indices,
    multilevel_gradient,
)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class HierarchicalActivation:
    """
    All three resolution levels of a single activation, plus metadata.

    Produced by HGNSHierarchy.compress() and consumed by the Retriever.

    Fields
    ------
    level0 : np.ndarray, shape (full_dim,)
        Raw activation from the model — maximum fidelity.
    level1 : np.ndarray, shape (level1_dim,)
        HGNS saliency-compressed activation.
    level2 : np.ndarray, shape (level2_dim,)
        HGNS pooled coarse summary.
    level1_indices : np.ndarray, shape (level1_dim,)
        Which dimensions of level0 were selected for level1.
        Required for aligned similarity search.
    source_text : str
        Original text chunk this activation came from.
    model_id : str
        Model that produced level0.
    full_dim : int
        Dimensionality of level0.
    """
    level0: np.ndarray
    level1: np.ndarray
    level2: np.ndarray
    level1_indices: np.ndarray
    source_text: str
    model_id: str
    full_dim: int

    def get_level(self, level: int) -> np.ndarray:
        """Return the activation at the requested HGNS level."""
        return {0: self.level0, 1: self.level1, 2: self.level2}[level]

    def __repr__(self) -> str:
        return (
            f"HierarchicalActivation("
            f"L0={self.level0.shape}, L1={self.level1.shape}, "
            f"L2={self.level2.shape}, model={self.model_id!r})"
        )


# ------------------------------------------------------------------
# Main hierarchy class
# ------------------------------------------------------------------

class HGNSHierarchy:
    """
    Compresses a full activation into the three-level HGNS hierarchy.

    Model-agnostic: accepts any embedding dimension and adapts the
    compression targets proportionally.

    Parameters
    ----------
    k : int
        HGNS base (graduations per level). Default 10.
    num_gradient_levels : int
        Depth of gradient recursion used in saliency computation. Default 4.
    level1_fraction : float
        Level 1 target as a fraction of full_dim. Default 0.33 (~33%).
    level2_fraction : float
        Level 2 target as a fraction of full_dim. Default 0.083 (~8%).
    min_level1_dim : int
        Floor on level 1 dim (prevents degenerate compression). Default 32.
    min_level2_dim : int
        Floor on level 2 dim. Default 8.
    """

    def __init__(
        self,
        k: int = 10,
        num_gradient_levels: int = 4,
        level1_fraction: float = 0.33,
        level2_fraction: float = 0.083,
        min_level1_dim: int = 32,
        min_level2_dim: int = 8,
    ):
        self.k = k
        self.num_gradient_levels = num_gradient_levels
        self.level1_fraction = level1_fraction
        self.level2_fraction = level2_fraction
        self.min_level1_dim = min_level1_dim
        self.min_level2_dim = min_level2_dim

    def _target_dims(self, full_dim: int) -> Tuple[int, int]:
        """
        Compute level1 and level2 target dimensions for a given full_dim.
        Ensures level2_dim < level1_dim < full_dim.
        """
        level1_dim = max(self.min_level1_dim, int(full_dim * self.level1_fraction))
        level2_dim = max(self.min_level2_dim, int(full_dim * self.level2_fraction))

        # Enforce strict hierarchy
        level1_dim = min(level1_dim, full_dim - 1)
        level2_dim = min(level2_dim, level1_dim - 1)
        return level1_dim, level2_dim

    def compress(
        self,
        activation: np.ndarray,
        source_text: str,
        model_id: str,
    ) -> HierarchicalActivation:
        """
        Compress a full activation into the three-level HGNS hierarchy.

        Parameters
        ----------
        activation : np.ndarray, shape (full_dim,)
            Mean-pooled hidden state from the model. Must be 1-D.
        source_text : str
            The text chunk this activation was computed from.
        model_id : str
            Stable model identifier.

        Returns
        -------
        HierarchicalActivation
            All three levels plus metadata.
        """
        if activation.ndim != 1:
            raise ValueError(
                f"Expected 1-D activation, got shape {activation.shape}. "
                f"Mean-pool sequence dimension before calling compress()."
            )

        activation = activation.astype(np.float32)
        full_dim = len(activation)
        level1_dim, level2_dim = self._target_dims(full_dim)

        # Level 0: identity (full resolution)
        level0 = activation.copy()

        # Level 1: HGNS saliency selection
        level1, level1_indices = hgns_compress_with_indices(
            activation,
            target_dim=level1_dim,
            k=self.k,
            num_levels=self.num_gradient_levels,
        )

        # Level 2: HGNS-weighted mean pooling
        level2 = self._hgns_pool(activation, target_dim=level2_dim)

        return HierarchicalActivation(
            level0=level0,
            level1=level1,
            level2=level2,
            level1_indices=level1_indices,
            source_text=source_text,
            model_id=model_id,
            full_dim=full_dim,
        )

    def _hgns_pool(self, activation: np.ndarray, target_dim: int) -> np.ndarray:
        """
        HGNS-weighted mean pooling: split activation into `target_dim`
        segments, weight each by its mean gradient contribution.

        Weighting: segment at position i gets weight proportional to
        mean |gradient| in that segment, modulated by 1/k^l decay —
        mimicking HGNS's hierarchical digit weighting.

        Returns
        -------
        np.ndarray, shape (target_dim,)
        """
        full_dim = len(activation)
        segments = np.array_split(activation, target_dim)

        # Compute gradient for weighting
        grad = multilevel_gradient(
            activation.astype(np.float64),
            k=self.k,
            num_levels=self.num_gradient_levels,
        )
        grad_segments = np.array_split(np.abs(grad), target_dim)

        pooled = np.zeros(target_dim, dtype=np.float32)
        for i, (seg, g_seg) in enumerate(zip(segments, grad_segments)):
            if len(seg) == 0:
                continue
            # HGNS weight: mean gradient magnitude, decayed by level position
            hgns_weight = float(np.mean(g_seg)) / (self.k ** (i % self.num_gradient_levels))
            # Weighted mean of segment values
            seg_mean = float(np.mean(seg))
            pooled[i] = seg_mean * (1.0 + hgns_weight)

        return pooled

    def dims_for(self, full_dim: int) -> Dict[str, int]:
        """Return the compression dimensions that would be used for a given full_dim."""
        l1, l2 = self._target_dims(full_dim)
        return {"level0": full_dim, "level1": l1, "level2": l2}

    def compression_ratio(self, full_dim: int) -> Dict[str, float]:
        """Return compression ratios relative to level0."""
        dims = self.dims_for(full_dim)
        return {
            "level1_ratio": dims["level1"] / dims["level0"],
            "level2_ratio": dims["level2"] / dims["level0"],
        }

    # ------------------------------------------------------------------
    # Similarity (used at retrieval time)
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.
        Returns 0.0 if either vector is zero-norm.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def aligned_cosine_similarity(
        a: np.ndarray,
        a_indices: np.ndarray,
        b: np.ndarray,
        b_indices: np.ndarray,
    ) -> float:
        """
        Cosine similarity between two Level-1 activations that may have
        selected different dimension subsets.

        Computes similarity only on the intersection of selected dimensions,
        ensuring we compare apples to apples.

        Parameters
        ----------
        a, b : np.ndarray, shape (level1_dim,)
            Level-1 compressed activations.
        a_indices, b_indices : np.ndarray
            Which original dimensions were selected for a and b respectively.

        Returns
        -------
        float
            Cosine similarity on shared dimensions, or 0.0 if no overlap.
        """
        shared = np.intersect1d(a_indices, b_indices)
        if len(shared) == 0:
            return 0.0

        a_pos = np.searchsorted(a_indices, shared)
        b_pos = np.searchsorted(b_indices, shared)

        a_shared = a[a_pos]
        b_shared = b[b_pos]

        return HGNSHierarchy.cosine_similarity(a_shared, b_shared)
