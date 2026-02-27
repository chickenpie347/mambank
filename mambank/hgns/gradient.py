"""
mambank/hgns/gradient.py

HGNS Gradient Structure — direct implementation of the math from the paper.

From "The Hierarchical Gradient Number System" (chickenpie347, Sep 2025):

    ∇f(x) ≈ [f(x + 1/k^l) - f(x)] / (1/k^l)

As l → ∞ this converges to the true derivative for differentiable f.

In MemBank's context, 'f' is the activation tensor viewed as a function
over its coordinate space. The HGNS gradient tells us the rate of change
of activation values across the embedding dimensions — which we use to:

  1. Build a saliency mask (which dims carry the most information)
  2. Drive the iterative compression update rule from Classical__Quantum.pdf:
         v_{n+1} = v_n + η ∇_attr v_n,   stop when ||v_{n+1} - v_n|| < ε
  3. Approximate multi-level gradients with geometric decay:
         ∇^l f = Σ_{l=0}^{L} ∂^(l)f / k^l

These are the building blocks that HGNSHierarchy (hierarchy.py) assembles
into the three-level compression pipeline.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Core HGNS gradient approximation
# ------------------------------------------------------------------

def hgns_gradient_1d(values: np.ndarray, k: int = 10, level: int = 1) -> np.ndarray:
    """
    Approximate the gradient of a 1-D array using HGNS finite differences
    at recursion level `level`.

    The step size at level l is  h = 1 / k^l.
    Uses forward difference:  ∇f[i] ≈ (f[i+h_idx] - f[i]) / h
    where h_idx = max(1, round(n / k^l)) maps the continuous step to an index.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        The 1-D signal to differentiate (e.g. a single activation vector).
    k : int
        HGNS base (graduations per level). Default 10.
    level : int
        Recursion depth. Higher = finer step size = more precise gradient.

    Returns
    -------
    np.ndarray, shape (n,)
        Gradient approximation at each position.
    """
    n = len(values)
    step_size = 1.0 / (k ** level)
    # Map continuous step to discrete index offset (at least 1)
    h_idx = max(1, round(n * step_size))

    grad = np.zeros(n, dtype=np.float64)
    # Forward difference where possible
    grad[: n - h_idx] = (values[h_idx:] - values[: n - h_idx]) / step_size
    # Backward difference at the boundary
    grad[n - h_idx :] = (values[n - h_idx :] - values[n - h_idx - 1 : n - 1]) / step_size
    return grad


def multilevel_gradient(
    values: np.ndarray,
    k: int = 10,
    num_levels: int = 4,
) -> np.ndarray:
    """
    Multi-level HGNS gradient with geometric decay across levels.

    Implements:  ∇^L f = Σ_{l=1}^{L} ∂^(l)f / k^l

    Each level's gradient contribution is downweighted by 1/k^l,
    so fine-grained levels contribute less than coarse ones.
    This mirrors the HGNS number representation:  x = n + Σ m_i / k^i

    Parameters
    ----------
    values : np.ndarray, shape (n,)
    k : int
        HGNS base.
    num_levels : int
        Number of recursion levels to sum.

    Returns
    -------
    np.ndarray, shape (n,)
        Combined multi-level gradient.
    """
    combined = np.zeros_like(values, dtype=np.float64)
    for level in range(1, num_levels + 1):
        level_grad = hgns_gradient_1d(values, k=k, level=level)
        combined += level_grad / (k ** level)
    return combined


def saliency_mask(
    activation: np.ndarray,
    k: int = 10,
    num_levels: int = 4,
    top_fraction: float = 0.5,
) -> np.ndarray:
    """
    Boolean mask of the most salient dimensions in an activation vector,
    derived from the HGNS multi-level gradient magnitude.

    High gradient magnitude → dimension is changing rapidly → high information
    content → should be retained in compressed representations.

    Parameters
    ----------
    activation : np.ndarray, shape (dim,)
        A single activation vector (mean-pooled hidden state).
    k : int
    num_levels : int
    top_fraction : float
        Fraction of dimensions to mark as salient. Default 0.5 (keep top 50%).

    Returns
    -------
    np.ndarray of bool, shape (dim,)
        True where dimension is salient (should be kept).
    """
    grad = multilevel_gradient(activation, k=k, num_levels=num_levels)
    magnitude = np.abs(grad)
    threshold = np.percentile(magnitude, (1.0 - top_fraction) * 100)
    return magnitude >= threshold


# ------------------------------------------------------------------
# Iterative HGNS attribute convergence
# (from Classical__Quantum.pdf: v_{n+1} = v_n + η ∇_attr v_n)
# ------------------------------------------------------------------

def hgns_attribute_convergence(
    v: np.ndarray,
    eta: float = 0.1,
    k: int = 10,
    num_levels: int = 4,
    epsilon: float = 1e-6,
    max_iter: int = 200,
) -> tuple[np.ndarray, int]:
    """
    Iterative HGNS attribute refinement until convergence.

    Update rule:  v_{n+1} = v_n + η * ∇_attr(v_n)
    Stop when:    ||v_{n+1} - v_n|| < ε

    This is the core recursion from the Classical__Quantum paper.
    In MemBank, we use it to "stabilise" a noisy activation toward
    a more deterministic compressed representation — analogous to
    how HGNS tames the butterfly effect by iteratively refining
    initial conditions.

    Parameters
    ----------
    v : np.ndarray
        Initial attribute vector (activation to be stabilised).
    eta : float
        Learning rate / step size.
    k : int
        HGNS base.
    num_levels : int
        Gradient levels to sum.
    epsilon : float
        Convergence threshold.
    max_iter : int
        Safety cap on iterations.

    Returns
    -------
    (converged_vector, n_iterations)
    """
    v = v.astype(np.float64).copy()
    for n in range(max_iter):
        grad = multilevel_gradient(v, k=k, num_levels=num_levels)
        v_new = v + eta * grad
        delta = np.linalg.norm(v_new - v)
        v = v_new
        if delta < epsilon:
            return v.astype(np.float32), n + 1
    return v.astype(np.float32), max_iter


# ------------------------------------------------------------------
# HGNS-based dimensionality reduction
# ------------------------------------------------------------------

def hgns_compress(
    activation: np.ndarray,
    target_dim: int,
    k: int = 10,
    num_levels: int = 4,
) -> np.ndarray:
    """
    Compress an activation vector to `target_dim` using HGNS saliency.

    Strategy:
      1. Compute multi-level gradient magnitude across all dimensions.
      2. Select the `target_dim` dimensions with highest gradient magnitude
         (most information-dense according to HGNS).
      3. Return the selected dimensions as the compressed representation.

    This is content-aware compression: different activations will select
    different dimension subsets, unlike fixed PCA projection.

    Parameters
    ----------
    activation : np.ndarray, shape (dim,)
    target_dim : int
        Number of dimensions in the output.
    k : int
    num_levels : int

    Returns
    -------
    np.ndarray, shape (target_dim,)
        Compressed activation.

    Raises
    ------
    ValueError
        If target_dim >= len(activation).
    """
    dim = len(activation)
    if target_dim >= dim:
        raise ValueError(
            f"target_dim ({target_dim}) must be less than activation dim ({dim}). "
            f"No compression needed."
        )

    grad = multilevel_gradient(activation.astype(np.float64), k=k, num_levels=num_levels)
    magnitude = np.abs(grad)
    top_indices = np.argsort(magnitude)[-target_dim:][::-1]  # descending
    top_indices_sorted = np.sort(top_indices)  # preserve original ordering
    return activation[top_indices_sorted].astype(np.float32)


def hgns_compress_with_indices(
    activation: np.ndarray,
    target_dim: int,
    k: int = 10,
    num_levels: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like hgns_compress, but also returns the selected dimension indices.

    The indices are needed for reconstruction / alignment when comparing
    compressed activations at retrieval time — two activations must be
    compared on their shared dimensions, not arbitrary subsets.

    Returns
    -------
    (compressed_activation, selected_indices)
        compressed_activation : shape (target_dim,)
        selected_indices      : shape (target_dim,) — sorted ascending
    """
    dim = len(activation)
    if target_dim >= dim:
        raise ValueError(f"target_dim ({target_dim}) must be < activation dim ({dim})")

    grad = multilevel_gradient(activation.astype(np.float64), k=k, num_levels=num_levels)
    magnitude = np.abs(grad)
    top_indices = np.argsort(magnitude)[-target_dim:]
    top_indices_sorted = np.sort(top_indices)
    return activation[top_indices_sorted].astype(np.float32), top_indices_sorted
