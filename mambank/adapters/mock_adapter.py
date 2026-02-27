"""
mambank/adapters/mock_adapter.py

MockAdapter — deterministic fake model for testing without GPU/model weights.

Purpose
-------
Allows the full MemBank pipeline (buffer, registry, HGNS hierarchy,
retrieval) to be tested and benchmarked without loading an actual LLM.

Behaviour
---------
- encode(text) produces a deterministic float32 vector seeded by the
  SHA256 of the input text. Same text → same vector, always.
- Different texts produce meaningfully different vectors (not random noise):
  semantically similar texts (same words reordered) get similar vectors
  because we hash individual words and average their contributions.
- hidden_dim is configurable at construction time.

This makes MockAdapter useful not just for unit tests, but for integration
tests that verify recall accuracy — you can set up known text pairs and
assert that similar texts retrieve each other.

Semantic Similarity Model
-------------------------
For text "word1 word2 word3":
  1. Hash each word → seed → deterministic unit vector.
  2. Sum vectors, L2-normalise.
  3. Add small position-dependent noise (so order matters slightly).

This gives:
  - "HGNS tames butterfly" ≈ "butterfly HGNS tames"  (high similarity)
  - "HGNS tames butterfly" ≠ "quantum neural network"  (low similarity)
"""

from __future__ import annotations

import hashlib
import numpy as np

from mambank.adapters.base import ModelAdapter


class MockAdapter(ModelAdapter):
    """
    Deterministic fake adapter for testing.

    Parameters
    ----------
    hidden_dim : int
        Output dimensionality. Default 128 (small, fast for tests).
        Use 768 for GPT-2-scale tests, 4096 for Llama-scale tests.
    model_version : str
        Version string appended to model_id. Change this to simulate
        a model swap / pointer invalidation scenario.
    noise_scale : float
        Small noise added to break exact symmetry. Default 0.01.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        model_version: str = "v1",
        noise_scale: float = 0.01,
    ):
        self._hidden_dim = hidden_dim
        self._model_version = model_version
        self._noise_scale = noise_scale

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def model_id(self) -> str:
        return f"mock-{self._hidden_dim}d-{self._model_version}"

    def encode(self, text: str) -> np.ndarray:
        """
        Produce a deterministic pseudo-embedding for `text`.

        Algorithm:
          1. Split text into tokens (whitespace split).
          2. Each token → seeded RNG → unit vector.
          3. Accumulate with position weighting.
          4. Add tiny text-level noise (seeded by full text hash).
          5. L2-normalise and return as float32.
        """
        tokens = text.lower().split()
        if not tokens:
            tokens = ["<empty>"]

        accumulator = np.zeros(self._hidden_dim, dtype=np.float64)

        for pos, token in enumerate(tokens):
            seed = self._text_to_seed(token)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._hidden_dim)
            # Position weighting: earlier tokens slightly more influential
            position_weight = 1.0 / (1.0 + pos * 0.1)
            accumulator += vec * position_weight

        # Add small text-level noise (full text hash, not per-token)
        text_seed = self._text_to_seed(text)
        noise_rng = np.random.default_rng(text_seed ^ 0xDEADBEEF)
        accumulator += noise_rng.standard_normal(self._hidden_dim) * self._noise_scale

        # L2 normalise
        norm = np.linalg.norm(accumulator)
        if norm < 1e-10:
            accumulator = np.ones(self._hidden_dim)
            norm = np.linalg.norm(accumulator)

        result = (accumulator / norm).astype(np.float32)
        return result

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])

    def warmup(self) -> None:
        # Nothing to warm up for a mock, but validate it works
        self.encode("warmup")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_to_seed(text: str) -> int:
        """SHA256 of text → 64-bit integer seed for numpy RNG."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Take first 8 bytes → uint64
        return int.from_bytes(digest[:8], byteorder="little")

    def __repr__(self) -> str:
        return (
            f"MockAdapter(hidden_dim={self._hidden_dim}, "
            f"model_id={self.model_id()!r})"
        )
