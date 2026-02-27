"""
mambank/adapters/base.py

Abstract ModelAdapter — the model-agnostic interface contract.

Design Philosophy
-----------------
MemBank never imports a specific model framework at the top level.
All model-specific code lives in a concrete adapter subclass.
Adding support for a new model = writing a ~30-line subclass.

The adapter is responsible for exactly two things:
  1. Reporting the model's hidden state dimensionality.
  2. Encoding a text string into a mean-pooled hidden state vector.

Everything else (compression, storage, retrieval) is MemBank's problem.

Adapter Lifecycle
-----------------
Adapters are stateful — they hold a reference to the loaded model.
MemBank holds one adapter instance and calls encode() repeatedly.
Adapters should be thread-safe if MemBank is used concurrently.

model_id() must return a STABLE string across sessions for the same
model version. This is used as the pointer invalidation key — if model_id
changes, all stored pointers from the old model become GC candidates.

Recommended model_id format:  "{family}-{param_count}-{version}"
Examples:
  "gpt2-117M-v1"
  "llama3-8b-v1"
  "mistral-7b-v0.1"
  "mock-64-test"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    """
    Abstract base class for MemBank model adapters.

    Subclass this to integrate any model with MemBank.
    The interface is intentionally minimal — three methods,
    two of which are properties.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """
        Dimensionality of the mean-pooled hidden state this adapter produces.

        This value must be constant for a given adapter instance.
        MemBank uses it to pre-allocate buffer capacity and set HGNS
        compression targets.

        Example: GPT-2 small = 768, Llama-3-8B = 4096.
        """
        ...

    @abstractmethod
    def model_id(self) -> str:
        """
        Stable, unique identifier for this model version.

        Must not change across process restarts for the same model weights.
        Changing this value invalidates all pointers from the previous model.

        Returns
        -------
        str
            e.g. "gpt2-117M-v1", "llama3-8b-v1"
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a text string into a mean-pooled hidden state vector.

        Implementation requirements:
          - Tokenise `text` using the model's tokenizer.
          - Run a forward pass to obtain hidden states.
          - Mean-pool across the sequence dimension.
          - Return a 1-D float32 numpy array of shape (hidden_dim,).

        The returned array must:
          - Have dtype float32.
          - Have shape (hidden_dim,) — 1-D, already pooled.
          - Be a fresh array (not a view into model internals).

        Parameters
        ----------
        text : str
            A single text chunk (sentence, paragraph, etc.).
            Adapters should truncate gracefully if text exceeds max length.

        Returns
        -------
        np.ndarray, shape (hidden_dim,), dtype float32
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks (override if the model supports them)
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """
        Optional: run a dummy forward pass to warm up the model.
        Called once by MemBank before the first encode().
        Default implementation does nothing.
        """
        pass

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into a batch of hidden state vectors.

        Default implementation calls encode() in a loop — override
        with a batched forward pass for better performance.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        np.ndarray, shape (len(texts), hidden_dim), dtype float32
        """
        return np.stack([self.encode(t) for t in texts])

    # ------------------------------------------------------------------
    # Validation helper (called by MemBank on first use)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Sanity-check the adapter by encoding a short test string.

        Raises
        ------
        ValueError
            If the output shape or dtype is wrong.
        RuntimeError
            If encode() raises an exception.
        """
        test_text = "MemBank adapter validation."
        try:
            output = self.encode(test_text)
        except Exception as e:
            raise RuntimeError(
                f"Adapter {self.__class__.__name__}.encode() raised an exception "
                f"on test input: {e}"
            ) from e

        if output.ndim != 1:
            raise ValueError(
                f"Adapter {self.__class__.__name__}.encode() returned shape "
                f"{output.shape}, expected 1-D array of shape ({self.hidden_dim},)."
            )

        if output.shape[0] != self.hidden_dim:
            raise ValueError(
                f"Adapter {self.__class__.__name__}.encode() returned dim "
                f"{output.shape[0]}, but hidden_dim property reports {self.hidden_dim}."
            )

        if output.dtype != np.float32:
            raise ValueError(
                f"Adapter {self.__class__.__name__}.encode() returned dtype "
                f"{output.dtype}, expected float32."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id()!r}, hidden_dim={self.hidden_dim})"
