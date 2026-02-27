"""
mambank/adapters/hf_adapter.py

HuggingFaceAdapter — captures hidden states from any HF transformer model.

Design
------
Uses PyTorch forward hooks to intercept hidden states during the forward
pass without modifying the model. This is the "non-invasive" approach:
the model itself is unchanged — we just observe its internal state.

Hook Strategy
-------------
We register a hook on the LAST transformer layer. Why last?
- The last layer's hidden states are the most contextualised representations.
- They're what the model "thinks" about the input before producing output.
- This is the activation level with the highest semantic signal for retrieval.

The hook captures the layer output, mean-pools across the sequence
dimension (averaging over all token positions), and stores the result.
This gives a single vector summarising the entire input text.

Model Compatibility
-------------------
Works with any HuggingFace AutoModel that:
  1. Has a transformer backbone with .layers or .h or .blocks attribute.
  2. Returns hidden states when output_hidden_states=True.

Tested families: GPT-2, Llama, Mistral, Falcon, Gemma, BERT, RoBERTa.

For models that require special handling (e.g. MoE routing, custom
attention), subclass HuggingFaceAdapter and override _get_last_layer().

Usage
-----
    from mambank.adapters.hf_adapter import HuggingFaceAdapter

    adapter = HuggingFaceAdapter("gpt2")           # loads model + tokenizer
    adapter.warmup()                                # optional
    embedding = adapter.encode("Hello, HGNS!")     # shape: (768,)

Requirements
------------
    pip install transformers torch
    (Listed under extras_require["hf"] in setup.py)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mambank.adapters.base import ModelAdapter


class HuggingFaceAdapter(ModelAdapter):
    """
    MemBank adapter for HuggingFace transformer models.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
        e.g. "gpt2", "meta-llama/Llama-3-8B", "mistralai/Mistral-7B-v0.1"
    device : str, optional
        Torch device string. Defaults to "cuda" if available, else "cpu".
    max_length : int
        Maximum token sequence length. Longer inputs are truncated.
        Default: 512.
    layer_index : int or None
        Which transformer layer to hook. None = last layer (recommended).
        Positive int = 0-indexed from first layer.
        Negative int = -1 = last, -2 = second-to-last, etc.
    version_tag : str
        Appended to model_id for pointer invalidation control.
        Change this when fine-tuning or updating weights.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        layer_index: Optional[int] = None,
        version_tag: str = "v1",
    ):
        # Lazy import — only required if HF adapter is actually used
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HuggingFaceAdapter requires 'transformers' and 'torch'. "
                "Install with: pip install mambank[hf]"
            ) from e

        import torch as _torch
        self._torch = _torch

        self._model_name = model_name_or_path
        self._version_tag = version_tag
        self._max_length = max_length
        self._layer_index = layer_index
        self._captured: Optional[np.ndarray] = None

        # Device selection
        if device is None:
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
        ).to(self._device)
        self._model.eval()

        # Ensure tokenizer has a pad token (required for batching)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Determine hidden dim from model config
        self._hidden_dim = self._model.config.hidden_size

        # Register forward hook on the target layer
        self._hook_handle = self._register_hook()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _get_last_layer(self):
        """
        Locate the last transformer layer module.

        Handles the naming variations across model families:
          - GPT-2:   model.transformer.h[-1]
          - Llama:   model.model.layers[-1]
          - BERT:    model.encoder.layer[-1]
          - Falcon:  model.transformer.h[-1]
          - Generic: walk children looking for a ModuleList
        """
        m = self._model

        # Try common attribute paths
        for attr_path in [
            "transformer.h",        # GPT-2, Falcon
            "model.layers",         # Llama, Mistral, Gemma
            "encoder.layer",        # BERT, RoBERTa
            "transformer.blocks",   # Some custom models
            "layers",               # Fallback
        ]:
            obj = m
            found = True
            for part in attr_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and hasattr(obj, "__len__"):
                idx = self._layer_index if self._layer_index is not None else -1
                return obj[idx]

        # Last resort: find any ModuleList and use its last element
        import torch.nn as nn
        for module in m.modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                idx = self._layer_index if self._layer_index is not None else -1
                return module[idx]

        raise RuntimeError(
            f"Could not locate transformer layers in model {self._model_name}. "
            f"Override _get_last_layer() in a subclass for this architecture."
        )

    def _register_hook(self):
        """Register a forward hook that captures and mean-pools the layer output."""
        target_layer = self._get_last_layer()

        def hook_fn(module, input, output):
            # output may be a tuple (hidden_state, ...) or just a tensor
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            # hidden shape: (batch, seq_len, hidden_dim)
            # Mean-pool over sequence dimension → (batch, hidden_dim)
            pooled = hidden.mean(dim=1)  # (batch, hidden_dim)
            # Store as numpy float32 (detach from computation graph)
            self._captured = pooled.detach().cpu().float().numpy()

        return target_layer.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def model_id(self) -> str:
        # Normalise model name for use as a stable ID string
        safe_name = self._model_name.replace("/", "-").replace(".", "-").lower()
        return f"{safe_name}-{self._version_tag}"

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string → mean-pooled hidden state vector.

        Uses the registered forward hook to capture the last layer's
        hidden states during the forward pass.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray, shape (hidden_dim,), dtype float32
        """
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=False,
        ).to(self._device)

        self._captured = None  # Clear previous capture
        with torch.no_grad():
            self._model(**inputs)

        if self._captured is None:
            raise RuntimeError(
                "Forward hook did not capture any hidden states. "
                "The hook may not be attached to the correct layer."
            )

        # Hook captures (batch, hidden_dim); batch=1 here → squeeze
        return self._captured[0]  # shape: (hidden_dim,)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts efficiently with a single forward pass.
        """
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,  # pad to longest in batch
        ).to(self._device)

        self._captured = None
        with torch.no_grad():
            self._model(**inputs)

        if self._captured is None:
            raise RuntimeError("Forward hook did not capture hidden states.")

        return self._captured  # shape: (batch_size, hidden_dim)

    def warmup(self) -> None:
        """Run a dummy forward pass to warm up CUDA kernels."""
        self.encode("Warmup pass for MemBank HuggingFace adapter.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hook(self) -> None:
        """Unregister the forward hook. Call when done with this adapter."""
        self._hook_handle.remove()

    def __del__(self):
        try:
            self._hook_handle.remove()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"HuggingFaceAdapter("
            f"model={self._model_name!r}, "
            f"device={self._device!r}, "
            f"hidden_dim={self._hidden_dim})"
        )
