"""
mambank/core/pointer.py

PointerRecord — the atomic unit of MemBank storage.

A pointer is a lightweight metadata record that references a location
in the MemMapBuffer. The ptr_id IS the content hash (SHA256 of the raw
activation bytes), giving us:
  - Zero-copy dereference via buffer slice
  - Automatic deduplication (same activation → same ptr_id)
  - Invalidation guard via model_id
  - Smart-pointer style GC via ref_count

Nothing in this file touches an actual activation tensor.
The pointer only knows WHERE the activation lives, not what it is.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Tuple
import json


@dataclass
class PointerRecord:
    """
    A content-addressed pointer into a MemMapBuffer.

    Fields
    ------
    ptr_id : str
        SHA256 hex digest of the raw activation bytes.
        This IS the pointer — it uniquely identifies the activation
        and its location in the buffer.

    buffer_offset : int
        Byte offset into the memmap file where the activation starts.

    shape : Tuple[int, ...]
        Shape needed to reconstruct the numpy view on dereference.
        e.g. (4096,) for a mean-pooled hidden state.

    dtype : str
        numpy dtype string, e.g. "float32".

    model_id : str
        Stable identifier for the model that produced this activation.
        Format: "{model_name}-{param_count}-{version}"
        e.g. "gpt2-117M-v1"
        Used to detect stale pointers when the model is swapped.

    hgns_level : int
        Resolution level in the HGNS hierarchy:
          0 = full resolution (token-level, largest)
          1 = sentence-level (compressed)
          2 = topic/paragraph-level (coarsest, fastest to search)

    source_text_hash : str
        SHA256 of the original text chunk that produced this activation.
        Allows reverse lookup: given text → find its pointer.

    timestamp : float
        Unix timestamp of when this pointer was created.

    ref_count : int
        Number of logical references to this pointer.
        When ref_count reaches 0, the buffer slot is eligible for GC.
        Mirrors C++ shared_ptr semantics.

    metadata : dict
        Arbitrary key-value bag for application-layer annotations.
        e.g. {"conversation_id": "...", "turn": 3, "topic": "HGNS"}
    """

    ptr_id: str
    buffer_offset: int
    shape: Tuple[int, ...]
    dtype: str
    model_id: str
    hgns_level: int
    source_text_hash: str
    timestamp: float = field(default_factory=time.time)
    ref_count: int = 1
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Smart pointer operations
    # ------------------------------------------------------------------

    def retain(self) -> "PointerRecord":
        """Increment ref_count. Call when a new owner takes this pointer."""
        self.ref_count += 1
        return self

    def release(self) -> bool:
        """
        Decrement ref_count.
        Returns True if ref_count has hit 0 (buffer slot ready for GC).
        """
        self.ref_count -= 1
        return self.ref_count <= 0

    @property
    def is_alive(self) -> bool:
        return self.ref_count > 0

    # ------------------------------------------------------------------
    # Serialization — for the SQLite registry
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        d = asdict(self)
        d["shape"] = list(d["shape"])  # JSON can't serialize tuples
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "PointerRecord":
        d = json.loads(s)
        d["shape"] = tuple(d["shape"])
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"PointerRecord(ptr_id={self.ptr_id[:12]}..., "
            f"offset={self.buffer_offset}, shape={self.shape}, "
            f"level={self.hgns_level}, refs={self.ref_count})"
        )


# ------------------------------------------------------------------
# Content hashing utilities
# ------------------------------------------------------------------

def hash_activation(activation_bytes: bytes) -> str:
    """
    SHA256 of raw activation bytes → ptr_id.

    This is the content-address: two identical activations always
    produce the same ptr_id, enabling deduplication at the buffer level.

    Parameters
    ----------
    activation_bytes : bytes
        Raw bytes of the activation tensor (e.g. ndarray.tobytes()).

    Returns
    -------
    str
        64-character hex digest.
    """
    return hashlib.sha256(activation_bytes).hexdigest()


def hash_text(text: str) -> str:
    """
    SHA256 of UTF-8 encoded text → source_text_hash.

    Used for reverse lookup: given a text chunk, find its pointer
    without re-running the model.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_pointer(
    activation_bytes: bytes,
    buffer_offset: int,
    shape: Tuple[int, ...],
    dtype: str,
    model_id: str,
    hgns_level: int,
    source_text: str,
    metadata: dict | None = None,
) -> PointerRecord:
    """
    Factory function — creates a PointerRecord from raw activation bytes.

    The ptr_id and source_text_hash are computed here so callers
    never have to think about hashing.

    Parameters
    ----------
    activation_bytes : bytes
        ndarray.tobytes() of the activation tensor.
    buffer_offset : int
        Byte position in the memmap file where this activation was written.
    shape : tuple
        Shape of the activation array.
    dtype : str
        numpy dtype string.
    model_id : str
        Stable model identifier.
    hgns_level : int
        0 (full) / 1 (sentence) / 2 (topic).
    source_text : str
        The original text chunk this activation came from.
    metadata : dict, optional
        Extra annotations.

    Returns
    -------
    PointerRecord
    """
    return PointerRecord(
        ptr_id=hash_activation(activation_bytes),
        buffer_offset=buffer_offset,
        shape=shape,
        dtype=dtype,
        model_id=model_id,
        hgns_level=hgns_level,
        source_text_hash=hash_text(source_text),
        metadata=metadata or {},
    )
