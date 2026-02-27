"""
mambank/core/buffer.py

MemMapBuffer — persistent, zero-copy activation storage.

Design
------
A single flat numpy memmap file on disk holds all activations packed
end-to-end. Each activation is stored at a known byte offset, which
is recorded in the PointerRecord.

Dereference is a numpy slice — no deserialization, no copy:
    activation = buffer.deref(pointer)   # returns a np.ndarray VIEW

The buffer grows by doubling (like a dynamic array / C++ vector)
when capacity is exceeded. Existing offsets remain valid after growth
because we only append — never move existing data.

Thread safety: write lock on append, lock-free reads (memmap is
safe for concurrent readers after the write is flushed).

Layout
------
memmap file: [header_slot][activation_0][activation_1]...
              8 bytes        n_bytes        n_bytes

header_slot: stores (write_cursor: uint64) so the buffer can be
             reopened and resumed after a process restart.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Tuple

import numpy as np

from mambank.core.pointer import PointerRecord, make_pointer

# First 8 bytes of the file are reserved for the write cursor (uint64).
HEADER_BYTES = 8
HEADER_DTYPE = np.uint64


class MemMapBuffer:
    """
    Flat memory-mapped file that stores activation tensors.

    Parameters
    ----------
    path : str | Path
        File path for the memmap file. Created if it doesn't exist.
    initial_capacity_bytes : int
        Initial file size. Doubles automatically when full.
        Default: 64 MB (comfortable for ~16k activations at 4096-dim f32).
    """

    def __init__(
        self,
        path: str | Path,
        initial_capacity_bytes: int = 64 * 1024 * 1024,  # 64 MB
    ):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._initial_capacity = initial_capacity_bytes

        if self.path.exists():
            self._open_existing()
        else:
            self._create_new(initial_capacity_bytes)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _create_new(self, capacity_bytes: int) -> None:
        """Create a fresh buffer file with header."""
        total = HEADER_BYTES + capacity_bytes
        # Pre-allocate file
        with open(self.path, "wb") as f:
            f.seek(total - 1)
            f.write(b"\x00")

        self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(total,))
        # Write cursor starts just after the header
        self._write_cursor = HEADER_BYTES
        self._flush_cursor()
        self._capacity = total

    def _open_existing(self) -> None:
        """Reopen an existing buffer and restore write cursor."""
        file_size = self.path.stat().st_size
        self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(file_size,))
        self._capacity = file_size
        # Read cursor from header
        cursor_bytes = bytes(self._mmap[:HEADER_BYTES])
        self._write_cursor = int(np.frombuffer(cursor_bytes, dtype=HEADER_DTYPE)[0])

    def _flush_cursor(self) -> None:
        """Persist write cursor to header slot."""
        cursor_arr = np.array([self._write_cursor], dtype=HEADER_DTYPE)
        self._mmap[:HEADER_BYTES] = np.frombuffer(cursor_arr.tobytes(), dtype=np.uint8)
        self._mmap.flush()

    # ------------------------------------------------------------------
    # Core write / read
    # ------------------------------------------------------------------

    def write(
        self,
        activation: np.ndarray,
        model_id: str,
        hgns_level: int,
        source_text: str,
        metadata: dict | None = None,
    ) -> PointerRecord:
        """
        Write an activation to the buffer.

        If an identical activation already exists (same content hash),
        this is a no-op and the existing PointerRecord is returned —
        deduplication is handled by the Registry, not here.
        The buffer itself always writes; dedup is the Registry's job.

        Parameters
        ----------
        activation : np.ndarray
            The activation tensor to store.
        model_id : str
            Stable model identifier.
        hgns_level : int
            HGNS resolution level (0/1/2).
        source_text : str
            Original text this activation came from.
        metadata : dict, optional

        Returns
        -------
        PointerRecord
            Pointer to the written activation.
        """
        raw_bytes = activation.tobytes()
        n_bytes = len(raw_bytes)

        with self._lock:
            # Grow if needed
            if self._write_cursor + n_bytes > self._capacity:
                self._grow(n_bytes)

            offset = self._write_cursor

            # Write raw bytes into memmap
            byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            self._mmap[offset : offset + n_bytes] = byte_arr

            # Advance cursor
            self._write_cursor += n_bytes
            self._flush_cursor()

        return make_pointer(
            activation_bytes=raw_bytes,
            buffer_offset=offset,
            shape=activation.shape,
            dtype=str(activation.dtype),
            model_id=model_id,
            hgns_level=hgns_level,
            source_text=source_text,
            metadata=metadata or {},
        )

    def deref(self, pointer: PointerRecord) -> np.ndarray:
        """
        Dereference a pointer → numpy view of the activation.

        This is a ZERO-COPY operation. The returned array is a view
        into the memmap file. Do not modify it unless you want to
        corrupt the buffer.

        Parameters
        ----------
        pointer : PointerRecord

        Returns
        -------
        np.ndarray
            View of the stored activation with the correct shape/dtype.

        Raises
        ------
        ValueError
            If the pointer's offset is out of bounds.
        """
        offset = pointer.buffer_offset
        dtype = np.dtype(pointer.dtype)
        n_elements = int(np.prod(pointer.shape))
        n_bytes = n_elements * dtype.itemsize

        if offset + n_bytes > self._capacity:
            raise ValueError(
                f"Pointer offset {offset} + {n_bytes} bytes exceeds "
                f"buffer capacity {self._capacity}. Buffer may be corrupt."
            )

        raw_view = self._mmap[offset : offset + n_bytes]
        flat = raw_view.view(dtype)
        return flat.reshape(pointer.shape)

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _grow(self, min_additional_bytes: int) -> None:
        """
        Double buffer capacity (or grow by min_additional_bytes if larger).
        Called with self._lock held.
        """
        new_capacity = max(self._capacity * 2, self._capacity + min_additional_bytes)
        self._mmap.flush()
        del self._mmap  # Close memmap before resizing file

        with open(self.path, "ab") as f:
            extra = new_capacity - self._capacity
            f.seek(new_capacity - 1)
            f.write(b"\x00")

        self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(new_capacity,))
        self._capacity = new_capacity

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        """Bytes currently occupied by activations (excluding header)."""
        return self._write_cursor - HEADER_BYTES

    @property
    def capacity_bytes(self) -> int:
        """Total bytes available for activations (excluding header)."""
        return self._capacity - HEADER_BYTES

    @property
    def utilization(self) -> float:
        """Fill ratio 0.0 → 1.0."""
        return self.used_bytes / self.capacity_bytes if self.capacity_bytes > 0 else 0.0

    def stats(self) -> dict:
        return {
            "path": str(self.path),
            "used_bytes": self.used_bytes,
            "capacity_bytes": self.capacity_bytes,
            "utilization": f"{self.utilization:.1%}",
            "write_cursor": self._write_cursor,
        }

    def close(self) -> None:
        """Flush and close the memmap."""
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.flush()
            del self._mmap

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return (
            f"MemMapBuffer(path={self.path.name!r}, "
            f"used={self.used_bytes // 1024}KB / "
            f"{self.capacity_bytes // 1024}KB)"
        )
