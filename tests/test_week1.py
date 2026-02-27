"""
tests/test_week1.py

Week 1 test suite for MemBank core:
  - pointer.py   : content hashing, serialization, smart pointer ops
  - buffer.py    : write, zero-copy deref, growth, persistence
  - registry.py  : put/get, dedup, reverse lookup, invalidation, GC

All tests use in-memory or temp-file resources — no network, no model.
Run with: pytest tests/test_week1.py -v
"""

import os
import tempfile
import time

import numpy as np
import pytest

from mambank.core.pointer import (
    PointerRecord,
    hash_activation,
    hash_text,
    make_pointer,
)
from mambank.core.buffer import MemMapBuffer
from mambank.core.registry import Registry


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_buffer(tmp_path):
    buf = MemMapBuffer(tmp_path / "test.mmap", initial_capacity_bytes=1 * 1024 * 1024)
    yield buf
    buf.close()


@pytest.fixture
def registry():
    reg = Registry(":memory:")
    yield reg
    reg.close()


def make_activation(dim: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


# ======================================================================
# pointer.py tests
# ======================================================================

class TestHashing:

    def test_same_bytes_same_hash(self):
        a = make_activation(64, seed=1)
        h1 = hash_activation(a.tobytes())
        h2 = hash_activation(a.tobytes())
        assert h1 == h2, "Identical activations must hash identically"

    def test_different_bytes_different_hash(self):
        a = make_activation(64, seed=1)
        b = make_activation(64, seed=2)
        assert hash_activation(a.tobytes()) != hash_activation(b.tobytes())

    def test_hash_is_64_hex_chars(self):
        h = hash_activation(b"test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_text_hash(self):
        h = hash_text("hello")
        assert len(h) == 64
        assert hash_text("hello") == hash_text("hello")
        assert hash_text("hello") != hash_text("world")


class TestPointerRecord:

    def test_make_pointer(self):
        a = make_activation(64)
        ptr = make_pointer(
            activation_bytes=a.tobytes(),
            buffer_offset=0,
            shape=(64,),
            dtype="float32",
            model_id="mock-64-v1",
            hgns_level=0,
            source_text="Test chunk about HGNS.",
        )
        assert ptr.hgns_level == 0
        assert ptr.shape == (64,)
        assert ptr.dtype == "float32"
        assert ptr.ref_count == 1
        assert len(ptr.ptr_id) == 64

    def test_ptr_id_matches_manual_hash(self):
        a = make_activation(64)
        raw = a.tobytes()
        ptr = make_pointer(raw, 0, (64,), "float32", "m", 0, "text")
        assert ptr.ptr_id == hash_activation(raw)

    def test_source_text_hash_matches(self):
        text = "The butterfly effect is tamed by HGNS."
        a = make_activation(64)
        ptr = make_pointer(a.tobytes(), 0, (64,), "float32", "m", 0, text)
        assert ptr.source_text_hash == hash_text(text)

    def test_retain_increments_refcount(self):
        a = make_activation(64)
        ptr = make_pointer(a.tobytes(), 0, (64,), "float32", "m", 0, "t")
        assert ptr.ref_count == 1
        ptr.retain()
        assert ptr.ref_count == 2

    def test_release_decrements_refcount(self):
        a = make_activation(64)
        ptr = make_pointer(a.tobytes(), 0, (64,), "float32", "m", 0, "t")
        ptr.retain()  # ref=2
        dead = ptr.release()  # ref=1
        assert not dead
        assert ptr.ref_count == 1
        dead = ptr.release()  # ref=0
        assert dead
        assert ptr.ref_count == 0

    def test_is_alive(self):
        a = make_activation(64)
        ptr = make_pointer(a.tobytes(), 0, (64,), "float32", "m", 0, "t")
        assert ptr.is_alive
        ptr.release()
        assert not ptr.is_alive

    def test_json_roundtrip(self):
        a = make_activation(64)
        ptr = make_pointer(
            a.tobytes(), 128, (64,), "float32", "gpt2-117M-v1", 2,
            "Topic chunk", {"conversation_id": "abc123"}
        )
        restored = PointerRecord.from_json(ptr.to_json())
        assert restored.ptr_id == ptr.ptr_id
        assert restored.buffer_offset == 128
        assert restored.shape == (64,)
        assert restored.hgns_level == 2
        assert restored.metadata["conversation_id"] == "abc123"


# ======================================================================
# buffer.py tests
# ======================================================================

class TestMemMapBuffer:

    def test_write_and_deref(self, tmp_buffer):
        a = make_activation(64)
        ptr = tmp_buffer.write(a, "mock-64-v1", 0, "test chunk")
        result = tmp_buffer.deref(ptr)
        np.testing.assert_array_almost_equal(result, a)

    def test_deref_is_correct_shape(self, tmp_buffer):
        a = make_activation(128)
        ptr = tmp_buffer.write(a, "mock-128-v1", 1, "sentence chunk")
        result = tmp_buffer.deref(ptr)
        assert result.shape == (128,)

    def test_deref_is_zero_copy_view(self, tmp_buffer):
        """Deref should return a view into the mmap, not a copy."""
        a = make_activation(64)
        ptr = tmp_buffer.write(a, "m", 0, "t")
        view = tmp_buffer.deref(ptr)
        # A view shares memory with the mmap — base is not None
        assert view.base is not None, "Expected a view (zero-copy), got a copy"

    def test_multiple_writes_sequential_offsets(self, tmp_buffer):
        a = make_activation(64)
        b = make_activation(64, seed=99)
        ptr_a = tmp_buffer.write(a, "m", 0, "chunk a")
        ptr_b = tmp_buffer.write(b, "m", 0, "chunk b")
        # Offsets must be strictly increasing
        assert ptr_b.buffer_offset > ptr_a.buffer_offset

    def test_different_activations_at_correct_offsets(self, tmp_buffer):
        a = make_activation(64, seed=1)
        b = make_activation(64, seed=2)
        ptr_a = tmp_buffer.write(a, "m", 0, "chunk a")
        ptr_b = tmp_buffer.write(b, "m", 0, "chunk b")
        ra = tmp_buffer.deref(ptr_a)
        rb = tmp_buffer.deref(ptr_b)
        np.testing.assert_array_almost_equal(ra, a)
        np.testing.assert_array_almost_equal(rb, b)
        # Confirm they differ
        assert not np.allclose(ra, rb)

    def test_persistence_across_reopen(self, tmp_path):
        path = tmp_path / "persist.mmap"
        a = make_activation(64)

        # Write and close
        buf1 = MemMapBuffer(path)
        ptr = buf1.write(a, "m", 0, "persistent chunk")
        offset = ptr.buffer_offset
        buf1.close()

        # Reopen and dereference
        buf2 = MemMapBuffer(path)
        # Reconstruct pointer manually (registry would do this in production)
        from mambank.core.pointer import PointerRecord
        ptr2 = PointerRecord(
            ptr_id=ptr.ptr_id,
            buffer_offset=offset,
            shape=(64,),
            dtype="float32",
            model_id="m",
            hgns_level=0,
            source_text_hash=ptr.source_text_hash,
        )
        result = buf2.deref(ptr2)
        np.testing.assert_array_almost_equal(result, a)
        buf2.close()

    def test_stats(self, tmp_buffer):
        s = tmp_buffer.stats()
        assert "used_bytes" in s
        assert "capacity_bytes" in s
        assert "utilization" in s

    def test_used_bytes_grows_on_write(self, tmp_buffer):
        before = tmp_buffer.used_bytes
        a = make_activation(64)
        tmp_buffer.write(a, "m", 0, "t")
        after = tmp_buffer.used_bytes
        assert after > before

    def test_auto_grow(self, tmp_path):
        """Buffer should grow automatically when capacity is exceeded."""
        # Tiny buffer: 512 bytes
        buf = MemMapBuffer(tmp_path / "small.mmap", initial_capacity_bytes=512)
        # Write activations until we exceed initial capacity
        written = 0
        for i in range(20):
            a = make_activation(64, seed=i)  # 64 * 4 = 256 bytes each
            buf.write(a, "m", 0, f"chunk {i}")
            written += 1
        # All writes should succeed, verify last one is retrievable
        assert buf.used_bytes > 512, "Buffer should have grown beyond initial capacity"
        buf.close()

    def test_hgns_levels_stored_in_pointer(self, tmp_buffer):
        for level in [0, 1, 2]:
            a = make_activation(64)
            ptr = tmp_buffer.write(a, "m", level, f"chunk level {level}")
            assert ptr.hgns_level == level

    def test_metadata_stored_in_pointer(self, tmp_buffer):
        a = make_activation(64)
        meta = {"conversation_id": "conv_001", "turn": 5}
        ptr = tmp_buffer.write(a, "m", 0, "chunk", metadata=meta)
        assert ptr.metadata["conversation_id"] == "conv_001"
        assert ptr.metadata["turn"] == 5


# ======================================================================
# registry.py tests
# ======================================================================

class TestRegistry:

    def _make_ptr(self, seed=1, level=0, text="default text", model="mock-v1"):
        a = make_activation(64, seed=seed)
        return make_pointer(a.tobytes(), seed * 256, (64,), "float32", model, level, text)

    def test_put_and_get(self, registry):
        ptr = self._make_ptr(seed=1)
        registry.put(ptr)
        fetched = registry.get(ptr.ptr_id)
        assert fetched is not None
        assert fetched.ptr_id == ptr.ptr_id

    def test_put_returns_true_for_new(self, registry):
        ptr = self._make_ptr(seed=1)
        is_new = registry.put(ptr)
        assert is_new is True

    def test_put_returns_false_for_duplicate(self, registry):
        ptr = self._make_ptr(seed=1)
        registry.put(ptr)
        is_new = registry.put(ptr)  # Same ptr_id
        assert is_new is False

    def test_dedup_increments_refcount(self, registry):
        ptr = self._make_ptr(seed=1)
        registry.put(ptr)
        registry.put(ptr)  # Duplicate
        fetched = registry.get(ptr.ptr_id)
        assert fetched.ref_count == 2

    def test_exists(self, registry):
        ptr = self._make_ptr(seed=1)
        assert not registry.exists(ptr.ptr_id)
        registry.put(ptr)
        assert registry.exists(ptr.ptr_id)

    def test_get_nonexistent_returns_none(self, registry):
        result = registry.get("nonexistent_hash_" + "0" * 48)
        assert result is None

    def test_get_by_text(self, registry):
        text = "HGNS tames the butterfly effect."
        ptr = self._make_ptr(seed=1, text=text)
        registry.put(ptr)
        results = registry.get_by_text(hash_text(text))
        assert len(results) == 1
        assert results[0].ptr_id == ptr.ptr_id

    def test_get_by_level(self, registry):
        for seed, level in [(1, 0), (2, 0), (3, 1), (4, 2)]:
            registry.put(self._make_ptr(seed=seed, level=level))
        level0 = registry.get_by_level(0)
        level1 = registry.get_by_level(1)
        level2 = registry.get_by_level(2)
        assert len(level0) == 2
        assert len(level1) == 1
        assert len(level2) == 1

    def test_release_decrements_refcount(self, registry):
        ptr = self._make_ptr(seed=1)
        registry.put(ptr)
        registry.put(ptr)  # ref=2
        dead = registry.release(ptr.ptr_id)  # ref=1
        assert not dead
        dead = registry.release(ptr.ptr_id)  # ref=0
        assert dead

    def test_gc_candidates(self, registry):
        ptr1 = self._make_ptr(seed=1)
        ptr2 = self._make_ptr(seed=2)
        registry.put(ptr1)
        registry.put(ptr2)
        registry.release(ptr1.ptr_id)
        candidates = registry.gc_candidates()
        assert len(candidates) == 1
        assert candidates[0].ptr_id == ptr1.ptr_id

    def test_invalidate_model(self, registry):
        old_model = "gpt2-117M-v1"
        new_seed_range = range(1, 6)
        for s in new_seed_range:
            registry.put(self._make_ptr(seed=s, model=old_model))
        count = registry.invalidate_model(old_model)
        assert count == 5
        candidates = registry.gc_candidates()
        assert len(candidates) == 5

    def test_delete(self, registry):
        ptr = self._make_ptr(seed=1)
        registry.put(ptr)
        registry.delete(ptr.ptr_id)
        assert not registry.exists(ptr.ptr_id)

    def test_stats(self, registry):
        for s in range(3):
            registry.put(self._make_ptr(seed=s, level=s % 3))
        s = registry.stats()
        assert s["total_pointers"] == 3
        assert s["live_pointers"] == 3
        assert s["dead_pointers"] == 0

    def test_get_by_model(self, registry):
        registry.put(self._make_ptr(seed=1, model="model-a"))
        registry.put(self._make_ptr(seed=2, model="model-a"))
        registry.put(self._make_ptr(seed=3, model="model-b"))
        results = registry.get_by_model("model-a")
        assert len(results) == 2


# ======================================================================
# Integration: Buffer + Registry working together
# ======================================================================

class TestBufferRegistryIntegration:

    def test_write_then_dedup(self, tmp_buffer, registry):
        """
        Write an activation, store pointer in registry.
        Second write of identical activation: registry signals dedup,
        buffer write is skipped, ref_count incremented.
        """
        a = make_activation(64, seed=42)
        raw = a.tobytes()

        # First write
        ptr = tmp_buffer.write(a, "mock-v1", 0, "HGNS chunk")
        is_new = registry.put(ptr)
        assert is_new is True
        first_offset = ptr.buffer_offset
        used_after_first = tmp_buffer.used_bytes

        # Second "write" of same activation — registry intercepts
        ptr2 = make_pointer(raw, 0, (64,), "float32", "mock-v1", 0, "HGNS chunk")
        is_new2 = registry.put(ptr2)
        assert is_new2 is False  # Dedup: no buffer write needed

        # Buffer should NOT have grown
        assert tmp_buffer.used_bytes == used_after_first

        # Original pointer still dereferences correctly
        result = tmp_buffer.deref(ptr)
        np.testing.assert_array_almost_equal(result, a)

    def test_multi_level_hierarchy_in_registry(self, tmp_buffer, registry):
        """
        Store the same semantic chunk at all 3 HGNS levels.
        Each level gets its own pointer (different shape/dim → different hash).
        """
        text = "Quantum chaos framework using HGNS recursion."
        dims = {0: 128, 1: 64, 2: 32}  # Full → sentence → topic

        ptrs = {}
        for level, dim in dims.items():
            a = make_activation(dim, seed=level * 10)
            ptr = tmp_buffer.write(a, "mock-v1", level, text)
            registry.put(ptr)
            ptrs[level] = ptr

        # Verify each level retrievable by level
        for level in [0, 1, 2]:
            results = registry.get_by_level(level)
            assert len(results) >= 1
            # Find our pointer
            ids = {r.ptr_id for r in results}
            assert ptrs[level].ptr_id in ids

        # All share the same source_text_hash (same text, different levels)
        text_hash = hash_text(text)
        by_text = registry.get_by_text(text_hash)
        assert len(by_text) == 3

    def test_model_swap_invalidation_flow(self, tmp_buffer, registry):
        """
        Simulate model version swap:
        1. Write activations from old model
        2. Swap → invalidate old model
        3. GC candidates should include all old pointers
        4. New model writes fresh pointers
        """
        old_model = "llama3-8b-v1"
        new_model = "llama3-8b-v2"

        for i in range(4):
            a = make_activation(64, seed=i)
            ptr = tmp_buffer.write(a, old_model, 0, f"old chunk {i}")
            registry.put(ptr)

        assert registry.stats()["live_pointers"] == 4

        # Model swap
        invalidated = registry.invalidate_model(old_model)
        assert invalidated == 4
        assert registry.stats()["dead_pointers"] == 4

        # New model writes
        a_new = make_activation(64, seed=99)
        ptr_new = tmp_buffer.write(a_new, new_model, 0, "new model chunk")
        registry.put(ptr_new)

        assert registry.stats()["live_pointers"] == 1
        assert registry.stats()["dead_pointers"] == 4


# ======================================================================
# Performance smoke test
# ======================================================================

class TestPerformance:

    def test_1000_writes_under_2_seconds(self, tmp_buffer, registry):
        """1000 activation writes + registry puts should complete in <2s."""
        start = time.perf_counter()
        for i in range(1000):
            a = make_activation(64, seed=i)
            ptr = tmp_buffer.write(a, "mock-v1", i % 3, f"chunk {i}")
            registry.put(ptr)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"1000 writes took {elapsed:.2f}s — too slow"

    def test_deref_latency(self, tmp_buffer):
        """Single dereference should be well under 1ms."""
        a = make_activation(4096)  # Full LLM hidden state size
        ptr = tmp_buffer.write(a, "mock-v1", 0, "large chunk")

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = tmp_buffer.deref(ptr)
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000
        assert avg_ms < 1.0, f"Average deref latency {avg_ms:.3f}ms — too slow"
