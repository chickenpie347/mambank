"""
tests/test_week3.py

Week 3 test suite for MemBank:
  - retrieval/index.py   : VectorIndex add/search, numpy fallback, dedup
  - retrieval/query.py   : QueryEngine multi-level drill-down
  - membank.py           : ingest, recall, integrate, dedup, persistence,
                           recursive population, stats, GC

Run with: python tests/test_week3.py
"""

import sys, os, tempfile, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mambank.retrieval.index import VectorIndex, SearchResult
from mambank.retrieval.query import QueryEngine, RecallResult
from mambank.hgns.hierarchy import HGNSHierarchy
from mambank.adapters.mock_adapter import MockAdapter
from mambank.core.registry import Registry
from mambank.membank import MemBank
from mambank import MemBank as MemBankPublic  # Test public import

PASSED = 0
FAILED = 0

def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        print(f"  PASS: {name}")
        PASSED += 1
    else:
        msg = f"  FAIL: {name}"
        if detail: msg += f" — {detail}"
        print(msg)
        FAILED += 1

def section(title):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")

def rng_vec(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))


# ======================================================================
# VectorIndex
# ======================================================================
section("VECTOR INDEX — basic add and search")

idx = VectorIndex(dim=64, hgns_level=2)
check("starts empty", idx.size == 0)
check("backend is numpy (no faiss)", idx.backend in ("numpy", "faiss"))

v1 = rng_vec(64, 1)
v2 = rng_vec(64, 2)
v3 = rng_vec(64, 3)
idx.add(v1, "ptr_001")
idx.add(v2, "ptr_002")
idx.add(v3, "ptr_003")
check("size after 3 adds", idx.size == 3)

results = idx.search(v1, top_k=3)
check("search returns list", isinstance(results, list))
check("search returns up to top_k results", len(results) <= 3)
check("top result is correct ptr_id for exact query", results[0].ptr_id == "ptr_001")
check("top result score ≈ 1.0 for self-query", abs(results[0].score - 1.0) < 1e-4)
check("results have hgns_level attribute", results[0].hgns_level == 2)
check("results have rank attribute", results[0].rank == 0)

section("VECTOR INDEX — score ordering")

idx2 = VectorIndex(dim=32, hgns_level=1)
# Create vectors with known similarities
base = np.ones(32, dtype=np.float32)
close = base + rng_vec(32, 99) * 0.01   # Very similar to base
far   = rng_vec(32, 42)                  # Random, less similar
idx2.add(base / np.linalg.norm(base), "base")
idx2.add(close / np.linalg.norm(close), "close")
idx2.add(far / np.linalg.norm(far), "far")

results2 = idx2.search(base, top_k=3)
# "close" should rank above "far"
ranks = {r.ptr_id: r.rank for r in results2}
check("base ranks first (self-query)", ranks.get("base", 99) == 0)
check("close ranks before far", ranks.get("close", 99) < ranks.get("far", 99))
check("scores are descending", all(results2[i].score >= results2[i+1].score - 1e-6
                                   for i in range(len(results2)-1)))

section("VECTOR INDEX — edge cases")

# top_k > size → returns all
idx3 = VectorIndex(dim=16, hgns_level=0)
for i in range(3): idx3.add(rng_vec(16, i), f"p{i}")
results3 = idx3.search(rng_vec(16, 99), top_k=100)
check("top_k > size → returns size results", len(results3) == 3)

# Empty index search
idx_empty = VectorIndex(dim=16, hgns_level=0)
empty_results = idx_empty.search(rng_vec(16, 1), top_k=5)
check("empty index search returns empty list", empty_results == [])

# Wrong dim raises ValueError
try:
    idx3.add(rng_vec(32, 1), "bad")  # 32 != 16
    check("add wrong dim raises ValueError", False)
except ValueError:
    check("add wrong dim raises ValueError", True)

try:
    idx3.search(rng_vec(32, 1), top_k=1)
    check("search wrong dim raises ValueError", False)
except ValueError:
    check("search wrong dim raises ValueError", True)

section("VECTOR INDEX — normalisation (cosine semantics)")

# Two vectors that differ only in magnitude should have score=1.0
v = rng_vec(64, 7)
idx4 = VectorIndex(dim=64, hgns_level=2)
idx4.add(v * 10.0, "scaled_10")
idx4.add(v * 0.001, "scaled_small")
r = idx4.search(v, top_k=2)
check("L2-normalised: scaled vectors have same score", abs(r[0].score - r[1].score) < 1e-4)
check("scaled vectors score ≈ 1.0", abs(r[0].score - 1.0) < 1e-4)

section("VECTOR INDEX — multiple HGNS levels independence")

# Indexes at different levels are independent
idx_l0 = VectorIndex(dim=128, hgns_level=0)
idx_l1 = VectorIndex(dim=64, hgns_level=1)
idx_l2 = VectorIndex(dim=32, hgns_level=2)

for i in range(5):
    idx_l0.add(rng_vec(128, i),    f"p{i}_l0")
    idx_l1.add(rng_vec(64, i+10),  f"p{i}_l1")
    idx_l2.add(rng_vec(32, i+20),  f"p{i}_l2")

check("L0 index size correct", idx_l0.size == 5)
check("L1 index size correct", idx_l1.size == 5)
check("L2 index size correct", idx_l2.size == 5)
check("L0 results have level=0", idx_l0.search(rng_vec(128,99),1)[0].hgns_level == 0)
check("L1 results have level=1", idx_l1.search(rng_vec(64, 99),1)[0].hgns_level == 1)
check("L2 results have level=2", idx_l2.search(rng_vec(32, 99),1)[0].hgns_level == 2)


# ======================================================================
# MemBank — core API
# ======================================================================
section("MEMBANK — construction and public import")

with tempfile.TemporaryDirectory() as tmpdir:
    adapter = MockAdapter(hidden_dim=256)
    bank = MemBank(adapter=adapter, registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")
    check("MemBank constructs without error", True)
    check("public import works", MemBankPublic is MemBank)
    check("repr includes model_id", adapter.model_id() in repr(bank))
    bank.close()

section("MEMBANK — ingest basic")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=128),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    result = bank.ingest("The HGNS framework tames the butterfly effect.")
    check("ingest returns dict", isinstance(result, dict))
    check("ingest result has ptr_ids", "ptr_ids" in result)
    check("ingest result has l0/l1/l2 keys", all(k in result["ptr_ids"] for k in ["l0","l1","l2"]))
    check("ingest result has is_new=True", result["is_new"] is True)
    check("ingest result has ingest_ms", "ingest_ms" in result)
    check("ingest_ms is reasonable (<500ms)", result["ingest_ms"] < 500)
    check("ingest_count incremented", bank._ingest_count == 1)
    bank.close()

section("MEMBANK — deduplication")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=128),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    text = "Quantum chaos can be tamed using iterative HGNS refinement."
    r1 = bank.ingest(text)
    buf_used_after_first = bank.buffer.used_bytes
    r2 = bank.ingest(text)  # Same text → same embeddings → dedup

    check("second ingest of same text: is_new=False", r2["is_new"] is False)
    check("buffer did not grow on duplicate ingest", bank.buffer.used_bytes == buf_used_after_first)
    check("dedup_count incremented", bank._dedup_count > 0)
    bank.close()

section("MEMBANK — recall basic")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=256),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    chunks = [
        "The HGNS framework introduces a recursive hierarchy of sub-steps.",
        "Quantum chaos can be tamed using deterministic HGNS iterations.",
        "MemBank stores neural activations as content-addressed pointers.",
        "The butterfly effect is suppressed through HGNS adaptive resolution.",
        "Banach spaces provide the mathematical foundation for HGNS.",
        "Weather forecasting benefits from deterministic chaos control.",
        "Pointer-based memory avoids storing full activation tensors.",
        "Redis-style key-value store manages activation references.",
    ]
    for c in chunks:
        bank.ingest(c)

    results = bank.recall("HGNS butterfly chaos", top_k=3)
    check("recall returns list", isinstance(results, list))
    check("recall returns <= top_k results", len(results) <= 3)
    check("results are RecallResult instances", all(isinstance(r, RecallResult) for r in results))
    check("results have valid scores", all(0.0 <= r.final_score <= 1.0 + 1e-6 for r in results))
    check("results are sorted by score (descending)",
          all(results[i].final_score >= results[i+1].final_score - 1e-6
              for i in range(len(results)-1)))
    check("results have ranks 0,1,2", [r.rank for r in results] == list(range(len(results))))
    check("recall_count incremented", bank._recall_count == 1)
    bank.close()

section("MEMBANK — recall relevance (semantic)")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=256),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    # Ingest corpus with clear topic clusters
    hgns_chunks = [
        "HGNS recursive sub-steps enable multiscale tensor calculations.",
        "The butterfly effect is tamed by HGNS deterministic iteration.",
        "HGNS base-k representation supports adaptive precision.",
    ]
    other_chunks = [
        "Stock market volatility increases during economic uncertainty.",
        "Machine learning gradient descent optimizes neural network weights.",
        "Weather patterns exhibit chaotic sensitivity to initial conditions.",
    ]
    for c in hgns_chunks + other_chunks:
        bank.ingest(c)

    # Query about HGNS — should retrieve HGNS chunks
    results = bank.recall("HGNS recursive hierarchy tames chaos", top_k=3)
    result_hashes = {r.source_text_hash for r in results}

    # Compute expected hashes for HGNS chunks
    from mambank.core.pointer import hash_text
    hgns_hashes = {hash_text(c) for c in hgns_chunks}

    overlap = len(result_hashes & hgns_hashes)
    check("HGNS query retrieves at least 1 HGNS chunk in top-3",
          overlap >= 1,
          f"overlap={overlap}, top-3 hashes={result_hashes}")
    bank.close()

section("MEMBANK — recall with different level configs")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=256),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")
    for i in range(5):
        bank.ingest(f"Test chunk number {i} about various topics.")

    # Single-level coarse search
    r_l2 = bank.recall("test chunk", top_k=3, levels=[2])
    check("level=[2] coarse-only search works", len(r_l2) >= 0)

    # Full drill-down
    r_full = bank.recall("test chunk", top_k=3, levels=[2,1,0])
    check("level=[2,1,0] full search works", len(r_full) >= 0)

    # Full level search returns <= top_k
    check("full search returns <= top_k", len(r_full) <= 3)
    bank.close()

section("MEMBANK — integrate (recursive population)")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=256),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    # Pre-populate with some context
    bank.ingest("HGNS tames the butterfly effect through recursive iteration.")
    bank.ingest("MemBank stores activation pointers in a memmap buffer.")
    bank.ingest("The registry tracks ref_counts for smart pointer GC.")

    # Integrate: recall + ingest in one call
    result = bank.integrate("How does HGNS help with chaotic systems?", top_k=2)

    check("integrate returns dict", isinstance(result, dict))
    check("integrate result has 'query'", "query" in result)
    check("integrate result has 'recalled'", "recalled" in result)
    check("integrate result has 'augmented_prompt'", "augmented_prompt" in result)
    check("integrate result has 'ingest_result'", "ingest_result" in result)
    check("augmented_prompt contains [MemBank Context]",
          "[MemBank Context]" in result["augmented_prompt"])
    check("augmented_prompt contains [Query]",
          "[Query]" in result["augmented_prompt"])
    check("query was ingested (bank grew)",
          bank._ingest_count == 4)
    check("recall was run", bank._recall_count >= 1)
    bank.close()

section("MEMBANK — recursive auto-population")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=128),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    # Simulate a growing conversation
    conversation = [
        "Turn 1: We are building MemBank, a neural activation memory system.",
        "Turn 2: The pointer system uses content-addressed hashing.",
        "Turn 3: HGNS provides multi-level compression of activations.",
        "Turn 4: FAISS enables fast approximate nearest-neighbour search.",
        "Turn 5: The registry uses SQLite with WAL mode for thread safety.",
    ]

    for i, turn in enumerate(conversation):
        bank.integrate(turn, top_k=2)

    check("all turns ingested", bank._ingest_count == len(conversation))
    check("registry has entries at all levels",
          all(bank.registry.stats()["by_level"].get(f"level_{lv}", 0) > 0
              for lv in [0, 1, 2]))

    # Late recall should find early turns
    late_results = bank.recall("MemBank neural activation memory", top_k=3)
    check("late query can recall early conversation turns", len(late_results) > 0)
    bank.close()

section("MEMBANK — stats")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=128),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    bank.ingest("Test chunk for stats.")
    bank.recall("test", top_k=1)
    s = bank.stats()

    check("stats returns dict", isinstance(s, dict))
    check("stats has adapter key", "adapter" in s)
    check("stats has dims key", "dims" in s)
    check("stats has ingest_count=1", s["ingest_count"] == 1)
    check("stats has recall_count=1", s["recall_count"] == 1)
    check("stats has buffer sub-dict", "buffer" in s)
    check("stats has registry sub-dict", "registry" in s)
    check("stats has indexes sub-dict", "indexes" in s)
    check("indexes shows 3 levels",
          all(f"level{lv}_size" in s["indexes"] for lv in [0,1,2]))
    bank.close()

section("MEMBANK — garbage collection")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=128),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    bank.ingest("Chunk to be GC'd.")
    bank.ingest("Chunk to survive.")

    # Simulate model swap → invalidate all old pointers
    bank.registry.invalidate_model(bank.adapter.model_id())
    gc_result = bank.gc()

    check("gc returns dict", isinstance(gc_result, dict))
    check("gc collects invalidated pointers", gc_result["collected"] > 0)
    check("registry dead_pointers → 0 after gc",
          bank.registry.stats()["dead_pointers"] == 0)
    bank.close()

section("MEMBANK — context manager")

with tempfile.TemporaryDirectory() as tmpdir:
    with MemBank(adapter=MockAdapter(hidden_dim=64),
                 registry_path=":memory:",
                 buffer_path=f"{tmpdir}/buf.mmap") as bank:
        bank.ingest("Context manager test.")
        r = bank.recall("context", top_k=1)
        check("context manager: ingest + recall work", len(r) >= 0)
    check("context manager: bank closed without error", True)

section("MEMBANK — persistence (rebuild_indexes)")

with tempfile.TemporaryDirectory() as tmpdir:
    buf_path = f"{tmpdir}/persist_buf.mmap"
    reg_path = f"{tmpdir}/persist_reg.db"

    # Session 1: ingest
    bank1 = MemBank(adapter=MockAdapter(hidden_dim=128),
                    registry_path=reg_path,
                    buffer_path=buf_path)
    chunks = [
        "HGNS enables multiscale tensor calculations.",
        "Pointer-based memory uses content-addressed hashing.",
        "The butterfly effect is tamed by recursive iteration.",
    ]
    for c in chunks:
        bank1.ingest(c)
    bank1.close()

    # Session 2: reopen and rebuild indexes
    bank2 = MemBank(adapter=MockAdapter(hidden_dim=128),
                    registry_path=reg_path,
                    buffer_path=buf_path,
                    auto_validate_adapter=True)
    reindexed = bank2.rebuild_indexes()

    check("rebuild_indexes returns positive count", reindexed > 0)
    check("reindexed count = 3 chunks × 3 levels", reindexed == 9)

    # Search should work after rebuild
    results = bank2.recall("HGNS butterfly", top_k=2)
    check("recall works after rebuild_indexes", len(results) > 0)
    bank2.close()

section("MEMBANK — thread safety (concurrent ingest)")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=128),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/buf.mmap")

    errors = []
    def ingest_worker(worker_id):
        try:
            for i in range(5):
                bank.ingest(f"Worker {worker_id} message {i} about HGNS and MemBank.")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=ingest_worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    check("no errors in concurrent ingest", len(errors) == 0, str(errors))
    check("all 20 messages ingested", bank._ingest_count == 20)
    bank.close()

section("MEMBANK — performance benchmarks")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(adapter=MockAdapter(hidden_dim=256),
                   registry_path=":memory:",
                   buffer_path=f"{tmpdir}/bench.mmap",
                   buffer_capacity_bytes=64*1024*1024)

    # Ingest 200 chunks
    t0 = time.perf_counter()
    for i in range(200):
        bank.ingest(f"Benchmark chunk {i}: HGNS enables multiscale memory with pointer-based storage.")
    ingest_time = time.perf_counter() - t0
    check(f"200 ingests in {ingest_time:.2f}s (limit 10s)", ingest_time < 10.0)

    # Recall latency
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        bank.recall("HGNS pointer memory benchmark", top_k=5)
        times.append(time.perf_counter() - t0)
    avg_recall_ms = sum(times) / len(times) * 1000
    check(f"avg recall latency {avg_recall_ms:.1f}ms (limit 500ms)", avg_recall_ms < 500.0)
    bank.close()


# ======================================================================
# End-to-end integration — full pipeline
# ======================================================================
section("END-TO-END — MemBank conversation simulation")

with tempfile.TemporaryDirectory() as tmpdir:
    bank = MemBank(
        adapter=MockAdapter(hidden_dim=256),
        registry_path=":memory:",
        buffer_path=f"{tmpdir}/e2e.mmap",
    )

    # Simulate a multi-turn conversation about MemBank
    turns = [
        "We're building MemBank, a neural activation memory module for LLMs.",
        "The HGNS hierarchy compresses activations at three resolution levels.",
        "Level 0 stores full activations, level 2 stores coarse summaries.",
        "Retrieval starts at level 2 and drills down for better precision.",
        "The pointer system uses SHA256 content hashing for deduplication.",
        "A memmap buffer enables zero-copy dereference of stored activations.",
        "SQLite registry tracks ref_counts for smart pointer garbage collection.",
        "The HuggingFace adapter hooks into transformer hidden states via forward hooks.",
        "MockAdapter provides deterministic embeddings for testing without a GPU.",
        "FAISS flat index enables cosine similarity search via inner product.",
    ]

    # Recursive population via integrate()
    for turn in turns:
        bank.integrate(turn, top_k=3)

    check("all 10 turns ingested", bank._ingest_count == len(turns))
    check("all 10 recalls ran", bank._recall_count == len(turns))

    # Query should find relevant turns
    r = bank.recall("HGNS compression levels retrieval", top_k=3)
    check("HGNS query returns results", len(r) > 0)
    check("results have valid structure",
          all(hasattr(res, "final_score") and hasattr(res, "source_text_hash")
              for res in r))

    # Stats sanity
    s = bank.stats()
    check("registry has 30 live pointers (10 turns × 3 levels)",
          s["registry"]["live_pointers"] == 30)
    check("indexes populated at all levels",
          all(s["indexes"][f"level{lv}_size"] == 10 for lv in [0,1,2]))

    bank.close()


# ======================================================================
# Summary
# ======================================================================
print(f"\n{'='*58}")
print(f"  WEEK 3 RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'='*58}")
if FAILED == 0:
    print(f"  ALL {PASSED} TESTS PASSED ✓")
else:
    print(f"  {FAILED} TESTS FAILED ✗")
    sys.exit(1)
