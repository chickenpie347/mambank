"""
benchmarks/benchmark.py

MemBank Benchmark Suite
=======================

Measures four dimensions of performance:

1. INGEST THROUGHPUT   — chunks/second at various corpus sizes
2. RECALL LATENCY      — p50/p95/p99 ms at various corpus sizes
3. MEMORY EFFICIENCY   — bytes/activation at each HGNS level
4. RECALL ACCURACY     — precision@k vs naive full-dim cosine RAG

All benchmarks run with MockAdapter (deterministic, CPU-only) so
results are reproducible without GPU or model weights.

Run:
    python benchmarks/benchmark.py

Optional flags (edit SETTINGS below):
    CORPUS_SIZES     — list of corpus sizes to test
    HIDDEN_DIM       — embedding dimension (128=fast, 768=gpt2-scale)
    TOP_K            — recall depth
    N_RECALL_QUERIES — number of queries for latency/accuracy stats
"""

from __future__ import annotations

import sys
import os
import tempfile
import time
import gc as python_gc
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mambank import MemBank
from mambank.adapters.mock_adapter import MockAdapter
from mambank.core.pointer import hash_text

# ======================================================================
# Settings
# ======================================================================

SETTINGS = {
    "CORPUS_SIZES":      [50, 200, 500],   # number of chunks to ingest
    "HIDDEN_DIM":        256,               # embedding dimension
    "TOP_K":             5,                 # recall depth
    "N_RECALL_QUERIES":  30,                # queries for latency stats
    "HGNS_K":            10,
    "HGNS_GRADIENT_LEVELS": 4,
}

# ======================================================================
# Helpers
# ======================================================================

CORPUS_TEMPLATES = [
    "The HGNS framework introduces a recursive hierarchy between integer pairs.",
    "Quantum chaos can be tamed using deterministic HGNS iterative refinement.",
    "MemBank stores neural activations as content-addressed pointer records.",
    "The butterfly effect is suppressed via HGNS adaptive resolution sub-steps.",
    "Banach spaces provide the mathematical foundation for HGNS transformations.",
    "Weather forecasting benefits from deterministic chaos control frameworks.",
    "Pointer-based memory avoids storing full activation tensors on disk.",
    "Redis-style key-value stores manage activation references efficiently.",
    "FAISS enables fast approximate nearest-neighbour search on embeddings.",
    "SQLite with WAL mode provides thread-safe registry for pointer metadata.",
    "The butterfly effect emerges when small perturbations grow exponentially.",
    "Content-addressed hashing enables automatic deduplication at ingest time.",
    "Multi-level HGNS compression reduces storage by over 90% at coarse level.",
    "Forward hooks on transformer layers capture hidden states non-invasively.",
    "Mean-pooling across token positions gives a fixed-size activation vector.",
    "Coarse-to-fine retrieval minimises expensive full-vector comparisons.",
    "The registry tracks ref_counts for garbage collection of dead pointers.",
    "Memory-mapped files enable zero-copy dereference of activation tensors.",
    "HGNS base-k representation emphasises hierarchical multiscale precision.",
    "Open-source LLMs benefit from plug-and-play persistent memory modules.",
]

def make_corpus(n: int) -> List[str]:
    """Generate n unique text chunks by cycling through templates."""
    chunks = []
    for i in range(n):
        template = CORPUS_TEMPLATES[i % len(CORPUS_TEMPLATES)]
        # Vary slightly to ensure unique embeddings
        chunks.append(f"[{i:04d}] {template}")
    return chunks

def percentiles(values: List[float], ps=(50, 95, 99)):
    arr = sorted(values)
    results = {}
    for p in ps:
        idx = max(0, int(len(arr) * p / 100) - 1)
        results[f"p{p}"] = arr[idx]
    return results

def fmt_bytes(n: int) -> str:
    if n < 1024: return f"{n}B"
    if n < 1024**2: return f"{n/1024:.1f}KB"
    return f"{n/1024**2:.2f}MB"

def banner(title: str):
    w = 62
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def row(label: str, value: str, pad: int = 38):
    print(f"  {label:<{pad}} {value}")

# ======================================================================
# Benchmark 1: Ingest Throughput
# ======================================================================

def bench_ingest_throughput(hidden_dim: int, corpus_sizes: List[int]) -> dict:
    banner("BENCHMARK 1 — Ingest Throughput (chunks/sec)")
    results = {}
    for n in corpus_sizes:
        corpus = make_corpus(n)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank = MemBank(
                adapter=MockAdapter(hidden_dim=hidden_dim),
                registry_path=":memory:",
                buffer_path=f"{tmpdir}/buf.mmap",
                buffer_capacity_bytes=64 * 1024 * 1024,
            )
            t0 = time.perf_counter()
            for chunk in corpus:
                bank.ingest(chunk)
            elapsed = time.perf_counter() - t0

            cps = n / elapsed
            ms_per = elapsed / n * 1000
            results[n] = {"elapsed_s": elapsed, "chunks_per_sec": cps, "ms_per_chunk": ms_per}
            row(f"corpus={n:>4} chunks", f"{cps:>8.1f} chunks/s  ({ms_per:.2f} ms/chunk)")
            bank.close()
    return results

# ======================================================================
# Benchmark 2: Recall Latency
# ======================================================================

def bench_recall_latency(hidden_dim: int, corpus_sizes: List[int],
                          top_k: int, n_queries: int) -> dict:
    banner(f"BENCHMARK 2 — Recall Latency (top_k={top_k})")
    results = {}

    # Build query set from corpus templates
    adapter = MockAdapter(hidden_dim=hidden_dim)
    query_texts = [
        "HGNS butterfly effect deterministic chaos",
        "pointer activation memory deduplication",
        "quantum chaos tamed recursive iteration",
        "FAISS retrieval cosine similarity search",
        "transformer hidden states forward hook",
    ] * (n_queries // 5 + 1)
    query_texts = query_texts[:n_queries]

    for n in corpus_sizes:
        corpus = make_corpus(n)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank = MemBank(
                adapter=MockAdapter(hidden_dim=hidden_dim),
                registry_path=":memory:",
                buffer_path=f"{tmpdir}/buf.mmap",
                buffer_capacity_bytes=64 * 1024 * 1024,
            )
            for chunk in corpus:
                bank.ingest(chunk)

            # Warm up
            bank.recall(query_texts[0], top_k=top_k)

            latencies = []
            for q in query_texts:
                t0 = time.perf_counter()
                bank.recall(q, top_k=top_k)
                latencies.append((time.perf_counter() - t0) * 1000)

            ps = percentiles(latencies)
            results[n] = ps
            row(f"corpus={n:>4} chunks",
                f"p50={ps['p50']:.1f}ms  p95={ps['p95']:.1f}ms  p99={ps['p99']:.1f}ms")
            bank.close()
    return results

# ======================================================================
# Benchmark 3: Memory Efficiency
# ======================================================================

def bench_memory_efficiency(hidden_dim: int, n: int = 200) -> dict:
    banner("BENCHMARK 3 — Memory Efficiency (bytes per activation)")
    corpus = make_corpus(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        bank = MemBank(
            adapter=MockAdapter(hidden_dim=hidden_dim),
            registry_path=":memory:",
            buffer_path=f"{tmpdir}/buf.mmap",
        )
        for chunk in corpus:
            bank.ingest(chunk)

        dims = bank._dims
        buf_used = bank.buffer.used_bytes
        reg_stats = bank.registry.stats()
        idx_stats = bank.query_engine.stats()

        # Bytes per raw activation (float32)
        bytes_l0 = dims["level0"] * 4
        bytes_l1 = dims["level1"] * 4
        bytes_l2 = dims["level2"] * 4

        # Actual buffer usage per chunk (3 levels stored)
        avg_buf_bytes = buf_used / max(n, 1)

        results = {
            "dims": dims,
            "bytes_l0": bytes_l0,
            "bytes_l1": bytes_l1,
            "bytes_l2": bytes_l2,
            "compression_l1": bytes_l1 / bytes_l0,
            "compression_l2": bytes_l2 / bytes_l0,
            "buf_used_total": buf_used,
            "avg_buf_bytes_per_chunk": avg_buf_bytes,
            "n_chunks": n,
        }

        row("Full dim (Level 0)",    f"{dims['level0']} dims  →  {fmt_bytes(bytes_l0)}/activation")
        row("Sentence dim (Level 1)",f"{dims['level1']} dims  →  {fmt_bytes(bytes_l1)}/activation  ({results['compression_l1']*100:.0f}% of L0)")
        row("Topic dim (Level 2)",   f"{dims['level2']} dims  →  {fmt_bytes(bytes_l2)}/activation  ({results['compression_l2']*100:.0f}% of L0)")
        row("Buffer used (total)",   f"{fmt_bytes(buf_used)} for {n} chunks (3 levels each)")
        row("Avg per chunk",         f"{fmt_bytes(int(avg_buf_bytes))} (vs {fmt_bytes(bytes_l0*3)} naive 3×L0)")
        row("Registry live pointers",f"{reg_stats['live_pointers']}")

        bank.close()
    return results

# ======================================================================
# Benchmark 4: Recall Accuracy vs Naive RAG
# ======================================================================

def bench_recall_accuracy(hidden_dim: int, n: int = 200, top_k: int = 5,
                           n_queries: int = 20) -> dict:
    banner(f"BENCHMARK 4 — Recall Accuracy: MemBank vs Naive RAG (top_k={top_k})")

    adapter = MockAdapter(hidden_dim=hidden_dim)
    corpus = make_corpus(n)
    corpus_embs = adapter.encode_batch(corpus)

    # Build ground truth: for each query, rank all corpus entries by
    # full-dim cosine similarity (this IS the naive RAG baseline)
    def naive_topk(query_emb: np.ndarray, k: int) -> List[int]:
        sims = corpus_embs @ (query_emb / np.linalg.norm(query_emb))
        return list(np.argsort(sims)[-k:][::-1])

    queries = [
        "HGNS butterfly effect deterministic chaos control recursive",
        "pointer content-addressed memory buffer storage activation",
        "quantum chaos tamed recursive Banach space iteration",
        "FAISS retrieval cosine similarity nearest neighbour search",
        "transformer forward hook hidden state mean pooling encoder",
        "multiscale tensor calculation gradient approximation level",
        "memory deduplication SHA256 hash ref count garbage collect",
        "open source LLM persistent memory module plug-in adapter",
        "SQLite WAL registry metadata pointer invalidation model swap",
        "weather forecasting chaos sensitivity initial conditions",
    ] * (n_queries // 10 + 1)
    queries = queries[:n_queries]

    with tempfile.TemporaryDirectory() as tmpdir:
        bank = MemBank(
            adapter=MockAdapter(hidden_dim=hidden_dim),
            registry_path=":memory:",
            buffer_path=f"{tmpdir}/acc.mmap",
            buffer_capacity_bytes=64*1024*1024,
        )
        corpus_hashes = []
        for chunk in corpus:
            bank.ingest(chunk)
            corpus_hashes.append(hash_text(chunk))

        precision_at_k_values = []
        recall_at_k_values = []

        for q_text in queries:
            q_emb = adapter.encode(q_text)

            # Ground truth (naive full-dim cosine)
            gt_indices = naive_topk(q_emb, top_k)
            gt_hashes = {corpus_hashes[i] for i in gt_indices}

            # MemBank results
            mb_results = bank.recall(q_text, top_k=top_k)
            mb_hashes = {r.source_text_hash for r in mb_results}

            # Precision@k: what fraction of MemBank results are in GT?
            hit_count = len(mb_hashes & gt_hashes)
            precision = hit_count / max(len(mb_hashes), 1)
            recall_val = hit_count / max(len(gt_hashes), 1)

            precision_at_k_values.append(precision)
            recall_at_k_values.append(recall_val)

        avg_precision = np.mean(precision_at_k_values)
        avg_recall = np.mean(recall_at_k_values)

        results = {
            "avg_precision_at_k": avg_precision,
            "avg_recall_at_k": avg_recall,
            "n_queries": n_queries,
            "corpus_size": n,
            "top_k": top_k,
        }

        row("Corpus size",           f"{n} chunks")
        row("Queries evaluated",     f"{n_queries}")
        row("Naive RAG baseline",    f"100% precision (it IS the ground truth)")
        row(f"MemBank Precision@{top_k}", f"{avg_precision*100:.1f}%  (vs naive full-dim cosine)")
        row(f"MemBank Recall@{top_k}",   f"{avg_recall*100:.1f}%")
        row("Interpretation",       f"{'✓ Strong' if avg_precision >= 0.6 else '~ Acceptable' if avg_precision >= 0.4 else '✗ Weak'} agreement with naive RAG")

        bank.close()
    return results

# ======================================================================
# Benchmark 5: HGNS Level Contribution
# ======================================================================

def bench_hgns_level_contribution(hidden_dim: int, n: int = 200, top_k: int = 5) -> dict:
    banner("BENCHMARK 5 — HGNS Level Contribution to Recall Quality")

    adapter_full = MockAdapter(hidden_dim=hidden_dim)
    corpus = make_corpus(n)

    queries = [
        "HGNS butterfly effect deterministic chaos",
        "pointer memory storage activation buffer",
        "quantum chaos recursive Banach space",
        "FAISS retrieval cosine similarity",
        "transformer hidden state mean pooling",
    ]

    # Ground truth: full-dim naive cosine
    corpus_embs = adapter_full.encode_batch(corpus)
    corpus_hashes = [hash_text(c) for c in corpus]

    def naive_topk_hashes(q_text, k):
        q = adapter_full.encode(q_text)
        sims = corpus_embs @ (q / (np.linalg.norm(q) + 1e-10))
        idxs = list(np.argsort(sims)[-k:][::-1])
        return {corpus_hashes[i] for i in idxs}

    level_configs = {
        "L2 only (coarse)":   [2],
        "L2+L1 (mid)":        [2, 1],
        "L2+L1+L0 (full)":    [2, 1, 0],
    }

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        bank = MemBank(
            adapter=MockAdapter(hidden_dim=hidden_dim),
            registry_path=":memory:",
            buffer_path=f"{tmpdir}/lvl.mmap",
            buffer_capacity_bytes=64*1024*1024,
        )
        for chunk in corpus:
            bank.ingest(chunk)

        for config_name, levels in level_configs.items():
            precisions = []
            for q in queries:
                gt = naive_topk_hashes(q, top_k)
                mb = {r.source_text_hash for r in bank.recall(q, top_k=top_k, levels=levels)}
                precisions.append(len(mb & gt) / max(len(mb), 1))
            avg = np.mean(precisions)
            results[config_name] = avg
            row(config_name, f"{avg*100:.1f}% precision@{top_k}")

        bank.close()
    return results

# ======================================================================
# Main runner
# ======================================================================

def main():
    s = SETTINGS
    dim = s["HIDDEN_DIM"]
    sizes = s["CORPUS_SIZES"]
    top_k = s["TOP_K"]
    n_q = s["N_RECALL_QUERIES"]

    print(f"\n{'█'*62}")
    print(f"  MemBank™ Benchmark Suite")
    print(f"  hidden_dim={dim}  top_k={top_k}  numpy backend (no FAISS)")
    print(f"{'█'*62}")

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    r1 = bench_ingest_throughput(dim, sizes)
    r2 = bench_recall_latency(dim, sizes, top_k, n_q)
    r3 = bench_memory_efficiency(dim, n=sizes[-1])
    r4 = bench_recall_accuracy(dim, n=sizes[-1], top_k=top_k, n_queries=n_q)
    r5 = bench_hgns_level_contribution(dim, n=sizes[-1], top_k=top_k)

    # Summary table
    banner("SUMMARY")
    largest = sizes[-1]
    row("Ingest throughput",      f"{r1[largest]['chunks_per_sec']:.0f} chunks/s at corpus={largest}")
    row("Recall p50 latency",     f"{r2[largest]['p50']:.1f} ms at corpus={largest}")
    row("Recall p99 latency",     f"{r2[largest]['p99']:.1f} ms at corpus={largest}")
    row("L1 compression ratio",   f"{r3['compression_l1']*100:.0f}% of full dim")
    row("L2 compression ratio",   f"{r3['compression_l2']*100:.0f}% of full dim")
    row(f"Recall accuracy@{top_k}",f"{r4['avg_precision_at_k']*100:.1f}% precision vs naive RAG")
    row("Full HGNS drill-down",   f"{r5.get('L2+L1+L0 (full)', 0)*100:.1f}% precision vs coarse-only {r5.get('L2 only (coarse)', 0)*100:.1f}%")

    print(f"\n{'═'*62}")
    print(f"  Benchmark complete.")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
