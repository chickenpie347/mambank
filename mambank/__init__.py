"""
MemBank — Pointer-based neural activation memory for open-source LLMs.

Quick start
-----------
    from mambank import MemBank
    from mambank.adapters.mock_adapter import MockAdapter

    bank = MemBank(adapter=MockAdapter(hidden_dim=256))
    bank.ingest("The HGNS framework tames the butterfly effect.")
    results = bank.recall("HGNS chaos", top_k=3)
"""

from mambank.membank import MemBank

__version__ = "0.1.0-alpha"
__author__ = "chickenpie347"
__all__ = ["MemBank"]
