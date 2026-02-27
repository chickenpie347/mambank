"""
mambank/core/registry.py

Registry — SQLite-backed metadata store for PointerRecords.

Responsibilities
----------------
1. Persist PointerRecords indexed by ptr_id (content hash).
2. Deduplication: if a ptr_id already exists, increment ref_count
   rather than creating a duplicate buffer entry.
3. Reverse lookup: source_text_hash → ptr_id (find pointer by text).
4. HGNS level filtering: query all pointers at a given resolution level.
5. Model invalidation: flag/tombstone pointers when model_id changes.
6. GC support: return all pointers with ref_count <= 0.

Schema
------
Table: pointers
  ptr_id          TEXT PRIMARY KEY   -- SHA256 content hash
  json_record     TEXT               -- Full PointerRecord as JSON
  model_id        TEXT               -- Indexed for invalidation queries
  hgns_level      INTEGER            -- Indexed for level-filtered queries
  source_text_hash TEXT              -- Indexed for reverse lookup
  ref_count       INTEGER            -- Cached here for fast GC queries
  timestamp       REAL               -- For TTL / LRU eviction
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from mambank.core.pointer import PointerRecord


class Registry:
    """
    SQLite-backed store for PointerRecord metadata.

    Thread-safe via a per-instance lock (SQLite WAL mode for
    concurrent readers, serialised writers).

    Parameters
    ----------
    path : str | Path
        Path to SQLite database file. Use ":memory:" for tests.
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pointers (
                ptr_id           TEXT PRIMARY KEY,
                json_record      TEXT NOT NULL,
                model_id         TEXT NOT NULL,
                hgns_level       INTEGER NOT NULL,
                source_text_hash TEXT NOT NULL,
                ref_count        INTEGER NOT NULL DEFAULT 1,
                timestamp        REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_model_id
                ON pointers (model_id);

            CREATE INDEX IF NOT EXISTS idx_hgns_level
                ON pointers (hgns_level);

            CREATE INDEX IF NOT EXISTS idx_source_text_hash
                ON pointers (source_text_hash);

            CREATE INDEX IF NOT EXISTS idx_ref_count
                ON pointers (ref_count);
        """)
        self._conn.commit()

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def put(self, record: PointerRecord) -> bool:
        """
        Insert a PointerRecord, or increment ref_count if already present.

        Parameters
        ----------
        record : PointerRecord

        Returns
        -------
        bool
            True  → new record inserted (buffer write was necessary)
            False → duplicate detected, ref_count incremented (no buffer write needed)
        """
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT ref_count FROM pointers WHERE ptr_id = ?",
                (record.ptr_id,),
            ).fetchone()

            if existing is not None:
                # Deduplication: just bump ref count
                conn.execute(
                    "UPDATE pointers SET ref_count = ref_count + 1 WHERE ptr_id = ?",
                    (record.ptr_id,),
                )
                return False  # Caller: skip buffer write
            else:
                conn.execute(
                    """INSERT INTO pointers
                       (ptr_id, json_record, model_id, hgns_level,
                        source_text_hash, ref_count, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.ptr_id,
                        record.to_json(),
                        record.model_id,
                        record.hgns_level,
                        record.source_text_hash,
                        record.ref_count,
                        record.timestamp,
                    ),
                )
                return True  # Caller: proceed with buffer write

    def release(self, ptr_id: str) -> bool:
        """
        Decrement ref_count. Returns True if ref_count hit 0 (GC eligible).
        """
        with self._tx() as conn:
            row = conn.execute(
                "SELECT ref_count FROM pointers WHERE ptr_id = ?", (ptr_id,)
            ).fetchone()
            if row is None:
                return False
            new_count = row[0] - 1
            conn.execute(
                "UPDATE pointers SET ref_count = ? WHERE ptr_id = ?",
                (new_count, ptr_id),
            )
            return new_count <= 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, ptr_id: str) -> Optional[PointerRecord]:
        """Fetch a single PointerRecord by ptr_id, with live ref_count from DB."""
        row = self._conn.execute(
            "SELECT json_record, ref_count FROM pointers WHERE ptr_id = ?", (ptr_id,)
        ).fetchone()
        if row is None:
            return None
        record = PointerRecord.from_json(row[0])
        record.ref_count = row[1]  # Always use live DB value, not stale JSON
        return record

    def _to_records(self, rows) -> List[PointerRecord]:
        """Deserialize (json_record, ref_count) rows with live ref_count."""
        result = []
        for json_str, live_ref in rows:
            r = PointerRecord.from_json(json_str)
            r.ref_count = live_ref
            result.append(r)
        return result

    def get_by_text(self, source_text_hash: str) -> List[PointerRecord]:
        """Reverse lookup: find all pointers produced from a given text chunk."""
        rows = self._conn.execute(
            "SELECT json_record, ref_count FROM pointers WHERE source_text_hash = ?",
            (source_text_hash,),
        ).fetchall()
        return self._to_records(rows)

    def get_by_level(self, hgns_level: int) -> List[PointerRecord]:
        """Return all live pointers at a given HGNS resolution level."""
        rows = self._conn.execute(
            "SELECT json_record, ref_count FROM pointers WHERE hgns_level = ? AND ref_count > 0",
            (hgns_level,),
        ).fetchall()
        return self._to_records(rows)

    def get_by_model(self, model_id: str) -> List[PointerRecord]:
        """Return all pointers produced by a specific model."""
        rows = self._conn.execute(
            "SELECT json_record, ref_count FROM pointers WHERE model_id = ?", (model_id,)
        ).fetchall()
        return self._to_records(rows)

    def exists(self, ptr_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM pointers WHERE ptr_id = ?", (ptr_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Model invalidation
    # ------------------------------------------------------------------

    def invalidate_model(self, model_id: str) -> int:
        """
        Zero-out ref_counts for all pointers from a stale model.
        Returns count of invalidated records.

        Call this when the model is swapped to a new version.
        The buffer slots become GC-eligible but are NOT freed here —
        call gc_collect() separately.
        """
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE pointers SET ref_count = 0 WHERE model_id = ?",
                (model_id,),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Garbage collection support
    # ------------------------------------------------------------------

    def gc_candidates(self) -> List[PointerRecord]:
        """Return all pointers with ref_count <= 0 (buffer slots to reclaim)."""
        rows = self._conn.execute(
            "SELECT json_record, ref_count FROM pointers WHERE ref_count <= 0"
        ).fetchall()
        return self._to_records(rows)

    def delete(self, ptr_id: str) -> None:
        """Hard-delete a pointer record. Call after reclaiming its buffer slot."""
        with self._tx() as conn:
            conn.execute("DELETE FROM pointers WHERE ptr_id = ?", (ptr_id,))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        row = self._conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN ref_count > 0 THEN 1 ELSE 0 END) as live,
                SUM(CASE WHEN ref_count <= 0 THEN 1 ELSE 0 END) as dead,
                COUNT(DISTINCT model_id) as models
               FROM pointers"""
        ).fetchone()
        level_rows = self._conn.execute(
            "SELECT hgns_level, COUNT(*) FROM pointers GROUP BY hgns_level"
        ).fetchall()
        return {
            "total_pointers": row[0],
            "live_pointers": row[1],
            "dead_pointers": row[2],
            "distinct_models": row[3],
            "by_level": {f"level_{r[0]}": r[1] for r in level_rows},
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"Registry(path={self.path!r}, "
            f"live={s['live_pointers']}, dead={s['dead_pointers']})"
        )
