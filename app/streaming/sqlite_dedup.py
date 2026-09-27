"""Durable dedup store backed by SQLite.

Why SQLite: stdlib only, no extra dependency, works offline, easy to
inspect. For multi-worker deployments, swap in a Redis or Postgres
adapter behind the same `DedupStore` protocol -- the schema and
semantics here are a good template.

Schema:
    CREATE TABLE dedup (
      event_id TEXT PRIMARY KEY,
      seen_at  REAL NOT NULL  -- unix seconds
    );
    CREATE INDEX dedup_seen_at ON dedup(seen_at);

Concurrency: SQLite serializes writes; a single dedup store per worker
is fine. For multiple workers sharing one DB, use WAL mode (set below)
and small transactions.

TTL: `prune(now)` deletes rows older than `ttl_s`. Callers should prune
periodically (e.g. from the ingestor's main loop). If never pruned, the
DB grows -- acceptable for a portfolio demo, not for production.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dedup (
    event_id TEXT PRIMARY KEY,
    seen_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS dedup_seen_at ON dedup(seen_at);
"""


class SQLiteDedupStore:
    def __init__(
        self,
        path: str | Path,
        *,
        ttl_s: float = 86_400.0,  # 24h default
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        self._path = str(path)
        self._ttl_s = ttl_s

        # `check_same_thread=False` is fine: the ingestor is single-task.
        # Callers sharing the store across threads must serialize access.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_if_new(self, event_id: str) -> bool:
        """Insert if absent. Return True if inserted (new), False if present."""
        now = time.time()
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO dedup(event_id, seen_at) VALUES (?, ?)",
            (event_id, now),
        )
        self._conn.commit()
        return cur.rowcount == 1  # 1 if inserted, 0 if ignored

    def prune(self, now: float | None = None) -> int:
        """Delete rows older than ttl. Return number of rows deleted."""
        cutoff = (now if now is not None else time.time()) - self._ttl_s
        cur = self._conn.execute("DELETE FROM dedup WHERE seen_at < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            log.debug("sqlite_dedup.close_failed", exc_info=True)

    # ---- introspection (tests) ----

    @property
    def size(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM dedup")
        (n,) = cur.fetchone()
        return int(n)

    def __contains__(self, event_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM dedup WHERE event_id = ? LIMIT 1", (event_id,)
        )
        return cur.fetchone() is not None
