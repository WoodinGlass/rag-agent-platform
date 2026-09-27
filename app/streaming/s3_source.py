"""S3 event source: poll a bucket for new objects.

Lazy import so `boto3` is only required when streaming_backend is set
to "s3". Missing package -> clear ImportError.

Design notes:
- We do not use SQS/SNS notifications here. Polling `list_objects_v2`
  is simpler, works with any S3-compatible service (MinIO, R2, B2),
  and gives the same at-least-once semantics our ingestor already
  handles.
- Commit is in-memory (a set of processed keys). This is deliberate
  for the MVP: if the worker restarts, it re-lists the bucket, and the
  ingestor's dedup window absorbs the redelivery. Persisting a state
  file to S3 is a documented follow-up.
- Blocking boto3 calls (list/get) run in `asyncio.to_thread` so the
  event loop stays responsive.
- `LastModified` is not consulted; we treat any not-yet-committed key
  as new. Sorted iteration by key keeps tests deterministic.
"""
from __future__ import annotations

import asyncio
import io
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.streaming.events import Event

log = get_logger(__name__)


@dataclass
class S3Config:
    bucket: str
    prefix: str = ""
    endpoint_url: str = ""
    region: str = ""
    max_keys: int = 1000
    poll_interval_s: float = 5.0


class S3ObjectSource:
    """EventSource over S3 objects.

    Events:
        id   = "s3://<bucket>/<key>"
        data = raw object bytes
        meta = {bucket, key, size, etag}
    """

    def __init__(
        self,
        config: S3Config,
        *,
        s3_client: Any | None = None,
    ) -> None:
        if not config.bucket:
            raise ValueError("S3Config.bucket is required")
        self._config = config
        self._client = s3_client if s3_client is not None else self._build_client()

        # Keys to process (FIFO), keys already emitted in this worker,
        # and keys explicitly committed by the ingestor.
        #
        # `_emitted` prevents re-listing an object we've already handed
        # to the ingestor but that hasn't been committed yet. Without it,
        # `_refresh()` would enqueue the same key forever.
        #
        # Growth: bounded in practice by bucket size. Persisting this to
        # S3 (state file) is a documented follow-up.
        self._pending: deque[str] = deque()
        self._emitted: set[str] = set()
        self._committed: set[str] = set()
        self._last_list_at: float = 0.0
        self._closed = False

    # ---- client ----

    def _build_client(self) -> Any:
        try:
            import boto3
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "boto3 not installed; run: pip install 'rag-agent-platform[s3]'"
            ) from e
        kwargs: dict[str, Any] = {}
        if self._config.endpoint_url:
            kwargs["endpoint_url"] = self._config.endpoint_url
        if self._config.region:
            kwargs["region_name"] = self._config.region
        return boto3.client("s3", **kwargs)

    # ---- listing ----

    async def _refresh(self) -> None:
        """List objects and enqueue unseen keys. Bounded by poll interval."""
        now = time.monotonic()
        if now - self._last_list_at < self._config.poll_interval_s and self._pending:
            return
        self._last_list_at = now

        resp = await asyncio.to_thread(
            self._client.list_objects_v2,
            Bucket=self._config.bucket,
            Prefix=self._config.prefix,
            MaxKeys=self._config.max_keys,
        )
        contents = resp.get("Contents") or []
        # Deterministic order for tests and stable demos.
        for obj in sorted(contents, key=lambda o: o.get("Key", "")):
            key = obj.get("Key")
            if not key:
                continue
            if (
                key in self._committed
                or key in self._emitted
                or key in self._pending
            ):
                continue
            self._pending.append(key)

        if self._pending:
            log.info(
                "s3.listed",
                extra={"bucket": self._config.bucket, "queued": len(self._pending)},
            )

    # ---- EventSource protocol ----

    async def poll(self, timeout_s: float = 1.0) -> Event | None:
        if self._closed:
            return None

        if not self._pending:
            try:
                await self._refresh()
            except Exception:
                log.warning("s3.list_failed", exc_info=True)
                await asyncio.sleep(min(timeout_s, 0.5))
                return None

        if not self._pending:
            # Nothing to do; yield and return None so the ingestor idles.
            await asyncio.sleep(min(timeout_s, 0.5))
            return None

        key = self._pending.popleft()
        # Mark as emitted now: if fetch fails we requeue, but if fetch
        # succeeds we must not re-list this key next refresh.
        self._emitted.add(key)
        try:
            body = await asyncio.to_thread(self._fetch_body, key)
        except Exception as e:  # noqa: BLE001 -- source boundary
            log.warning(
                "s3.get_failed", extra={"key": key, "error": str(e)[:200]}
            )
            # Re-queue once so a transient error can be retried on next poll.
            self._pending.appendleft(key)
            await asyncio.sleep(min(timeout_s, 0.5))
            return None

        return Event(
            id=f"s3://{self._config.bucket}/{key}",
            data=body,
            metadata={
                "bucket": self._config.bucket,
                "key": key,
                "size": len(body),
            },
        )

    def _fetch_body(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._config.bucket, Key=key)
        body = obj["Body"]
        # boto3 uses StreamingBody; tests may inject io.BytesIO.
        if hasattr(body, "read"):
            data = body.read()
        else:  # pragma: no cover - defensive
            data = bytes(body)
        return data if isinstance(data, bytes) else bytes(data)

    async def commit(self, event_id: str) -> None:
        # event_id is "s3://<bucket>/<key>"; store the key.
        prefix = f"s3://{self._config.bucket}/"
        if event_id.startswith(prefix):
            key = event_id[len(prefix):]
        else:
            key = event_id
        self._committed.add(key)

    async def close(self) -> None:
        self._closed = True

    # ---- test helpers ----

    @property
    def committed_keys(self) -> set[str]:
        return set(self._committed)

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def _bytesio(data: bytes) -> io.BytesIO:  # pragma: no cover - test helper
    return io.BytesIO(data)
