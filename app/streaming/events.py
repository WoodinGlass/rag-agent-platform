"""Event envelope for the streaming pipeline.

Why a wrapper and not raw bytes:
- Producers carry an id (for dedup / offset tracking).
- Producers carry metadata (source, tenant, ingestion ts).
- Consumers get a stable shape to ack/nack, regardless of broker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class Event:
    id: str
    data: bytes
    metadata: dict = field(default_factory=dict)
    received_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
