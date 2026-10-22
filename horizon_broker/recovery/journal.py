"""Local write-ahead journal.

The journal is the broker's own durability layer. Every snapshot is appended
here *before* the EdgeRecord commit is attempted, which means a ledger outage
degrades durability (the snapshot is local-only) but never loses the ability
to resume a job.

The file format is newline-delimited JSON; one line per record.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


class LocalJournal:
    """Append-only JSONL journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_sync(self, kind: str, payload: dict[str, Any]) -> None:
        """Append one record (blocking; call from a worker thread)."""
        self._ensure_parent()
        line = json.dumps({"kind": kind, **payload}, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    async def append(self, kind: str, payload: dict[str, Any]) -> None:
        """Append one record without blocking the event loop."""
        await asyncio.to_thread(self.append_sync, kind, payload)

    def read_all_sync(self, kind: str) -> list[dict[str, Any]]:
        """Return every record of ``kind`` in write order."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("kind") == kind:
                    records.append(item)
        return records

    async def read_all(self, kind: str) -> list[dict[str, Any]]:
        """Async wrapper around :meth:`read_all_sync`."""
        return await asyncio.to_thread(self.read_all_sync, kind)

    async def latest(self, kind: str, key: str, field: str = "job_id") -> dict[str, Any] | None:
        """Return the most recent record of ``kind`` whose ``field`` equals ``key``."""
        records = await self.read_all(kind)
        matching = [item for item in records if item.get(field) == key]
        return matching[-1] if matching else None

    def trim_sync(self, max_bytes: int) -> int:
        """Drop the oldest lines until the journal fits into ``max_bytes``.

        The newest records are kept: they are the ones a recovery needs.
        Returns the number of bytes removed.
        """
        if max_bytes <= 0 or not self.path.exists():
            return 0
        size = self.path.stat().st_size
        if size <= max_bytes:
            return 0
        with self.path.open("rb") as handle:
            content = handle.read()
        kept = content[-max_bytes:]
        newline = kept.find(b"\n")
        if newline != -1:
            kept = kept[newline + 1 :]
        self.path.write_bytes(kept)
        return size - len(kept)

    async def trim(self, max_bytes: int) -> int:
        """Async wrapper around :meth:`trim_sync`."""
        return await asyncio.to_thread(self.trim_sync, max_bytes)
