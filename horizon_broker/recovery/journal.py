"""Local write-ahead journal.

The journal is the broker's own durability layer. Every snapshot is appended
here *before* the EdgeRecord commit is attempted, which means a ledger outage
degrades durability (the snapshot is local-only) but never loses the ability
to resume a job.

The file format is newline-delimited JSON; one line per record.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn


class LocalJournal:
    """Append-only JSONL journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _file_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Coordinate journal readers and writers across broker processes."""
        self._ensure_parent()
        with self._lock_path.open("a+") as handle:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append_sync(self, kind: str, payload: dict[str, Any]) -> None:
        """Append one record (blocking; call from a worker thread)."""
        with self._write_lock:
            with self._file_lock(exclusive=True):
                line = json.dumps(
                    {**payload, "kind": kind},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
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
        with self._file_lock(exclusive=False):
            with self.path.open("rb") as handle:
                for raw_line in handle:
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        continue
                    if not line:
                        continue
                    try:
                        item = json.loads(line, parse_constant=_reject_json_constant)
                    except (json.JSONDecodeError, ValueError):
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
        with self._write_lock:
            with self._file_lock(exclusive=True):
                if max_bytes <= 0 or not self.path.exists():
                    return 0
                size = self.path.stat().st_size
                if size <= max_bytes:
                    return 0
                with self.path.open("rb") as handle:
                    content = handle.read()
                if not content.endswith(b"\n"):
                    last_complete_end = content.rfind(b"\n") + 1
                    content = content[:last_complete_end]
                kept = content[-max_bytes:]
                newline = kept.find(b"\n")
                if newline <= 0:
                    # A single record can be larger than the configured budget. Keep
                    # that complete record rather than leaving an unrecoverable suffix.
                    previous_newline = content.rfind(b"\n", 0, len(content) - 1)
                    kept = content[previous_newline + 1 :]
                else:
                    kept = kept[newline + 1 :]
                temporary_path: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=self.path.parent,
                        prefix=f".{self.path.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        temporary_path = temporary.name
                        temporary.write(kept)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    os.replace(temporary_path, self.path)
                    temporary_path = None
                finally:
                    if temporary_path is not None:
                        try:
                            os.unlink(temporary_path)
                        except FileNotFoundError:
                            pass
                return size - len(kept)

    async def trim(self, max_bytes: int) -> int:
        """Async wrapper around :meth:`trim_sync`."""
        return await asyncio.to_thread(self.trim_sync, max_bytes)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python-specific non-standard JSON numeric constants."""
    raise ValueError(f"non-standard JSON constant: {value}")
