"""EdgeRecord ledger client.

The broker persists recovery state in EdgeRecord through a narrow HTTP port.
The exact routes are configurable so that the ledger implementation can move
without touching the scheduler or the recovery coordinator.

Expected contract:

``POST {base_url}{commit_path}``
    Body: ``{"document_id", "payload", "content_hash", "tags"}``.
    Response: ``{"document_id", "commit_url", "content_hash"}``.

``GET {base_url}{query_path}?job_id=...&limit=1``
    Response: ``{"records": [{"document_id", "payload", "content_hash"}]}``.

Every transport problem is surfaced as :class:`LedgerUnavailableError` so the
caller can fall back to the local journal.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from horizon_broker.config import LedgerConfig
from horizon_broker.contracts.errors import LedgerUnavailableError
from horizon_broker.contracts.ledger import LedgerCommitReceipt, LedgerRecord

LOGGER = logging.getLogger(__name__)


class EdgeRecordClient:
    """HTTP client for the EdgeRecord ledger."""

    def __init__(
        self,
        config: LedgerConfig,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Release the HTTP session when this client owns it."""
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def commit(self, record: LedgerRecord) -> LedgerCommitReceipt:
        """Commit a record durably and return the receipt."""
        if not self.config.enabled:
            raise LedgerUnavailableError("ledger is disabled by configuration")
        session = await self._get_session()
        url = f"{self.config.base_url.rstrip('/')}{self.config.commit_path.format(collection=record.collection)}"
        body = {
            "document_id": record.document_id,
            "payload": record.payload,
            "content_hash": record.content_hash,
            "tags": list(record.tags),
        }
        try:
            async with session.post(url, json=body) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise LedgerUnavailableError(
                        f"ledger commit rejected: HTTP {response.status} {detail[:200]}"
                    )
                try:
                    data = await response.json()
                except Exception:  # noqa: BLE001 - ledger may answer with an empty body
                    data = {}
        except LedgerUnavailableError:
            raise
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            raise LedgerUnavailableError(f"ledger commit failed: {exc}") from exc

        if not isinstance(data, dict):
            data = {}
        return LedgerCommitReceipt(
            collection=record.collection,
            document_id=str(data.get("document_id", record.document_id)),
            content_hash=str(data.get("content_hash", record.content_hash)),
            commit_url=data.get("commit_url"),
        )

    async def latest_for_job(self, job_id: str) -> LedgerRecord | None:
        """Return the newest record for ``job_id`` or ``None`` when absent."""
        if not self.config.enabled:
            raise LedgerUnavailableError("ledger is disabled by configuration")
        session = await self._get_session()
        url = (
            f"{self.config.base_url.rstrip('/')}"
            f"{self.config.query_path.format(collection=self.config.collection)}"
        )
        try:
            async with session.get(url, params={"job_id": job_id, "limit": 1}) as response:
                if response.status == 404:
                    return None
                if response.status >= 400:
                    detail = await response.text()
                    raise LedgerUnavailableError(
                        f"ledger query rejected: HTTP {response.status} {detail[:200]}"
                    )
                data = await response.json()
        except LedgerUnavailableError:
            raise
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            raise LedgerUnavailableError(f"ledger query failed: {exc}") from exc

        records = _extract_records(data)
        if not records:
            return None
        raw: dict[str, Any] = records[0]
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        return LedgerRecord(
            collection=self.config.collection,
            document_id=str(raw.get("document_id", "")),
            schema_version=str(raw.get("schema_version", "1.0")),
            payload=payload,
            content_hash=str(raw.get("content_hash", "")),
            tags=tuple(raw.get("tags", ()) or ()),
        )


def _extract_records(data: Any) -> list[dict[str, Any]]:
    """Normalise the ledger query response into a list of records."""
    if isinstance(data, dict):
        for key in ("records", "items", "results", "documents"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "document_id" in data:
            return [data]
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []
