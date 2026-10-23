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
    An empty collection means the record is absent; a malformed envelope is
    unavailable ledger state and raises :class:`LedgerUnavailableError`.

Every transport problem is surfaced as :class:`LedgerUnavailableError` so the
caller can fall back to the local journal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, NoReturn
from urllib.parse import quote, urlsplit

import aiohttp

from horizon_broker.config import LedgerConfig
from horizon_broker.contracts.errors import LedgerUnavailableError
from horizon_broker.contracts.ledger import SCHEMA_VERSION, LedgerCommitReceipt, LedgerRecord

LOGGER = logging.getLogger(__name__)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python-specific non-standard JSON numeric constants."""
    raise ValueError(f"non-standard JSON constant: {value}")


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

    def _auth_headers(self) -> dict[str, str]:
        """Build the auth headers from the environment.

        The token is read per request so that it can be rotated without
        restarting the broker.
        """
        token = os.environ.get(self.config.auth_token_env, "").strip()
        if not token:
            return {}
        scheme = self.config.auth_scheme.strip()
        value = f"{scheme} {token}" if scheme else token
        return {self.config.auth_header: value}

    async def close(self) -> None:
        """Release the HTTP session when this client owns it."""
        if self._session is not None and self._owns_session:
            session = self._session
            try:
                await session.close()
            finally:
                self._session = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def commit(self, record: LedgerRecord) -> LedgerCommitReceipt:
        """Commit a record durably and return the receipt.

        Transient failures (network errors, 5xx) are retried with a growing
        backoff. A 4xx is permanent and is raised immediately.
        """
        if not self.config.enabled:
            raise LedgerUnavailableError("ledger is disabled by configuration")
        if not record.verify():
            raise LedgerUnavailableError("ledger record content_hash does not match its payload")
        session = await self._get_session()
        url = _ledger_url(self.config.base_url, self.config.commit_path, record.collection)
        body = {
            "document_id": record.document_id,
            "payload": record.payload,
            "content_hash": record.content_hash,
            "tags": _with_owner_tag(record.tags, self.config.owner_id),
        }
        data: dict[str, Any] = {}
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.post(
                    url,
                    json=body,
                    headers=self._auth_headers(),
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    if response.status >= 500:
                        detail = await response.text()
                        last_error = LedgerUnavailableError(
                            f"ledger commit failed: HTTP {response.status} {detail[:200]}"
                        )
                    elif response.status >= 400:
                        detail = await response.text()
                        raise LedgerUnavailableError(
                            f"ledger commit rejected: HTTP {response.status} {detail[:200]}"
                        )
                    elif response.status >= 300:
                        detail = await response.text()
                        raise LedgerUnavailableError(
                            f"ledger commit redirected: HTTP {response.status} {detail[:200]}"
                        )
                    else:
                        try:
                            parsed = json.loads(
                                await response.text(), parse_constant=_reject_json_constant
                            )
                        except Exception as exc:  # noqa: BLE001 - preserve the ledger error taxonomy
                            raise LedgerUnavailableError(
                                "ledger commit returned invalid JSON"
                            ) from exc
                        if not isinstance(parsed, dict):
                            raise LedgerUnavailableError(
                                "ledger commit returned a non-object response"
                            )
                        returned_document_id = parsed.get("document_id")
                        if not isinstance(returned_document_id, str) or not returned_document_id.strip():
                            raise LedgerUnavailableError(
                                "ledger commit returned no document_id"
                            )
                        if returned_document_id != returned_document_id.strip():
                            raise LedgerUnavailableError(
                                "ledger commit returned a non-canonical document_id"
                            )
                        if (
                            returned_document_id != record.document_id
                        ):
                            raise LedgerUnavailableError(
                                "ledger commit returned a mismatched document_id"
                            )
                        returned_content_hash = parsed.get("content_hash")
                        if returned_content_hash is not None:
                            if (
                                not isinstance(returned_content_hash, str)
                                or not returned_content_hash.strip()
                            ):
                                raise LedgerUnavailableError(
                                    "ledger commit returned an invalid content_hash"
                                )
                            if returned_content_hash != record.content_hash:
                                raise LedgerUnavailableError(
                                    "ledger commit returned a mismatched content_hash"
                                )
                        commit_url = parsed.get("commit_url")
                        if (
                            commit_url is not None
                            and (
                                not isinstance(commit_url, str)
                                or not _is_http_url(commit_url)
                            )
                        ):
                            raise LedgerUnavailableError(
                                "ledger commit returned an invalid commit_url"
                            )
                        data = parsed
                        break
            except LedgerUnavailableError:
                raise
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                last_error = LedgerUnavailableError(f"ledger commit failed: {exc}")
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt))
        else:
            raise last_error or LedgerUnavailableError("ledger commit failed")

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
        url = _ledger_url(self.config.base_url, self.config.query_path, self.config.collection)
        data: Any = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.get(
                    url,
                    params={
                        "job_id": job_id,
                        "owner_id": self.config.owner_id,
                        "limit": 1,
                    },
                    headers=self._auth_headers(),
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    if response.status == 404:
                        return None
                    if response.status >= 500:
                        detail = await response.text()
                        last_error: Exception = LedgerUnavailableError(
                            f"ledger query failed: HTTP {response.status} {detail[:200]}"
                        )
                    elif response.status >= 400:
                        detail = await response.text()
                        raise LedgerUnavailableError(
                            f"ledger query rejected: HTTP {response.status} {detail[:200]}"
                        )
                    elif response.status >= 300:
                        detail = await response.text()
                        raise LedgerUnavailableError(
                            f"ledger query redirected: HTTP {response.status} {detail[:200]}"
                        )
                    else:
                        try:
                            data = json.loads(
                                await response.text(), parse_constant=_reject_json_constant
                            )
                        except Exception as exc:  # noqa: BLE001 - normalize decode failures
                            raise LedgerUnavailableError(
                                "ledger query returned invalid JSON"
                            ) from exc
                        break
            except LedgerUnavailableError:
                raise
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                last_error = LedgerUnavailableError(f"ledger query failed: {exc}")
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_backoff_seconds * (2**attempt))
        else:
            raise last_error or LedgerUnavailableError("ledger query failed")

        records = _extract_records(data)
        if not records:
            return None
        raw: dict[str, Any] = records[0]
        returned_collection = raw.get("collection")
        if returned_collection is not None:
            if not isinstance(returned_collection, str) or not returned_collection.strip():
                raise LedgerUnavailableError("ledger query returned an invalid collection")
            if returned_collection != self.config.collection:
                raise LedgerUnavailableError(
                    "ledger query returned a record from a different collection"
                )
        document_id = raw.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise LedgerUnavailableError(
                "ledger query returned a record without a document_id"
            )
        if document_id != document_id.strip():
            raise LedgerUnavailableError(
                "ledger query returned a non-canonical document_id"
            )
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise LedgerUnavailableError(
                "ledger query returned a record without a payload"
            )
        returned_job_id = payload.get("job_id")
        if returned_job_id != job_id:
            raise LedgerUnavailableError(
                "ledger query returned a record for a different job"
            )
        returned_content_hash = raw.get("content_hash")
        if returned_content_hash is not None:
            if not isinstance(returned_content_hash, str) or not returned_content_hash.strip():
                raise LedgerUnavailableError("ledger query returned an invalid content_hash")
            try:
                expected_content_hash = LedgerRecord.build(
                    collection=self.config.collection,
                    document_id=document_id,
                    payload=payload,
                ).content_hash
            except (TypeError, ValueError) as exc:
                raise LedgerUnavailableError(
                    "ledger query returned an unserializable payload"
                ) from exc
            if returned_content_hash != expected_content_hash:
                raise LedgerUnavailableError(
                    "ledger query returned a mismatched content_hash"
                )
        schema_version = raw.get("schema_version", SCHEMA_VERSION)
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise LedgerUnavailableError(
                "ledger query returned an invalid schema_version"
            )
        if schema_version != SCHEMA_VERSION:
            raise LedgerUnavailableError(
                f"ledger query returned unsupported schema_version {schema_version!r}"
            )
        tags = raw.get("tags", ())
        if tags is None:
            tags = ()
        if not isinstance(tags, list | tuple) or not all(
            isinstance(tag, str) and tag.strip() and tag == tag.strip() for tag in tags
        ):
            raise LedgerUnavailableError("ledger query returned invalid tags")
        if not isinstance(returned_content_hash, str) or not returned_content_hash.strip():
            raise LedgerUnavailableError("ledger query returned an invalid content_hash")
        owner_tag = f"broker:{self.config.owner_id}"
        owner_tags = [tag for tag in tags if tag.startswith("broker:")]
        if owner_tag not in tags or any(tag != owner_tag for tag in owner_tags):
            raise LedgerUnavailableError(
                "ledger query returned a record for a different broker"
            )
        try:
            return LedgerRecord(
                collection=self.config.collection,
                document_id=document_id,
                schema_version=schema_version,
                payload=payload,
                content_hash=returned_content_hash,
                tags=tuple(tags),
            )
        except (TypeError, ValueError) as exc:
            raise LedgerUnavailableError(
                "ledger query returned an invalid record envelope"
            ) from exc


def _extract_records(data: Any) -> list[dict[str, Any]]:
    """Normalize a query envelope without mistaking malformed data for absence."""
    if isinstance(data, dict):
        for key in ("records", "items", "results", "documents"):
            if key not in data:
                continue
            value = data[key]
            if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                raise LedgerUnavailableError(
                    f"ledger query returned a malformed {key} collection"
                )
            return list(value)
        if "document_id" in data:
            return [data]
        raise LedgerUnavailableError("ledger query returned an unrecognized object")
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise LedgerUnavailableError("ledger query returned a malformed collection")
        return list(data)
    raise LedgerUnavailableError("ledger query returned an unsupported JSON value")


def _is_http_url(value: str) -> bool:
    """Accept only absolute HTTP(S) receipt URLs from the remote ledger."""
    if value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    try:
        valid_port = parsed.port is None or 1 <= parsed.port <= 65_535
    except ValueError:
        valid_port = False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and valid_port
    )


def _ledger_url(base_url: str, route_template: str, collection: str) -> str:
    """Build a ledger URL without allowing collection text to alter its path."""
    encoded_collection = quote(collection, safe="")
    return f"{base_url.rstrip('/')}{route_template.format(collection=encoded_collection)}"


def _with_owner_tag(tags: tuple[str, ...], owner_id: str) -> list[str]:
    """Keep recovery records attributable when several brokers share a ledger."""
    owner_tag = f"broker:{owner_id}"
    conflicting = [tag for tag in tags if tag.startswith("broker:") and tag != owner_tag]
    if conflicting:
        raise LedgerUnavailableError(
            "ledger record carries a conflicting broker ownership tag"
        )
    return [*tags, owner_tag] if owner_tag not in tags else list(tags)
