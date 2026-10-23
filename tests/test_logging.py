"""Structured logging tests."""

from __future__ import annotations

import logging

import pytest

from horizon_broker.observability.logging import JsonFormatter


def test_structured_logs_reject_non_standard_json_numbers() -> None:
    record = logging.LogRecord(
        name="synthetic",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="heartbeat",
        args=(),
        exc_info=None,
    )
    record.event = {"power_headroom": float("nan")}

    with pytest.raises(ValueError, match="Out of range float values"):
        JsonFormatter().format(record)
