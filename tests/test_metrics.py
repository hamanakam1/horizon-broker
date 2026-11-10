"""The metrics registry: honest series, real quantiles."""

from __future__ import annotations

import math

import pytest

from horizon_broker.observability.metrics import MetricsRegistry


def test_a_summary_exports_quantiles() -> None:
    """The runbook asks for p95 recovery time; count and sum cannot answer it."""
    registry = MetricsRegistry()
    for value in (0.1, 0.2, 0.3, 4.0):
        registry.observe("horizon_recovery_seconds", value)

    rendered = registry.render()
    assert 'horizon_recovery_seconds{quantile="0.5"}' in rendered
    assert 'horizon_recovery_seconds{quantile="0.95"}' in rendered
    assert 'horizon_recovery_seconds{quantile="0.99"}' in rendered
    assert "horizon_recovery_seconds_count 4" in rendered
    assert "horizon_recovery_seconds_sum" in rendered


def test_a_declared_summary_is_never_rendered_as_a_counter() -> None:
    """A metric has one type: a summary must not also appear as a counter."""
    registry = MetricsRegistry()
    registry.describe("horizon_recovery_seconds", "Recovery time in seconds.")
    registry.declare_summary("horizon_recovery_seconds")

    rendered = registry.render()
    assert "# TYPE horizon_recovery_seconds summary" in rendered
    assert "horizon_recovery_seconds 0.0" not in rendered


def test_a_labelled_metric_has_no_bare_series() -> None:
    """One name, one shape: no unlabelled zero next to the labelled series."""
    registry = MetricsRegistry()
    registry.describe("horizon_flex_batches_total", "Flex batches dispatched.")
    registry.incr("horizon_flex_batches_total", reason="fleet_idle", model="qwen2.5:7b")

    rendered = registry.render()
    assert 'horizon_flex_batches_total{model="qwen2.5:7b",reason="fleet_idle"} 1.0' in rendered
    assert "\nhorizon_flex_batches_total " not in rendered


def test_the_counter_sums_every_label_combination() -> None:
    registry = MetricsRegistry()
    registry.incr("horizon_jobs_submitted_total", tier="flex")
    registry.incr("horizon_jobs_submitted_total", tier="flex")
    registry.incr("horizon_jobs_submitted_total", tier="interactive")
    assert registry.counter("horizon_jobs_submitted_total") == 3.0
    assert registry.counter("horizon_jobs_submitted_total", tier="flex") == 2.0


@pytest.mark.parametrize("method", ["declare", "incr", "set_gauge", "observe"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 10**1000, True, False])
def test_metric_recording_rejects_non_finite_values(method: str, value: float) -> None:
    registry = MetricsRegistry()

    with pytest.raises(ValueError, match="finite"):
        getattr(registry, method)("horizon_invalid", value)
