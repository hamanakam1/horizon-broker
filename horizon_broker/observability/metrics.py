"""In-process metrics.

A tiny Prometheus-compatible registry: no third-party dependency, no
scrape endpoint surprises. The broker exposes the rendered text at
``GET /metrics``.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

_LABEL_SAFE = str.maketrans({c: "_" for c in '"\\\n'})


class MetricsRegistry:
    """Counters, gauges and histograms with optional labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._help: dict[str, str] = {}

    # -- declaration --------------------------------------------------------

    def describe(self, name: str, text: str) -> None:
        """Attach a help string to a metric."""
        self._help[name] = text

    # -- recording ----------------------------------------------------------

    def declare(self, name: str, value: float = 0.0, **labels: Any) -> None:
        """Pre-register a counter so it is exposed even before the first event."""
        with self._lock:
            key = _key(name, labels)
            self._counters.setdefault(key, value)

    def declare_summary(self, name: str, **labels: Any) -> None:
        """Pre-register a summary (a histogram) so it is exposed when empty.

        A summary must never also appear as a counter: Prometheus gives a
        metric one type, and two types under one name is a scrape error.
        """
        with self._lock:
            key = _key(name, labels)
            self._histograms.setdefault(key, [])

    def incr(self, name: str, value: float = 1.0, **labels: Any) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def set_gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._histograms[_key(name, labels)].append(value)

    def counter(self, name: str, **labels: Any) -> float:
        """Read a counter.

        Without labels the values of every label combination are summed; with
        labels only the combinations that carry all of them are summed.
        """
        with self._lock:
            if not labels:
                return sum(value for key, value in self._counters.items() if key.split("{")[0] == name)
            total = 0.0
            for key, value in self._counters.items():
                metric_name, metric_labels = _parse_key(key)
                if metric_name != name:
                    continue
                if all(metric_labels.get(label) == expected for label, expected in labels.items()):
                    total += value
            return total

    def histogram(self, name: str) -> list[float]:
        """Every observation of ``name``, across all label combinations."""
        with self._lock:
            observations: list[float] = []
            for key, values in self._histograms.items():
                metric_name, _ = _split(key)
                if metric_name == name:
                    observations.extend(values)
            return observations

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        """Render the registry in the Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            counters: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for key, value in self._counters.items():
                name, label_text = _split(key)
                counters[name].append((label_text, value))
            for name, samples in counters.items():
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} counter")
                for label_text, value in samples:
                    lines.append(f"{name}{label_text} {value}")

            gauges: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for key, value in self._gauges.items():
                name, label_text = _split(key)
                gauges[name].append((label_text, value))
            for name, samples in gauges.items():
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} gauge")
                for label_text, value in samples:
                    lines.append(f"{name}{label_text} {value}")

            histograms: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
            for key, observations in self._histograms.items():
                name, label_text = _split(key)
                histograms[name].append((label_text, observations))
            for name, entries in histograms.items():
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} summary")
                for label_text, observations in entries:
                    lines.append(f"{name}_count{label_text} {len(observations)}")
                    lines.append(f"{name}_sum{label_text} {sum(observations):.6f}")
                    # The runbook asks for percentiles, not just a mean: a
                    # sum cannot answer "did the recovery stay under 30 s".
                    for quantile, value in _quantiles(observations):
                        lines.append(
                            f"{name}{_with_quantile(label_text, quantile)} {value:.6f}"
                        )
        return "\n".join(lines) + "\n"


def _key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    rendered = ",".join(
        f'{key}="{str(value).translate(_LABEL_SAFE)}"' for key, value in sorted(labels.items())
    )
    return f"{name}{{{rendered}}}"


def _split(key: str) -> tuple[str, str]:
    if "{" in key:
        name, _, rest = key.partition("{")
        return name, "{" + rest
    return key, ""


def _parse_key(key: str) -> tuple[str, dict[str, str]]:
    """Split a metric key into its name and its label mapping."""
    name, label_text = _split(key)
    if not label_text:
        return name, {}
    labels: dict[str, str] = {}
    for pair in label_text.strip("{}").split(","):
        if not pair:
            continue
        key_name, _, value = pair.partition("=")
        labels[key_name.strip()] = value.strip().strip('"')
    return name, labels


QUANTILES = (("0.5", 0.5), ("0.95", 0.95), ("0.99", 0.99))
"""The percentiles the broker publishes for every summary."""


def _quantiles(observations: list[float]) -> list[tuple[str, float]]:
    """Return the (label, value) pairs of the published percentiles."""
    if not observations:
        return []
    ordered = sorted(observations)
    result: list[tuple[str, float]] = []
    for label, quantile in QUANTILES:
        index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
        result.append((label, ordered[index]))
    return result


def _with_quantile(label_text: str, quantile: str) -> str:
    """Add a ``quantile`` label to an existing label set."""
    addition = f'quantile="{quantile}"'
    inner = label_text.strip("{}")
    if inner:
        return "{" + f"{inner},{addition}" + "}"
    return "{" + addition + "}"
