"""Windowing policy for the Flex (sleep-time) tier.

Sleep-time compute is deliberately *slow*. The broker holds Flex jobs in an
open window so that many independent prompts can be served by one weight
load. A window opens when the first Flex job for a model arrives and closes
when any of these conditions fires:

* the window deadline (``window_seconds``) elapsed,
* the batch reached ``max_batch_size``,
* the batch reached ``min_batch_size`` *and* the minimum dwell time elapsed.

Windows are tracked per model: prompts for different models cannot share a
weight load, so mixing them would defeat the purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from horizon_broker.config import WindowConfig
from horizon_broker.contracts.jobs import InferenceJob


@dataclass(frozen=True)
class WindowPolicy:
    """Tunables of the Flex-tier window."""

    window_seconds: float = 45.0
    min_batch_size: int = 4
    max_batch_size: int = 32
    min_dwell_seconds: float = 5.0

    @classmethod
    def from_config(cls, config: WindowConfig) -> WindowPolicy:
        return cls(
            window_seconds=config.window_seconds,
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size,
            min_dwell_seconds=config.min_dwell_seconds,
        )


@dataclass
class WindowState:
    """Bookkeeping for one open window."""

    model: str
    opened_at: float
    deadline: float
    dispatched: int = 0

    @property
    def age(self) -> float:
        return self.deadline - self.opened_at

    def expired(self, now: float) -> bool:
        return now >= self.deadline

    def dwell_elapsed(self, now: float, min_dwell_seconds: float) -> bool:
        """True when the window stayed open for at least ``min_dwell_seconds``."""
        return (now - self.opened_at) >= min_dwell_seconds


@dataclass(frozen=True)
class FlexBatch:
    """A batch of Flex jobs that is ready to be dispatched."""

    model: str
    jobs: tuple[InferenceJob, ...]
    reason: str
    opened_at: float
    closed_at: float

    @property
    def size(self) -> int:
        return len(self.jobs)

    def job_ids(self) -> tuple[str, ...]:
        return tuple(job.job_id for job in self.jobs)


@dataclass(frozen=True)
class WindowDecision:
    """Outcome of evaluating one model's window."""

    should_dispatch: bool
    reason: str
    window: WindowState | None = None


def evaluate_window(
    window: WindowState,
    pending: int,
    policy: WindowPolicy,
    now: float,
) -> WindowDecision:
    """Decide whether an open window should be closed and dispatched."""
    if pending <= 0:
        return WindowDecision(False, "empty", window)
    if pending >= policy.max_batch_size:
        return WindowDecision(True, "batch_full", window)
    if window.expired(now):
        return WindowDecision(True, "window_expired", window)
    if pending >= policy.min_batch_size and window.dwell_elapsed(now, policy.min_dwell_seconds):
        return WindowDecision(True, "min_batch_reached", window)
    return WindowDecision(False, "holding", window)
