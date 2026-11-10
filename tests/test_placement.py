"""Node placement: quarantine expiry and per-node power gating."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from horizon_broker.contracts.nodes import PowerClass
from horizon_broker.contracts.recovery import utcnow
from horizon_broker.routing.arithmetic import analyse, profile_for
from horizon_broker.routing.placement import NodeSelector
from tests.conftest import make_job, make_node

MODEL = "qwen2.5:7b"


def _selector() -> NodeSelector:
    return NodeSelector(
        power_gated_classes=(PowerClass.SOLAR_TRAILER, PowerClass.BATTERY),
        power_headroom_threshold=0.25,
    )


def test_a_node_inside_its_quarantine_is_not_eligible() -> None:
    node = make_node("pi-01").model_copy(
        update={"quarantined_until": utcnow() + timedelta(seconds=60)}
    )
    assert _selector().eligible([node], MODEL) == []


def test_placement_rejects_a_naive_clock() -> None:
    with pytest.raises(ValueError, match="placement timestamp must be timezone-aware"):
        _selector().eligible([make_node("pi-01")], MODEL, now=datetime(2026, 10, 23, 12, 0, 0))


def test_a_node_whose_quarantine_expired_is_eligible_again() -> None:
    """The quarantine is a cooling-off period, not a life sentence.

    ``recovery.quarantine_seconds`` is a documented setting; a node that
    served out its time must be able to take work again.
    """
    node = make_node("pi-01").model_copy(
        update={"quarantined_until": utcnow() - timedelta(seconds=1)}
    )
    eligible = _selector().eligible([node], MODEL)
    assert [candidate.node_id for candidate in eligible] == ["pi-01"]


def test_a_power_gated_node_below_the_threshold_is_skipped_for_flex_work() -> None:
    """Power gating is per node: the trailer is skipped, the grid node is not."""
    trailer = make_node(
        "trailer-01",
        power_class=PowerClass.SOLAR_TRAILER,
        power_headroom=0.05,
    )
    grid = make_node("m3-max", power_headroom=1.0)
    eligible = _selector().eligible([trailer, grid], MODEL, flex_tier=True)
    assert [candidate.node_id for candidate in eligible] == ["m3-max"]


def test_a_power_gated_node_above_the_threshold_still_takes_flex_work() -> None:
    trailer = make_node(
        "trailer-01",
        power_class=PowerClass.SOLAR_TRAILER,
        power_headroom=0.8,
    )
    eligible = _selector().eligible([trailer], MODEL, flex_tier=True)
    assert [candidate.node_id for candidate in eligible] == ["trailer-01"]


def test_a_power_gated_node_below_the_threshold_may_still_take_interactive_work() -> None:
    """A human is waiting: power gating only holds back sleep-time compute."""
    trailer = make_node(
        "trailer-01",
        power_class=PowerClass.SOLAR_TRAILER,
        power_headroom=0.05,
    )
    eligible = _selector().eligible([trailer], MODEL, flex_tier=False)
    assert [candidate.node_id for candidate in eligible] == ["trailer-01"]


def test_the_selector_picks_the_node_with_headroom_for_flex_work() -> None:
    trailer = make_node(
        "trailer-01",
        power_class=PowerClass.SOLAR_TRAILER,
        power_headroom=0.05,
    )
    grid = make_node("m3-max", power_headroom=1.0)
    job = make_job(model=MODEL)
    profile = analyse(job.payload)
    chosen = _selector().select(
        [trailer, grid],
        profile,
        profile_for(MODEL),
        flex_tier=True,
    )
    assert chosen is not None
    assert chosen.node_id == "m3-max"
