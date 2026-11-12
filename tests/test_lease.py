"""The fleet lease: one broker owns a fleet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from horizon_broker.control.lease import FleetLease


def test_the_first_holder_owns_the_fleet(tmp_path: Path) -> None:
    lease = FleetLease(tmp_path / "fleet.lease", owner_id="broker-a")
    try:
        assert lease.acquire() is True
        assert lease.is_held is True
        assert lease.holder() == "broker-a"
    finally:
        lease.release()


def test_a_second_broker_cannot_take_a_held_lease(tmp_path: Path) -> None:
    """The fleet has one owner; the attempt never blocks."""
    path = tmp_path / "fleet.lease"
    owner = FleetLease(path, owner_id="broker-a")
    follower = FleetLease(path, owner_id="broker-b")
    try:
        assert owner.acquire() is True
        assert follower.acquire() is False
        assert follower.is_held is False
        assert follower.holder() == "broker-a"
    finally:
        owner.release()
        follower.release()


def test_the_lease_is_free_again_after_a_release(tmp_path: Path) -> None:
    path = tmp_path / "fleet.lease"
    owner = FleetLease(path, owner_id="broker-a")
    follower = FleetLease(path, owner_id="broker-b")
    try:
        assert owner.acquire() is True
        owner.release()
        assert follower.acquire() is True
        assert follower.holder() == "broker-b"
    finally:
        follower.release()


def test_a_dying_broker_does_not_leave_a_stale_lease(tmp_path: Path) -> None:
    """The operating system releases the lock when the process dies.

    A broker that crashes cannot hold the fleet for ever: the kernel drops
    the lock with the descriptor, so a follower takes over without waiting
    for a timeout.
    """
    path = tmp_path / "fleet.lease"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from horizon_broker.control.lease import FleetLease\n"
        f"lease = FleetLease({str(path)!r}, owner_id='broker-crashed')\n"
        "assert lease.acquire() is True\n"
        "print('acquired', flush=True)\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stderr

    survivor = FleetLease(path, owner_id="broker-b")
    try:
        assert survivor.acquire() is True
        assert survivor.holder() == "broker-b"
    finally:
        survivor.release()


def test_renewing_a_lease_keeps_the_owner_visible(tmp_path: Path) -> None:
    lease = FleetLease(tmp_path / "fleet.lease", owner_id="broker-a")
    try:
        assert lease.acquire() is True
        assert lease.renew() is True
        assert lease.holder() == "broker-a"
    finally:
        lease.release()


def test_renewing_a_lease_we_do_not_hold_is_false(tmp_path: Path) -> None:
    lease = FleetLease(tmp_path / "fleet.lease", owner_id="broker-a")
    assert lease.renew() is False
    assert lease.is_held is False


def test_the_lease_works_as_a_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "fleet.lease"
    with FleetLease(path, owner_id="broker-a") as lease:
        assert lease.is_held is True
    follower = FleetLease(path, owner_id="broker-b")
    try:
        assert follower.acquire() is True
    finally:
        follower.release()


def test_the_lease_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "state" / "nested" / "fleet.lease"
    lease = FleetLease(path, owner_id="broker-a")
    try:
        assert lease.acquire() is True
        assert path.exists()
    finally:
        lease.release()
