"""The control plane: the lease that says who owns the fleet."""

from __future__ import annotations

from horizon_broker.control.lease import DEFAULT_LEASE_NAME, FleetLease

__all__ = ["DEFAULT_LEASE_NAME", "FleetLease"]
