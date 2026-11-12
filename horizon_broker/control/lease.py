"""Fleet lease.

Phase 2 of the plan needs one broker to own a fleet: two brokers on one
network would both accept a job id and run it twice. The lease is the
answer, and it reuses the primitive the journal already uses - an advisory
file lock.

The properties that matter:

* **It is released by the operating system when the process dies.** A
  broker that crashes cannot leave a stale lease behind, so a follower
  takes over as soon as the kernel says the lock is free. There is no
  timeout to tune and no clock to trust.
* **It is held for the life of a file descriptor**, so a broker that
  stalls cannot lose the fleet by accident.
* **The owner is written into the file**, so an operator can see who owns
  the fleet without asking a running process.

The lock must live on a filesystem with real ``flock`` semantics. A local
disk is fine; a network filesystem that emulates ``flock`` loosely is not,
because two brokers could then both believe they own the fleet.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pathlib
from typing import IO

from horizon_broker.contracts.errors import LeaseUnavailableError

DEFAULT_LEASE_NAME = "fleet.lease"


class FleetLease:
    """An advisory lock over a shared file: the holder owns the fleet."""

    def __init__(self, path: str | pathlib.Path, *, owner_id: str) -> None:
        self.path = pathlib.Path(path)
        self.owner_id = owner_id
        self._handle: IO[str] | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_held(self) -> bool:
        """True while this process holds the lease."""
        return self._handle is not None

    def holder(self) -> str | None:
        """The owner recorded in the lease file, if there is one."""
        try:
            return self.path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    # -- the lock ------------------------------------------------------------

    def acquire(self) -> bool:
        """Try to take the lease. Returns False when someone else holds it.

        The attempt never blocks: a broker that cannot own the fleet must
        say so immediately rather than wait and stall the work in front of
        it.
        """
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
        self._handle = handle
        self._record_owner()
        return True

    def renew(self) -> bool:
        """Re-assert the lease and refresh the recorded owner.

        The lock itself is held by the descriptor and cannot be lost while
        the process lives; this is the operator-facing heartbeat that says
        the owner is alive and when it last said so.
        """
        if self._handle is None:
            return False
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        self._record_owner()
        return True

    def release(self) -> None:
        """Give the lease up so a follower can take it."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            handle.close()

    def _record_owner(self) -> None:
        """Write the owner into the lease file, so an operator can read it."""
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            self._handle.truncate(0)
            self._handle.write(f"{self.owner_id}\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError:
            # The lock is what matters; the note on it is a convenience.
            pass

    def __enter__(self) -> FleetLease:
        """Take the lease, or say who holds it."""
        if not self.acquire():
            holder = self.holder() or "another broker"
            raise LeaseUnavailableError(f"the fleet is owned by {holder}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
