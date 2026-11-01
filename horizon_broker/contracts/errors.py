"""Broker error taxonomy.

Every failure mode that the scheduler, router or recovery coordinator can
surface has a dedicated type so the ingress layer can map it onto an HTTP
status without string matching.
"""

from __future__ import annotations


class BrokerError(Exception):
    """Base class for all broker errors."""


class UnknownJobError(BrokerError):
    """The referenced job id is not known to the broker."""


class JobExpiredError(BrokerError):
    """The job's latency budget elapsed before it could be dispatched."""


class RoutingError(BrokerError):
    """The PD-DAG router could not place the job.

    The umbrella for every routing failure: a job that no node can take
    (:class:`NoCapacityError`) is a routing error too.
    """


class NoCapacityError(RoutingError):
    """No node in the mesh can accept the job right now."""


class NodeUnreachableError(BrokerError):
    """The control channel to a node closed or timed out."""


class LedgerError(BrokerError):
    """Base class for EdgeRecord transport failures."""


class LedgerUnavailableError(LedgerError):
    """EdgeRecord could not be reached; the local journal is used as a fallback."""


class RecoveryBudgetExceededError(BrokerError):
    """Recovery could not complete inside the configured budget."""


class MaxAttemptsExceededError(BrokerError):
    """The job exhausted its retry budget."""


class QueueFullError(BrokerError):
    """The queue reached its configured depth; the broker is applying backpressure."""


class BrokerDrainingError(BrokerError):
    """The broker is shutting down and is no longer accepting work."""


class InvalidSubmissionError(BrokerError):
    """A new submission carried a lifecycle state reserved for replay."""
