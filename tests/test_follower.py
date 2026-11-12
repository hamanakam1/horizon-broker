"""Two brokers, one fleet: only the lease holder admits work."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from horizon_broker.config import BrokerConfig
from horizon_broker.ingress.app import create_app
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_node

MODEL = "synthetic-model"


async def make_client(config: BrokerConfig) -> tuple[TestClient, BrokerEngine]:
    node = make_node("node-a", models=(MODEL,))
    engine = BrokerEngine(replace(config, static_nodes=(node,)))
    client = TestClient(TestServer(create_app(engine, config)))
    await client.start_server()
    return client, engine


def job_body(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "payload": {"model": MODEL, "prompt": "scan"},
        "max_acceptable_latency_ms": 600_000,
    }


async def test_only_the_lease_holder_admits_work(config: BrokerConfig, tmp_path: Path) -> None:
    """The lease decides who owns the fleet, and the owner admits the work."""
    lease = tmp_path / "fleet.lease"
    shared = replace(config, lease_path=str(lease), ledger=replace(config.ledger, owner_id="broker-a"))
    other = replace(
        config,
        lease_path=str(lease),
        ledger=replace(config.ledger, owner_id="broker-b"),
    )

    owner_client, owner = await make_client(shared)
    follower_client, follower = await make_client(other)
    try:
        await owner.start()
        await follower.start()
        assert owner.is_leader is True
        assert follower.is_leader is False
        assert follower.lease_holder == "broker-a"

        accepted = await owner_client.post("/v1/jobs", json=job_body("job-1"))
        assert accepted.status == 202

        refused = await follower_client.post("/v1/jobs", json=job_body("job-2"))
        assert refused.status == 503
        body = await refused.json()
        assert body["error"] == "broker is a follower"
        assert body["leader"] == "broker-a"

        # The fleet view stays open on a follower: it still knows the mesh.
        nodes = await follower_client.get("/v1/nodes")
        assert nodes.status == 200
        assert (await nodes.json())["nodes"]

        ready = await follower_client.get("/readyz")
        assert ready.status == 503
        assert (await ready.json())["status"] == "follower"
    finally:
        await follower.stop(drain=False)
        await owner.stop(drain=False)
        await follower_client.close()
        await owner_client.close()


async def test_the_openai_ingress_refuses_a_follower_too(
    config: BrokerConfig, tmp_path: Path
) -> None:
    lease = tmp_path / "fleet.lease"
    owner_client, owner = await make_client(
        replace(config, lease_path=str(lease), ledger=replace(config.ledger, owner_id="broker-a"))
    )
    follower_client, follower = await make_client(
        replace(config, lease_path=str(lease), ledger=replace(config.ledger, owner_id="broker-b"))
    )
    try:
        await owner.start()
        await follower.start()
        response = await follower_client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status == 503
        assert (await response.json())["error"]["type"] == "not_leader"
    finally:
        await follower.stop(drain=False)
        await owner.stop(drain=False)
        await follower_client.close()
        await owner_client.close()


async def test_a_follower_takes_over_when_the_owner_stops(
    config: BrokerConfig, tmp_path: Path
) -> None:
    """The owner releases the lease on shutdown, so the fleet is never stranded."""
    lease = tmp_path / "fleet.lease"
    owner_client, owner = await make_client(
        replace(config, lease_path=str(lease), ledger=replace(config.ledger, owner_id="broker-a"))
    )
    follower_client, follower = await make_client(
        replace(config, lease_path=str(lease), ledger=replace(config.ledger, owner_id="broker-b"))
    )
    try:
        await owner.start()
        await follower.start()
        assert follower.is_leader is False

        await owner.stop(drain=False)
        # The follower notices at its next start; the lock itself is free.
        assert follower.lease.acquire() is True
        follower.lease.release()
    finally:
        await follower.stop(drain=False)
        await owner_client.close()
        await follower_client.close()


async def test_a_broker_without_a_lease_always_leads(config: BrokerConfig) -> None:
    """The single-broker case is unchanged: no lease, no follower."""
    client, engine = await make_client(config)
    try:
        await engine.start()
        assert engine.is_leader is True
        assert engine.lease_holder is None
        response = await client.post("/v1/jobs", json=job_body("job-1"))
        assert response.status == 202
    finally:
        await engine.stop(drain=False)
        await client.close()
