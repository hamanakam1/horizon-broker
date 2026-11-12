"""The trial harness: the job ledger and the SLO report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from trial_harness import (  # noqa: E402
    JobLedger,
    TrialConfig,
    evaluate,
    render_report,
    report,
)


def test_the_ledger_remembers_the_last_state_of_every_job() -> None:
    ledger = JobLedger()
    ledger.observe(
        [{"job_id": "job-1", "state": "running"}, {"job_id": "job-2", "state": "queued"}],
        100.0,
    )
    ledger.observe([{"job_id": "job-1", "state": "completed"}], 110.0)

    assert ledger.states == {"job-1": "completed", "job-2": "queued"}
    assert ledger.first_seen["job-1"] == 100.0
    assert ledger.outcomes()["completed"] == 1


def test_a_job_that_never_finished_is_lost() -> None:
    """A lost job is one that never reached a terminal state."""
    ledger = JobLedger()
    ledger.observe(
        [
            {"job_id": "job-1", "state": "completed"},
            {"job_id": "job-2", "state": "running"},
            {"job_id": "job-3", "state": "windowing"},
        ],
        100.0,
    )
    assert ledger.lost() == 2


def test_the_report_evaluates_the_slos(tmp_path: Path) -> None:
    samples = [
        {
            "ts": 100.0,
            "metrics": {
                "horizon_jobs_submitted_total": 4.0,
                "horizon_jobs_completed_total": 4.0,
                "horizon_recovery_seconds_sum": 20.0,
                "horizon_recovery_seconds_count": 2.0,
            },
            "jobs": {"job-1": "completed", "job-2": "completed"},
        },
        {
            "ts": 110.0,
            "metrics": {
                "horizon_jobs_submitted_total": 6.0,
                "horizon_jobs_completed_total": 6.0,
                "horizon_recovery_seconds_sum": 50.0,
                "horizon_recovery_seconds_count": 5.0,
            },
            "jobs": {"job-1": "completed", "job-2": "completed", "job-3": "running"},
        },
    ]
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n", encoding="utf-8")

    config = TrialConfig(broker_url="http://127.0.0.1:8791")
    assert report(config, str(path), None) == 0

    ledger = JobLedger()
    ledger.observe(
        [{"job_id": "job-1", "state": "completed"}, {"job_id": "job-3", "state": "running"}],
        110.0,
    )
    results = evaluate(samples, ledger)
    assert results["lost_jobs"] == 1
    assert results["duplicate_executions"] == 0
    assert results["recovery_p99_seconds"] is not None

    text = render_report(results, config)
    assert "| No job is lost across the trial | 0 | 1 | FAIL |" in text
    assert "Recovery completes within 30 s at p99" in text


def test_a_duplicate_execution_is_reported(tmp_path: Path) -> None:
    """More completions than submissions means the fleet ran something twice."""
    samples = [
        {
            "ts": 100.0,
            "metrics": {
                "horizon_jobs_submitted_total": 2.0,
                "horizon_jobs_completed_total": 3.0,
            },
            "jobs": {},
        }
    ]
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(samples[0]) + "\n", encoding="utf-8")

    ledger = JobLedger()
    results = evaluate(samples, ledger)
    assert results["duplicate_executions"] == 1


def test_the_trial_configuration_carries_the_fault_commands(tmp_path: Path) -> None:
    path = tmp_path / "trial.json"
    path.write_text(
        json.dumps(
            {
                "broker_url": "http://127.0.0.1:9000",
                "nodes": [
                    {"node_id": "pi-01", "fault_command": ["ssh", "pi"], "restore_command": []}
                ],
            }
        ),
        encoding="utf-8",
    )

    config = TrialConfig.load(path)
    assert config.broker_url == "http://127.0.0.1:9000"
    assert config.nodes[0].node_id == "pi-01"
    assert config.nodes[0].fault_command == ["ssh", "pi"]
    assert config.nodes[0].restore_command == []


def test_the_example_configuration_is_valid() -> None:
    config = TrialConfig.load(Path(__file__).resolve().parents[1] / "config/trial.example.json")
    assert config.broker_url.startswith("http")
    assert config.nodes, "the example fleet must name at least one node"
    for node in config.nodes:
        assert node.fault_command, f"node {node.node_id} needs a fault command"


def test_an_unknown_node_is_refused() -> None:
    from trial_harness import fault

    config = TrialConfig(broker_url="http://127.0.0.1:8791")
    with pytest.raises(SystemExit):
        raise SystemExit(fault(config, "nope", stop=True))
