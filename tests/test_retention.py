"""Retention: the job store and the local journal must stay bounded."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from horizon_broker.contracts.jobs import JobState
from horizon_broker.recovery.journal import LocalJournal
from horizon_broker.scheduler import engine as engine_module
from horizon_broker.scheduler.engine import BrokerEngine
from tests.conftest import make_job, make_node


async def test_the_job_store_is_pruned(config) -> None:
    node = make_node("node-a")
    engine = BrokerEngine(config)
    engine.registry.upsert(node)
    try:
        # Fill the store past the cap with finished jobs.
        original_cap = engine_module.MAX_TRACKED_JOBS
        engine_module.MAX_TRACKED_JOBS = 10
        try:
            for index in range(20):
                job = make_job()
                job.state = JobState.COMPLETED
                engine._jobs[job.job_id] = job
            engine._prune_jobs()
            assert len(engine._jobs) <= 10
        finally:
            engine_module.MAX_TRACKED_JOBS = original_cap
    finally:
        await engine.stop()


async def test_running_jobs_are_never_pruned(config) -> None:
    engine = BrokerEngine(config)
    try:
        original_cap = engine_module.MAX_TRACKED_JOBS
        engine_module.MAX_TRACKED_JOBS = 1
        try:
            running = make_job()
            running.state = JobState.RUNNING
            finished = make_job()
            finished.state = JobState.COMPLETED
            engine._jobs[running.job_id] = running
            engine._jobs[finished.job_id] = finished
            engine._prune_jobs()
            assert running.job_id in engine._jobs
        finally:
            engine_module.MAX_TRACKED_JOBS = original_cap
    finally:
        await engine.stop()


async def test_the_journal_is_trimmed_to_its_limit(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    for index in range(200):
        await journal.append("snapshot", {"job_id": f"job-{index}", "n": index})

    assert journal.path.stat().st_size > 0
    removed = await journal.trim(1024)
    assert removed > 0
    assert journal.path.stat().st_size <= 1024

    records = await journal.read_all("snapshot")
    assert records
    # The newest records are the ones that survive.
    assert records[-1]["job_id"] == "job-199"


async def test_trimming_a_small_journal_is_a_no_op(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    await journal.append("snapshot", {"job_id": "job-1"})
    before = journal.path.read_bytes()
    assert await journal.trim(1024 * 1024) == 0
    assert journal.path.read_bytes() == before


async def test_the_journal_rejects_non_standard_json_numbers(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")

    with pytest.raises(ValueError, match="Out of range float values"):
        await journal.append("snapshot", {"value": float("nan")})

    assert not journal.path.exists()


async def test_the_append_kind_cannot_be_overridden_by_payload(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")

    await journal.append("snapshot", {"kind": "job_payload", "job_id": "job-1"})

    assert await journal.read_all("snapshot") == [
        {"kind": "snapshot", "job_id": "job-1"}
    ]
    assert await journal.read_all("job_payload") == []


async def test_trimming_a_missing_journal_is_safe(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "missing.jsonl")
    assert await journal.trim(1024) == 0


async def test_trimming_below_one_record_keeps_the_newest_complete_record(
    tmp_path: Path,
) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    await journal.append("snapshot", {"job_id": "job-1", "detail": "larger than one byte"})
    await journal.append("snapshot", {"job_id": "job-2", "detail": "newest"})

    await journal.trim(1)

    records = await journal.read_all("snapshot")
    assert records == [{"detail": "newest", "job_id": "job-2", "kind": "snapshot"}]


async def test_corrupt_utf8_lines_do_not_abort_journal_reads(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    valid = b'{"job_id":"job-1","kind":"snapshot"}\n'
    journal.path.write_bytes(valid + b"\xff\xfe\n" + valid)

    assert await journal.read_all("snapshot") == [
        {"job_id": "job-1", "kind": "snapshot"},
        {"job_id": "job-1", "kind": "snapshot"},
    ]


async def test_non_standard_json_numbers_do_not_enter_replay(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    valid = b'{"job_id":"job-1","kind":"snapshot"}\n'
    journal.path.write_bytes(valid + b'{"job_id":"job-2","kind":"snapshot","value":NaN}\n')

    assert await journal.read_all("snapshot") == [
        {"job_id": "job-1", "kind": "snapshot"}
    ]


async def test_trimming_discards_an_unterminated_final_record(tmp_path: Path) -> None:
    journal = LocalJournal(tmp_path / "journal.jsonl")
    complete = b'{"job_id":"job-1","kind":"snapshot"}\n'
    journal.path.write_bytes(complete + b'{"job_id":"partial"')

    await journal.trim(1)

    assert await journal.read_all("snapshot") == [
        {"job_id": "job-1", "kind": "snapshot"}
    ]


async def test_separate_journal_instances_do_not_interleave_writers(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journals = [LocalJournal(path) for _ in range(4)]

    def append(index: int) -> None:
        journals[index % len(journals)].append_sync("snapshot", {"job_id": f"job-{index}"})

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(100)))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    assert all(json.loads(line)["kind"] == "snapshot" for line in lines)
