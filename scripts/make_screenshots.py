#!/usr/bin/env python3
"""Render the README's terminal screenshots from a real Horizon Broker.

Every pixel in ``docs/screenshots/*.png`` comes from a real broker process
and the real operator client talking to it over HTTP. Nothing is drawn by
hand and nothing is pasted from a terminal that happened to look good: the
script starts the broker, submits a job to it, and paints the bytes the tools
actually wrote.

Usage::

    python3 -m pip install pillow
    python3 scripts/make_screenshots.py            # regenerate every shot

The script needs Pillow, which is a *development* dependency only.

What gets captured:

``broker.png``
    The broker starting up, then the operator client asking it three
    questions: is it healthy, what nodes are in the fleet, and what is
    already in the queue. The broker is started against the shipped example
    configuration in a throwaway directory, so the journal it writes never
    lands in the repository.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "docs" / "screenshots"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from term2png import FontBook, session  # noqa: E402


def free_port() -> int:
    """Ask the kernel for a port nobody is using."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_cli(args: list[str], cwd: Path) -> str:
    """Run the real operator client and return exactly what it printed."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["FORCE_COLOR"] = "1"
    env["COLUMNS"] = "200"
    env.pop("NO_COLOR", None)
    completed = subprocess.run(
        [sys.executable, "-m", "horizon_broker.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def start_broker(port: int, cwd: Path) -> tuple[subprocess.Popen, Path]:
    """Start the real broker and wait until it is listening."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["HORIZON_BROKER_PORT"] = str(port)
    log_path = cwd / "broker.log"
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "horizon_broker",
            "serve",
            "--config",
            str(REPO / "config" / "broker.example.yaml"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(cwd),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the broker exited early:\n{log_path.read_text()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return process, log_path
        except OSError:
            time.sleep(0.2)
    process.terminate()
    raise SystemExit(f"the broker never started listening:\n{log_path.read_text()}")


def build() -> int:
    fonts = FontBook()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="horizon-shots-") as raw:
        work = Path(raw)
        port = free_port()
        process, log_path = start_broker(port, work)
        try:
            base = ["--base-url", f"http://127.0.0.1:{port}"]

            startup = log_path.read_text(encoding="utf-8").strip()
            health = run_cli([*base, "health"], work)
            nodes = run_cli([*base, "nodes"], work)
            submitted = run_cli(
                [
                    *base,
                    "job",
                    "submit",
                    "--model",
                    "qwen2.5:7b",
                    "--prompt",
                    "Recursive codebase scan of the scheduler package",
                    "--tier",
                    "flex",
                ],
                work,
            )
            listing = run_cli([*base, "job", "list"], work)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                process.kill()

        shot = session(
            "horizon-broker — the fleet",
            [
                ("horizon-broker serve --config config/broker.example.yaml", startup),
                ("horizon-broker health", health),
                ("horizon-broker nodes", nodes),
            ],
            fonts,
        )
        path = OUTPUT_DIR / "fleet.png"
        shot.save(path, optimize=True)
        print(f"wrote {path.relative_to(REPO)} · {path.stat().st_size // 1024} KiB")

        shot = session(
            "horizon-broker — the queue",
            [
                (
                    "horizon-broker job submit --model qwen2.5:7b "
                    '--prompt "Recursive codebase scan of the scheduler package" --tier flex',
                    submitted,
                ),
                ("horizon-broker job list", listing),
            ],
            fonts,
        )
        path = OUTPUT_DIR / "queue.png"
        shot.save(path, optimize=True)
        print(f"wrote {path.relative_to(REPO)} · {path.stat().st_size // 1024} KiB")
    return 0


def main(argv: list[str] | None = None) -> int:
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
