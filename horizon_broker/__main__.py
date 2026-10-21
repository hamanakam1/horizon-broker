"""Command line entry point.

    python -m horizon_broker serve --config config/broker.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from dataclasses import replace
from typing import Any

from horizon_broker.config import load_config
from horizon_broker.ingress.server import BrokerServer
from horizon_broker.observability.logging import configure_logging
from horizon_broker.scheduler.engine import BrokerEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horizon-broker", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the broker and its ingress API")
    serve.add_argument("--config", default=None, help="path to the YAML configuration file")
    serve.add_argument("--host", default=None, help="override the bind address")
    serve.add_argument("--port", type=int, default=None, help="override the bind port")

    subparsers.add_parser("version", help="print the broker version")
    return parser


async def run_server(config_path: str | None, host: str | None, port: int | None) -> int:
    """Run the broker until interrupted."""
    config = load_config(config_path)
    overrides: dict[str, Any] = {}
    if host:
        overrides["host"] = host
    if port:
        overrides["port"] = port
    if overrides:
        config = replace(config, **overrides)
    configure_logging(config.log_level)

    engine = BrokerEngine(config)
    server = BrokerServer(engine, config)
    await server.start()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        await server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run the requested command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        from horizon_broker import __version__

        print(__version__)
        return 0
    if args.command == "serve":
        try:
            return asyncio.run(run_server(args.config, args.host, args.port))
        except KeyboardInterrupt:
            return 0
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
