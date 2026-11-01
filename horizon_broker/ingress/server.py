"""HTTP server lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from horizon_broker.config import BrokerConfig
from horizon_broker.ingress.app import create_app
from horizon_broker.scheduler.engine import BrokerEngine

LOGGER = logging.getLogger(__name__)


class BrokerServer:
    """Runs the ingress API next to the broker engine."""

    def __init__(self, engine: BrokerEngine, config: BrokerConfig) -> None:
        self.engine = engine
        self.config = config
        self.app = create_app(engine, config)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        """Start the engine and bind the HTTP listener."""
        try:
            await self.engine.start()
            self._runner = web.AppRunner(self.app, access_log=None)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self.config.host, self.config.port)
            await self._site.start()
        except BaseException:
            if self._site is not None:
                try:
                    await self._site.stop()
                except Exception as exc:  # noqa: BLE001 - preserve the startup failure
                    LOGGER.debug("could not stop a partially started HTTP site: %s", exc)
                self._site = None
            if self._runner is not None:
                try:
                    await self._runner.cleanup()
                except Exception as exc:  # noqa: BLE001 - preserve the startup failure
                    LOGGER.debug("could not clean up a partially started HTTP runner: %s", exc)
                self._runner = None
            await self.engine.stop(drain=False)
            raise
        LOGGER.info(
            "horizon broker ingress listening on http://%s:%d",
            self.config.host,
            self.config.port,
        )

    async def stop(self, *, drain: bool = True) -> None:
        """Stop the HTTP listener and the engine.

        With ``drain`` the engine finishes its in-flight jobs first; the
        listener is stopped before that so no new work arrives.
        """
        site = self._site
        runner = self._runner
        try:
            if site is not None:
                await site.stop()
        finally:
            self._site = None
            try:
                if runner is not None:
                    await runner.cleanup()
            finally:
                self._runner = None
                await self.engine.stop(drain=drain)

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def __enter__(self) -> BrokerServer:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None
