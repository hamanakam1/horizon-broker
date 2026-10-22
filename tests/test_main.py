"""Tests for the ``python -m horizon_broker`` entry point."""

from __future__ import annotations

from horizon_broker import __version__
from horizon_broker.__main__ import main
from horizon_broker.banner import BANNER


def test_version_prints_the_wordmark_and_version(capsys) -> None:
    """The version stamp carries the wordmark so the artwork ships with it."""
    exit_code = main(["version"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert BANNER in out
    assert f"horizon-broker {__version__}" in out
