"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from horizon_broker.__main__ import main
from horizon_broker.config import (
    BrokerConfig,
    HeartbeatConfig,
    RoutingConfig,
    SchedulingConfig,
    WindowConfig,
    load_config,
)


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "broker.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_minimal_configuration_loads(tmp_path: Path) -> None:
    path = write_config(tmp_path, "broker:\n  port: 9000\n")
    config = load_config(path)
    assert config.port == 9000
    assert config.static_nodes == ()


def test_the_defaults_are_valid() -> None:
    BrokerConfig().validate()


def test_a_batch_window_smaller_than_the_minimum_is_rejected() -> None:
    config = BrokerConfig(window=WindowConfig(min_batch_size=8, max_batch_size=4))
    with pytest.raises(ValueError, match="max_batch_size"):
        config.validate()


def test_a_heartbeat_timeout_below_the_interval_is_rejected() -> None:
    config = BrokerConfig(
        heartbeat=HeartbeatConfig(interval_seconds=5.0, timeout_seconds=2.0)
    )
    with pytest.raises(ValueError, match="timeout_seconds"):
        config.validate()


def test_a_zero_queue_depth_is_rejected() -> None:
    config = BrokerConfig(scheduling=SchedulingConfig(max_queue_depth=0))
    with pytest.raises(ValueError, match="max_queue_depth"):
        config.validate()


def test_duplicate_node_ids_are_rejected(tmp_path: Path) -> None:
    body = """
nodes:
  - node_id: node-a
    available_vram_bytes: 1000
    total_vram_bytes: 1000
  - node_id: node-a
    available_vram_bytes: 1000
    total_vram_bytes: 1000
"""
    path = write_config(tmp_path, body)
    with pytest.raises(ValueError, match="duplicate node id"):
        load_config(path)


def test_an_invalid_file_is_rejected_at_load_time(tmp_path: Path) -> None:
    path = write_config(tmp_path, "window:\n  min_batch_size: 9\n  max_batch_size: 2\n")
    with pytest.raises(ValueError, match="max_batch_size"):
        load_config(path)


def test_the_routing_section_is_loaded(tmp_path: Path) -> None:
    body = """
routing:
  kv_handoff_models:
    - qwen2.5:7b
  prefill_prompt_threshold: 2048
"""
    path = write_config(tmp_path, body)
    config = load_config(path)
    assert config.routing == RoutingConfig(
        kv_handoff_models=("qwen2.5:7b",),
        prefill_prompt_threshold=2048,
        decode_output_threshold=2048,
        prefill_intensity_ratio=4.0,
        decode_intensity_ratio=0.25,
    )


def test_power_gated_tiers_survive_a_config_file_that_omits_them(tmp_path: Path) -> None:
    """A config file with a ``window:`` section must not switch power gating off."""
    path = write_config(tmp_path, "window:\n  window_seconds: 30\n")
    config = load_config(path)
    assert config.window.power_gated_tiers == BrokerConfig().window.power_gated_tiers
    assert config.window.power_gated_tiers != ()


def test_the_journal_max_bytes_is_read_from_the_file(tmp_path: Path) -> None:
    """The journal size limit is a documented setting; it must be honoured."""
    path = write_config(tmp_path, "broker:\n  journal_max_bytes: 1024\n")
    config = load_config(path)
    assert config.journal_max_bytes == 1024


def test_the_check_config_command_accepts_a_good_file(tmp_path: Path, capsys) -> None:
    path = write_config(tmp_path, "broker:\n  port: 9000\n")
    assert main(["check-config", "--config", str(path)]) == 0
    assert "configuration is valid" in capsys.readouterr().out


def test_the_check_config_command_rejects_a_bad_file(tmp_path: Path, capsys) -> None:
    path = write_config(tmp_path, "window:\n  min_batch_size: 9\n  max_batch_size: 2\n")
    assert main(["check-config", "--config", str(path)]) == 1
    assert "not usable" in capsys.readouterr().err


def test_the_environment_overrides_the_file(tmp_path: Path, monkeypatch) -> None:
    path = write_config(tmp_path, "broker:\n  port: 9000\n")
    monkeypatch.setenv("HORIZON_BROKER_PORT", "9100")
    monkeypatch.setenv("HORIZON_BROKER_WINDOW_SECONDS", "12")
    config = load_config(path)
    assert config.port == 9100
    assert config.window.window_seconds == 12.0


def test_the_environment_can_disable_the_idle_flush(monkeypatch) -> None:
    monkeypatch.setenv("HORIZON_BROKER_WINDOW_IDLE_FLUSH", "false")
    config = load_config(None)
    assert config.window.idle_flush is False


def test_a_config_with_a_drain_timeout_loads(tmp_path: Path) -> None:
    path = write_config(tmp_path, "broker:\n  drain_timeout_seconds: 5\n")
    config = load_config(path)
    assert config.drain_timeout_seconds == 5.0
