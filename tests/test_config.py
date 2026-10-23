"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from horizon_broker.__main__ import main
from horizon_broker.config import (
    BrokerConfig,
    HeartbeatConfig,
    LedgerConfig,
    OllamaConfig,
    RoutingConfig,
    SchedulingConfig,
    WindowConfig,
    dump_config,
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


def test_dump_config_is_strict_json() -> None:
    """Keep the diagnostics endpoint consumable by strict JSON parsers."""
    rendered = dump_config(BrokerConfig())

    assert "NaN" not in rendered
    assert "Infinity" not in rendered
    assert isinstance(json.loads(rendered), dict)


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


def test_an_empty_ledger_owner_is_rejected() -> None:
    config = BrokerConfig(ledger=LedgerConfig(owner_id="  "))
    with pytest.raises(ValueError, match="owner_id"):
        config.validate()


@pytest.mark.parametrize(
    "field",
    ["owner_id", "collection", "commit_path", "query_path", "auth_token_env", "auth_header", "auth_scheme"],
)
def test_ledger_strings_reject_surrounding_whitespace(field: str) -> None:
    config = BrokerConfig(ledger=LedgerConfig(**{field: " value "}))
    with pytest.raises(ValueError, match=f"ledger.{field}"):
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
    monkeypatch.setenv("HORIZON_BROKER_LEDGER_OWNER_ID", "broker-a")
    config = load_config(path)
    assert config.port == 9100
    assert config.window.window_seconds == 12.0
    assert config.ledger.owner_id == "broker-a"


def test_the_environment_can_disable_the_idle_flush(monkeypatch) -> None:
    monkeypatch.setenv("HORIZON_BROKER_WINDOW_IDLE_FLUSH", "false")
    config = load_config(None)
    assert config.window.idle_flush is False


def test_an_invalid_boolean_environment_override_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("HORIZON_BROKER_WINDOW_IDLE_FLUSH", "definitely")
    with pytest.raises(ValueError, match="invalid boolean"):
        load_config(None)


def test_a_config_with_a_drain_timeout_loads(tmp_path: Path) -> None:
    path = write_config(tmp_path, "broker:\n  drain_timeout_seconds: 5\n")
    config = load_config(path)
    assert config.drain_timeout_seconds == 5.0


def test_boolean_integer_configuration_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "scheduling:\n  max_queue_depth: true\n")
    with pytest.raises(ValueError, match="max_queue_depth"):
        load_config(path)


def test_non_finite_float_configuration_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "heartbeat:\n  interval_seconds: .nan\n")
    with pytest.raises(ValueError, match="interval_seconds"):
        load_config(path)


def test_overflowing_float_configuration_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, f"heartbeat:\n  interval_seconds: {10**1000}\n")
    with pytest.raises(ValueError, match="interval_seconds"):
        load_config(path)


def test_string_boolean_configuration_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "window:\n  idle_flush: 'false'\n")
    with pytest.raises(ValueError, match="idle_flush"):
        load_config(path)


def test_scalar_model_whitelist_is_rejected(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
nodes:
  - node_id: node-a
    available_vram_bytes: 1000
    total_vram_bytes: 1000
    model_whitelist: synthetic-model
""",
    )
    with pytest.raises(ValueError, match="model_whitelist"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [("window", "power_gated_tiers"), ("routing", "kv_handoff_models")],
)
def test_scalar_string_sequence_configuration_is_rejected(
    tmp_path: Path, section: str, field: str
) -> None:
    path = write_config(tmp_path, f"{section}:\n  {field}: synthetic-model\n")
    with pytest.raises(ValueError, match=field):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("window", "power_gated_tiers", "  solar_trailer", "canonical"),
        ("routing", "kv_handoff_models", "synthetic-model", "duplicates"),
    ],
)
def test_sequence_configuration_rejects_non_canonical_entries(
    tmp_path: Path, section: str, field: str, value: str, message: str
) -> None:
    if message == "duplicates":
        body = f"{section}:\n  {field}:\n    - {value}\n    - {value}\n"
    else:
        body = f"{section}:\n  {field}:\n    - '{value}'\n"
    path = write_config(tmp_path, body)
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (BrokerConfig(window=WindowConfig(power_headroom_threshold=1.1)), "power_headroom"),
        (BrokerConfig(routing=RoutingConfig(prefill_prompt_threshold=-1)), "prompt_threshold"),
        (BrokerConfig(routing=RoutingConfig(prefill_intensity_ratio=0.0)), "intensity_ratio"),
    ],
)
def test_semantically_invalid_runtime_policies_are_rejected(
    config: BrokerConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (BrokerConfig(heartbeat=HeartbeatConfig(interval_seconds=0.0)), "interval_seconds"),
        (BrokerConfig(scheduling=SchedulingConfig(poll_interval_seconds=0.0)), "poll_interval"),
        (
            BrokerConfig(
                heartbeat=HeartbeatConfig(
                    reconnect_backoff_seconds=5.0, reconnect_backoff_max_seconds=1.0
                )
            ),
            "backoff_max",
        ),
    ],
)
def test_invalid_timing_policies_are_rejected(config: BrokerConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (BrokerConfig(port=0), "port must be"),
        (BrokerConfig(journal_max_bytes=0), "journal_max_bytes"),
        (BrokerConfig(ledger=LedgerConfig(max_retries=-1)), "max_retries"),
    ],
)
def test_invalid_service_and_storage_policies_are_rejected(
    config: BrokerConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_unknown_power_gated_tier_is_rejected() -> None:
    config = BrokerConfig(window=WindowConfig(power_gated_tiers=("synthetic-power",)))
    with pytest.raises(ValueError, match="unsupported values"):
        config.validate()


@pytest.mark.parametrize("field", ["ledger", "ollama"])
def test_invalid_service_base_url_is_rejected(field: str) -> None:
    config = (
        BrokerConfig(ledger=LedgerConfig(base_url="not-a-url"))
        if field == "ledger"
        else BrokerConfig(ollama=OllamaConfig(base_url="file:///tmp/ollama"))
    )
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        config.validate()


@pytest.mark.parametrize("field", ["ledger", "ollama"])
def test_service_base_url_rejects_surrounding_whitespace(field: str) -> None:
    config = (
        BrokerConfig(ledger=LedgerConfig(base_url=" http://ledger.local"))
        if field == "ledger"
        else BrokerConfig(ollama=OllamaConfig(base_url="http://ollama.local "))
    )
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        config.validate()


@pytest.mark.parametrize("field", ["ledger", "ollama"])
def test_service_base_url_rejects_embedded_credentials(field: str) -> None:
    config = (
        BrokerConfig(ledger=LedgerConfig(base_url="http://user:secret@ledger.local"))
        if field == "ledger"
        else BrokerConfig(ollama=OllamaConfig(base_url="http://user:secret@ollama.local"))
    )
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        config.validate()


@pytest.mark.parametrize("field", ["ledger", "ollama"])
def test_service_base_url_rejects_invalid_ports(field: str) -> None:
    config = (
        BrokerConfig(ledger=LedgerConfig(base_url="http://ledger.local:not-a-port"))
        if field == "ledger"
        else BrokerConfig(ollama=OllamaConfig(base_url="http://ollama.local:70000"))
    )
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        config.validate()


@pytest.mark.parametrize("field", ["ledger", "ollama"])
def test_service_base_url_rejects_malformed_bracketed_hosts(field: str) -> None:
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        if field == "ledger":
            BrokerConfig(ledger=LedgerConfig(base_url="http://[malformed")).validate()
        else:
            BrokerConfig(ollama=OllamaConfig(base_url="http://[malformed")).validate()


@pytest.mark.parametrize("field", ["ledger", "ollama"])
def test_service_base_url_rejects_empty_hosts(field: str) -> None:
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        if field == "ledger":
            BrokerConfig(ledger=LedgerConfig(base_url="http://:80")).validate()
        else:
            BrokerConfig(ollama=OllamaConfig(base_url="http://:80")).validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [(BrokerConfig(host=""), "host must not"), (BrokerConfig(log_level="TRACE"), "log_level")],
)
def test_invalid_broker_identity_settings_are_rejected(
    config: BrokerConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_malformed_string_configuration_types_are_rejected() -> None:
    config = BrokerConfig(host=True, log_level=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host must not|log_level"):
        config.validate()


@pytest.mark.parametrize("field", ["collection", "commit_path", "query_path"])
def test_empty_ledger_route_settings_are_rejected(field: str) -> None:
    config = BrokerConfig(ledger=LedgerConfig(**{field: " "}))
    with pytest.raises(ValueError, match=f"ledger.{field}"):
        config.validate()


@pytest.mark.parametrize("field", ["commit_path", "query_path"])
def test_ledger_routes_require_absolute_collection_paths(field: str) -> None:
    config = BrokerConfig(ledger=LedgerConfig(**{field: "records"}))
    with pytest.raises(ValueError, match="absolute path containing"):
        config.validate()


@pytest.mark.parametrize(
    "path",
    [
        "/collections/{unknown}/records",
        "/collections/{collection}/{collection}/records",
        "/collections/{collection!r}/records",
        "/collections/{collection:>10}/records",
        "/collections/{collection}/records?owner_id=1",
        "/collections/{collection}/records#fragment",
        "/collections/{collection/records",
    ],
)
def test_ledger_routes_reject_ambiguous_templates(path: str) -> None:
    config = BrokerConfig(ledger=LedgerConfig(commit_path=path))
    with pytest.raises(ValueError, match="absolute path containing"):
        config.validate()


@pytest.mark.parametrize("field", ["auth_token_env", "auth_header", "auth_scheme"])
def test_empty_ledger_auth_settings_are_rejected(field: str) -> None:
    config = BrokerConfig(ledger=LedgerConfig(**{field: " "}))
    with pytest.raises(ValueError, match=f"ledger.{field}"):
        config.validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            replace(
                BrokerConfig(),
                window=replace(BrokerConfig().window, min_batch_size="invalid"),
            ),
            "window.min_batch_size must be an integer",
        ),
        (replace(BrokerConfig(), port="invalid"), "port must be an integer"),
        (
            replace(
                BrokerConfig(),
                recovery=replace(BrokerConfig().recovery, recovery_budget_seconds="invalid"),
            ),
            "recovery.recovery_budget_seconds must be a finite number",
        ),
    ],
)
def test_invalid_primitive_types_are_reported_as_configuration_errors(
    config: BrokerConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()
