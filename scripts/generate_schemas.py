"""Generate the JSON Schema files from the Pydantic contracts.

    python scripts/generate_schemas.py

The schemas are the published wire format of the broker; they are checked in
so that consumers do not need the Python package to validate a payload.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from horizon_broker.contracts.jobs import InferenceJob, InferencePayload
from horizon_broker.contracts.ledger import LedgerRecord
from horizon_broker.contracts.nodes import NodeCapability
from horizon_broker.contracts.recovery import FaultRecoverySnapshot

TARGETS: dict[str, Any] = {
    "inference_job": InferenceJob,
    "inference_payload": InferencePayload,
    "node_capability": NodeCapability,
    "fault_recovery_snapshot": FaultRecoverySnapshot,
    "ledger_record": LedgerRecord,
}


def main() -> None:
    """Write one JSON Schema file per contract."""
    root = pathlib.Path(__file__).resolve().parents[1]
    target = root / "schemas"
    target.mkdir(parents=True, exist_ok=True)
    for name, model in TARGETS.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://hamanaka.local/horizon-broker/{name}.v1.json"
        schema["title"] = model.__name__
        path = target / f"{name}.v1.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
