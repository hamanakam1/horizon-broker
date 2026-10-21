# Contributor instructions

Read `README.md` and `docs/ARCHITECTURE.md` before changing behaviour.

## Ground rules

- The contracts in `horizon_broker/contracts/` are the wire format. Changing a
  field name or type is a breaking change: bump `schema_version`, regenerate
  `schemas/`, and update `docs/CONTRACTS.md`.
- Never let the ingress layer touch the queue, the router or the recovery
  subsystem directly. It validates, categorises and hands over.
- The scheduler never talks to a node. It produces dispatch batches.
- The watchdog only reports verdicts. It does not decide what happens next.
- Recovery must be idempotent: replaying the same snapshot twice must not
  corrupt the job.

## Before proposing a change

```sh
python -m pytest
ruff check .
mypy horizon_broker
python scripts/generate_schemas.py && git diff --exit-code schemas/
```

## Testing

- Keep fixtures synthetic. No real prompts, transcripts, tickets, node names or
  deployment addresses in the test suite.
- A test that touches the network must use a local fake server (see
  `tests/test_engine.py` for the fake Ollama pattern).
- Fault-path tests must assert the *observable* outcome (the job completed on
  another node), not the internal call sequence.

## Style

- Python 3.11+, `from __future__ import annotations` in every module.
- Type annotations everywhere; `mypy --strict` must pass.
- Ruff line length 100.
- Docstrings explain *why*, not *what*.
