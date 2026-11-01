.PHONY: help install dev doctor test lint typecheck schemas check check-ci serve clean

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "install    install the package"
	@echo "dev        install the package with the development extras"
	@echo "doctor     report missing validation tools and typing stubs"
	@echo "test       run the test suite"
	@echo "lint       run ruff"
	@echo "typecheck  run mypy in strict mode"
	@echo "schemas    regenerate schemas/*.v1.json from the Pydantic contracts"
	@echo "check      run the pre-proposal gate: lint, typecheck, schema drift"
	@echo "check-ci   run the gate with the selected PYTHON interpreter"
	@echo "serve      run the broker with config/broker.yaml"
	@echo "clean      remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(BIN)/python -m pip install -e ".[dev]"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

doctor:
	@missing=0; \
	for tool in pytest ruff mypy; do \
		if $(BIN)/python -m $$tool --version >/dev/null 2>&1; then echo "ok: $$tool"; \
		else echo "missing: $$tool in $(BIN)/python (run make dev)"; missing=1; fi; \
	done; \
	if $(BIN)/python -c 'import yaml' 2>/dev/null; then echo "ok: PyYAML"; \
	else echo "missing: PyYAML (run make dev)"; missing=1; fi; \
	if $(BIN)/python -m pip show types-PyYAML >/dev/null 2>&1; then echo "ok: types-PyYAML"; \
	else echo "note: types-PyYAML is required for mypy yaml imports (run make dev)"; fi; \
	exit $$missing

test:
	$(PYTHON) -m pytest

lint:
	$(BIN)/python -m ruff check .

typecheck:
	$(BIN)/python -m mypy horizon_broker

schemas:
	$(PYTHON) scripts/generate_schemas.py

check: doctor lint typecheck
	@$(BIN)/python scripts/generate_schemas.py
	@git diff --exit-code schemas/ > /dev/null || \
		(echo "schemas are stale; run make schemas" && false)
	@echo "check: lint, typecheck and schema drift all clean"

check-ci:
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy horizon_broker
	$(PYTHON) scripts/generate_schemas.py
	@git diff --exit-code schemas/ > /dev/null || \
		(echo "schemas are stale; run make schemas" && false)
	@echo "check-ci: tests, lint, typecheck and schema drift all clean"

serve:
	$(PYTHON) -m horizon_broker serve --config config/broker.yaml

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
