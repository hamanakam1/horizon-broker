.PHONY: help install dev test lint typecheck schemas check serve clean

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "install    install the package"
	@echo "dev        install the package with the development extras"
	@echo "test       run the test suite"
	@echo "lint       run ruff"
	@echo "typecheck  run mypy in strict mode"
	@echo "schemas    regenerate schemas/*.v1.json from the Pydantic contracts"
	@echo "check      run the pre-proposal gate: lint, typecheck, schema drift"
	@echo "serve      run the broker with config/broker.yaml"
	@echo "clean      remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(BIN)/ruff check .

typecheck:
	$(BIN)/mypy horizon_broker

schemas:
	$(PYTHON) scripts/generate_schemas.py

check: lint typecheck
	@$(BIN)/python scripts/generate_schemas.py
	@git diff --exit-code schemas/ > /dev/null || \
		(echo "schemas are stale; run make schemas" && false)
	@echo "check: lint, typecheck and schema drift all clean"

serve:
	$(PYTHON) -m horizon_broker serve --config config/broker.yaml

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
