.PHONY: help install dev test lint typecheck schemas serve clean

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

serve:
	$(PYTHON) -m horizon_broker serve --config config/broker.yaml

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
