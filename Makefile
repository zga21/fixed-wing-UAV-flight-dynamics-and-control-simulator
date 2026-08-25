.PHONY: test test-fast gate lint format types reproduce

test:
	python -m pytest -v

test-fast:
	python -m pytest -v -m "not gate and not slow"

gate:
	python -m pytest -v -m gate tests/gates/test_phase_$(PHASE).py

lint:
	python -m ruff check src tests
	python -m black --check src tests

format:
	python -m black src tests
	python -m ruff check --fix src tests

types:
	python -m mypy src

reproduce:
	python scripts/reproduce_all.py
