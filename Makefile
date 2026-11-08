.PHONY: install check lint test train evaluate

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint test

train:
	ltw train

evaluate:
	ltw evaluate --report docs/evaluation-report.md
