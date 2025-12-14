.PHONY: help install install-dev test lint format dvc-pull dvc-repro api docker-build docker-up

help:
	@echo "Targets:"
	@echo "  install       Install prod dependencies"
	@echo "  install-dev   Install dev dependencies"
	@echo "  test          Run pytest"
	@echo "  lint          Run ruff/black/isort checks"
	@echo "  format        Auto-format (ruff --fix, black, isort)"
	@echo "  dvc-pull      Pull DVC artifacts"
	@echo "  dvc-repro     Run full DVC pipeline"
	@echo "  api           Run FastAPI locally"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Run docker compose"

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	ruff check . --fix
	black .
	isort .

dvc-pull:
	dvc pull

dvc-repro:
	dvc repro

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t mlops-pipeline .

docker-up:
	docker compose up --build
