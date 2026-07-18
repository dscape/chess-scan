SHELL := /bin/bash

.PHONY: install dev dev-api dev-web test lint format-check typecheck build check qa-online qa-stress docker-build

install:
	uv sync --extra dev
	cd web && npm install

dev:
	@trap 'kill 0' EXIT; $(MAKE) dev-api & $(MAKE) dev-web & wait

dev-api:
	uv run uvicorn chess_scan.main:app --app-dir server --reload --port 8000

dev-web:
	cd web && npm run dev

test:
	uv run pytest

lint:
	uv run ruff check server scripts tests

format-check:
	uv run ruff format --check server scripts tests

typecheck:
	cd web && npm run typecheck

build:
	cd web && npm run build

check: lint format-check test typecheck build

qa-online:
	uv run --with 'pymupdf>=1.25,<2' python scripts/evaluate_online_examples.py --cache-dir data/qa-cache

qa-stress:
	uv run --with 'pymupdf>=1.25,<2' python scripts/evaluate_photo_stress.py --cache-dir data/qa-cache

docker-build:
	docker build -t chess-scan .
