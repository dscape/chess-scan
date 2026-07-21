SHELL := /bin/bash

.PHONY: install dev dev-api dev-web test web-test lint format-check typecheck build check prepare-argus-data prepare-platform-data prepare-lichess-puzzles train-argus-recovery train-platform-model train-print-recovery qa-argus qa-platform qa-print qa-review qa-online qa-stress docker-build

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

web-test:
	cd web && npm test

lint:
	uv run ruff check server scripts tests

format-check:
	uv run ruff format --check server scripts tests

typecheck:
	cd web && npm run typecheck

build:
	cd web && npm run build

check: lint format-check test web-test typecheck build

prepare-argus-data:
	uv run python scripts/prepare_argus_training_data.py

prepare-platform-data:
	uv run python scripts/prepare_platform_training_data.py

prepare-lichess-puzzles:
	uv run python scripts/prepare_lichess_puzzles.py --download

train-argus-recovery:
	uv run --extra ml --with 'pymupdf>=1.25,<2' python scripts/train_argus_recovery.py

train-platform-model:
	uv run --extra ml --with 'pymupdf>=1.25,<2' python scripts/train_platform_model.py --architecture wide --trainable-blocks 6

train-print-recovery:
	uv run --extra ml --with 'pymupdf>=1.25,<2' python scripts/train_print_recovery.py

qa-argus:
	uv run python scripts/evaluate_argus_replay.py --baseline models/chess-steps-v4.onnx

qa-platform:
	uv run python scripts/evaluate_platforms.py --baseline models/chess-steps-v4.onnx --variant clean --variant camera

qa-print:
	uv run python scripts/evaluate_print_regressions.py --baseline models/chess-steps-v4.onnx

qa-review:
	uv run python scripts/evaluate_lichess_puzzles.py --split validation

qa-online:
	uv run --with 'pymupdf>=1.25,<2' python scripts/evaluate_online_examples.py --cache-dir data/qa-cache

qa-stress:
	uv run --with 'pymupdf>=1.25,<2' python scripts/evaluate_photo_stress.py --cache-dir data/qa-cache

docker-build:
	docker build -t chess-scan .
