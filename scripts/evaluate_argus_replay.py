#!/usr/bin/env python3
"""Evaluate a model on retained synthetic and held-out chess-positions Argus data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_scan.argus_data import default_data_dir, evaluate_argus_model, evaluate_argus_pair
from qa_common import write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "chess-steps-v3.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if args.baseline is None:
        payload = evaluate_argus_model(args.model, data_dir)
    else:
        payload = evaluate_argus_pair(args.baseline, args.model, data_dir)
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if args.baseline is not None and not payload["passed"]:
        raise SystemExit("Argus replay gate failed")


if __name__ == "__main__":
    main()
