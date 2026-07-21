#!/usr/bin/env python3
"""Gate deterministic review themes against 1,000 official Lichess puzzles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_scan.lichess_puzzles import (
    default_data_dir,
    evaluate_theme_agreement,
    load_puzzles,
    theme_gate,
    verify_data_manifest,
)
from qa_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--split",
        choices=("development", "validation"),
        default="validation",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    manifest = verify_data_manifest(data_dir)
    puzzles = load_puzzles(data_dir, split=args.split)
    metrics = evaluate_theme_agreement(puzzles)
    passed, reasons = theme_gate(metrics)
    payload = {
        "passed": passed,
        "reasons": reasons,
        "dataset_version": manifest["version"],
        "split": args.split,
        "metrics": metrics,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("Lichess puzzle-theme gate failed")


if __name__ == "__main__":
    main()
