#!/usr/bin/env python3
"""Gate position-review explanations against derived expert commentary claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_scan.expert_commentary import (
    DEFAULT_MANIFEST_PATH,
    commentary_gate,
    evaluate_commentary,
    load_commentary_manifest,
    validate_commentary_manifests_disjoint,
)
from qa_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load_commentary_manifest(args.manifest)
    if args.split == "shadow":
        validate_commentary_manifests_disjoint(
            manifest,
            [load_commentary_manifest(DEFAULT_MANIFEST_PATH)],
        )
    metrics = evaluate_commentary(manifest, split=args.split)
    passed, reasons = commentary_gate(manifest, metrics)
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
        raise SystemExit("Expert commentary gate failed")


if __name__ == "__main__":
    main()
