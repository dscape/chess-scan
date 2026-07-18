#!/usr/bin/env python3
"""Activate one registered candidate model for subsequent scan requests."""

from __future__ import annotations

import argparse
import json

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Registered immutable model version")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement that held-out promotion gates were reviewed",
    )
    parser.add_argument(
        "--override-gate",
        action="store_true",
        help="Allow a failed/legacy model only for an intentional rollback or diagnosed exception",
    )
    args = parser.parse_args()
    if not args.confirm:
        parser.error("Pass --confirm after reviewing the candidate's held-out metrics")

    settings = Settings.load()
    database = initialize_database(settings)
    model = database.get_model(args.version)
    metadata = json.loads(model["metadata_json"])
    if not bool(metadata.get("eligible_for_promotion")) and not args.override_gate:
        parser.error(
            "This model did not pass the recorded promotion gate. "
            "Use --override-gate only for an intentional rollback or reviewed exception."
        )
    database.promote_model(args.version)
    print(f"Activated {args.version}. New scan requests will reload it automatically.")


if __name__ == "__main__":
    main()
