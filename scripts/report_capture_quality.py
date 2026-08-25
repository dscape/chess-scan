#!/usr/bin/env python3
"""Report aggregate quality for confirmed scans without exporting private board data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from chess_scan.bootstrap import initialize_database
from chess_scan.capture_audit import summarize_confirmed_captures
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--since", help="Only confirmations after this ISO-8601 timestamp")
    scope.add_argument("--all", action="store_true", help="Report all confirmed scans")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args()

    database = initialize_database(Settings.load())
    since = None if args.all else args.since
    if since is None and not args.all:
        cycle = database.active_learning_cycle()
        if cycle is not None and cycle["state"] == "shadowing":
            since = str(cycle["shadow_started_at"])
    if since is not None:
        parsed_since = datetime.fromisoformat(since)
        if parsed_since.tzinfo is None:
            parser.error("--since must include a UTC offset")

    summary = summarize_confirmed_captures(database.confirmed_capture_rows(created_after=since))
    payload = {"created_after": since, **summary}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n")
    print(rendered)


if __name__ == "__main__":
    main()
