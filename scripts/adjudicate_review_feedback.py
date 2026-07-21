#!/usr/bin/env python3
"""Append an expert adjudication to one immutable position-review rating."""

from __future__ import annotations

import argparse
import uuid

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings

DISPOSITIONS = ("confirmed_issue", "rejected", "duplicate", "approved_fix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_feedback_id")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--disposition", choices=DISPOSITIONS, required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--regression-fixture",
        help="Required for approved_fix; identify the committed regression test or fixture.",
    )
    args = parser.parse_args()

    database = initialize_database(Settings.load())
    adjudication_id = uuid.uuid4().hex
    database.append_position_review_adjudication(
        adjudication_id=adjudication_id,
        review_feedback_id=args.review_feedback_id,
        reviewer=args.reviewer,
        disposition=args.disposition,
        notes=args.notes,
        regression_fixture=args.regression_fixture,
    )
    print(f"Recorded review adjudication {adjudication_id}")


if __name__ == "__main__":
    main()
