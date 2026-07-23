#!/usr/bin/env python3
"""Export immutable position-review ratings with their exact analysis contracts."""

from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings
from chess_scan.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/position-review-feedback.jsonl"),
    )
    parser.add_argument(
        "--rating",
        choices=("helpful", "unhelpful"),
        help="Export only one rating class.",
    )
    args = parser.parse_args()

    database = initialize_database(Settings.load())
    adjudication_groups = iter(
        groupby(
            database.iter_position_review_adjudications_for_export(rating=args.rating),
            key=lambda row: str(row["review_feedback_id"]),
        )
    )
    current_group = next(adjudication_groups, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with args.output.open("w") as handle:
        for row in database.iter_position_review_feedback(rating=args.rating):
            review_feedback_id = str(row["review_feedback_id"])
            adjudications: list[dict[str, object]] = []
            if current_group is not None and current_group[0] == review_feedback_id:
                adjudications = [_adjudication(item) for item in current_group[1]]
                current_group = next(adjudication_groups, None)
            record = {
                "review_feedback_id": row["review_feedback_id"],
                "created_at": row["created_at"],
                "rating": row["rating"],
                "reason": row["reason"],
                "detail": row["detail"],
                "review_id": row["review_id"],
                "position_feedback_id": row["position_feedback_id"],
                "schema_version": row["schema_version"],
                "engine": row["engine"],
                "request": json.loads(row["request_json"]),
                "response": json.loads(row["response_json"]),
                "coaching_status_at_rating": row["coaching_status"],
                "coaching": _coaching(database, row),
                "adjudications": adjudications,
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            exported += 1

    print(f"Exported {exported} position-review ratings to {args.output}")


def _coaching(
    database: Database,
    row: dict[str, object],
) -> dict[str, object] | None:
    presented_run_id = row["presented_commentary_run_id"]
    if presented_run_id is None:
        return None
    snapshot = database.commentary_snapshot_from_feedback_row(row)
    if snapshot["run_id"] != presented_run_id:
        raise ValueError("Exported feedback coaching snapshot is inconsistent")
    return snapshot


def _adjudication(row: dict[str, object]) -> dict[str, object]:
    return {
        "adjudication_id": row["adjudication_id"],
        "created_at": row["created_at"],
        "reviewer": row["reviewer"],
        "disposition": row["disposition"],
        "notes": row["notes"],
        "regression_fixture": row["regression_fixture"],
    }


if __name__ == "__main__":
    main()
