#!/usr/bin/env python3
"""Export confirmed, consented boards as an auditable supervised dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/feedback.jsonl"))
    args = parser.parse_args()

    settings = Settings.load()
    database = initialize_database(settings)
    examples = database.training_examples()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as handle:
        for row in examples:
            record = {
                "feedback_id": row["feedback_id"],
                "scan_id": row["scan_id"],
                "created_at": row["created_at"],
                "image_path": row["rectified_image_path"],
                "image_sha256": row["image_sha256"],
                "labels": json.loads(row["final_labels_json"]),
                "orientation": row["orientation"],
                "side_to_move": row["side_to_move"],
                "fen": row["final_fen"],
                "changed_squares": row["changed_squares"],
                "source_model": row["model_version"],
                "client_session_id": row["client_session_id"],
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Exported {len(examples)} confirmed boards to {args.output}")


if __name__ == "__main__":
    main()
