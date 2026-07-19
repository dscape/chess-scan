#!/usr/bin/env python3
"""Export corrected predictions as chosen/rejected preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/preferences.jsonl"))
    args = parser.parse_args()

    settings = Settings.load()
    database = initialize_database(settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0

    with args.output.open("w") as handle:
        for row in database.iter_preference_examples():
            record = {
                "preference_id": row["feedback_id"],
                "image_path": row["rectified_image_path"],
                "chosen_labels": json.loads(row["final_labels_json"]),
                "rejected_labels": json.loads(row["predicted_labels_json"]),
                "rejected_probabilities": json.loads(row["predicted_probabilities_json"]),
                "source_model": row["model_version"],
                "changed_squares": row["changed_squares"],
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            exported += 1

    print(f"Exported {exported} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
