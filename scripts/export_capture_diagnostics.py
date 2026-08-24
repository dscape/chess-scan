#!/usr/bin/env python3
"""Export the manifest for short-lived, opt-in camera diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/capture-diagnostics.jsonl"),
    )
    args = parser.parse_args()

    database = initialize_database(Settings.load())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with args.output.open("w") as handle:
        for row in database.iter_capture_diagnostics():
            record = {
                "diagnostic_id": row["diagnostic_id"],
                "feedback_id": row["feedback_id"],
                "scan_id": row["scan_id"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "recording_path": row["recording_path"],
                "content_type": row["content_type"],
                "byte_count": row["byte_count"],
                "sha256": row["sha256"],
                "metadata": json.loads(row["metadata_json"]),
                "user_agent": row["user_agent"],
                "labels": json.loads(row["final_labels_json"]),
                "predicted_labels": json.loads(row["predicted_labels_json"]),
                "prediction_evidence": (
                    json.loads(row["prediction_evidence_json"])
                    if row["prediction_evidence_json"]
                    else None
                ),
                "orientation": row["orientation"],
                "fen": row["final_fen"],
                "model_version": row["model_version"],
                "detection_method": row["detection_method"],
                "final_glare_score": row["glare_score"],
                "final_glare_squares": json.loads(row["glare_squares_json"]),
                "final_sharpness_score": row["sharpness_score"],
                "accepted_frame_count": row["capture_frame_count"],
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            exported += 1
    print(f"Exported {exported} camera diagnostics to {args.output}")


if __name__ == "__main__":
    main()
