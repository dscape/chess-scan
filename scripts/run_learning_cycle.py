#!/usr/bin/env python3
"""Run one micro-batched learning cycle and optionally promote a gated winner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-total-boards", type=int, default=100)
    parser.add_argument("--min-new-boards", type=int, default=40)
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Activate the candidate only when train_candidate's untouched gate passes",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to train_candidate.py",
    )
    args = parser.parse_args()

    settings = Settings.load()
    database = initialize_database(settings)
    progress = database.learning_cycle_progress()
    if progress["total_training_boards"] < args.min_total_boards:
        print(
            f"Waiting for {args.min_total_boards} total boards; "
            f"have {progress['total_training_boards']}"
        )
        return
    if progress["new_training_boards"] < args.min_new_boards:
        print(
            f"Waiting for {args.min_new_boards} new boards; "
            f"have {progress['new_training_boards']} since the last completed run"
        )
        return

    previous = database.latest_candidate()
    forwarded = args.training_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    train_script = Path(__file__).resolve().parent / "train_candidate.py"
    command = [
        sys.executable,
        str(train_script),
        "--min-boards",
        str(args.min_total_boards),
        *forwarded,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

    candidate = database.latest_candidate()
    if candidate is None or (previous is not None and candidate["version"] == previous["version"]):
        raise RuntimeError("Training completed without registering a new candidate")
    metadata = json.loads(candidate["metadata_json"])
    eligible = bool(metadata.get("eligible_for_promotion"))
    print(
        json.dumps(
            {
                "candidate": candidate["version"],
                "eligible_for_promotion": eligible,
                "candidate_metrics": metadata.get("candidate_metrics"),
                "active_metrics": metadata.get("active_metrics"),
            },
            indent=2,
        )
    )

    if eligible and args.auto_promote:
        database.promote_model(str(candidate["version"]))
        print(f"Promoted {candidate['version']}; new scan requests will load it automatically")
    elif eligible:
        print(
            "Candidate passed. Promote after review with: "
            f"python scripts/promote_model.py {candidate['version']} --confirm"
        )
    else:
        print("Candidate failed the promotion gate; the active model is unchanged")


if __name__ == "__main__":
    main()
