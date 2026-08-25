#!/usr/bin/env python3
"""Register a fixed-QA candidate and begin fresh-feedback shadow evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from chess_scan.bootstrap import initialize_database
from chess_scan.classifier import load_runtime_classifier, read_model_metadata
from chess_scan.config import Settings
from chess_scan.learning import MAX_BOARDS_PER_CLIENT, PERCEPTUAL_HASH_MAX_DISTANCE, perceptual_hash
from chess_scan.model_artifact import verify_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", help="Versioned artifact and metadata stem in the model directory"
    )
    parser.add_argument(
        "--training-feedback-ids-file",
        type=Path,
        required=True,
        help="JSON array containing the capped, deduplicated development feedback IDs",
    )
    parser.add_argument(
        "--split-assignments-file",
        type=Path,
        help="Optional JSON object mapping audited feedback IDs to train/selection/gate",
    )
    parser.add_argument("--shadow-target", type=int, default=40)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement that fixed QA and development grouping were reviewed",
    )
    args = parser.parse_args()
    if not args.confirm:
        parser.error("Pass --confirm after reviewing the fixed and grouped development gates")
    if args.shadow_target < 40:
        parser.error("Shadow target cannot be lower than 40 boards")

    settings = Settings.load()
    database = initialize_database(settings)
    if database.active_learning_cycle() is not None:
        raise SystemExit("A learning cycle is already active")

    model_path = settings.model_dir / f"{args.version}.onnx"
    metadata_path = settings.model_dir / f"{args.version}.json"
    metadata = read_model_metadata(metadata_path)
    if metadata.get("version") != args.version:
        raise SystemExit("Candidate metadata version does not match its filename")
    if not bool(metadata.get("eligible_for_shadow")) or not bool(
        metadata.get("fixed_qa", {}).get("passed")
    ):
        raise SystemExit("Candidate metadata has not passed the shadow-entry gates")
    if bool(metadata.get("eligible_for_promotion")):
        raise SystemExit("Fresh-shadow candidates must not be pre-marked for promotion")
    verify_model_artifact(model_path, metadata)
    load_runtime_classifier(model_path, version=args.version, metadata=metadata)

    active = database.get_active_model()
    if active["version"] != metadata.get("base_model"):
        raise SystemExit(
            f"Candidate expects {metadata.get('base_model')}; active model is {active['version']}"
        )
    feedback_ids = _feedback_ids(args.training_feedback_ids_file)
    _validate_training_snapshot(database.training_examples(feedback_ids), feedback_ids)
    assignments = _split_assignments(args.split_assignments_file)
    if assignments and any(assignments.get(feedback_id) != "train" for feedback_id in feedback_ids):
        raise SystemExit("Every candidate training ID must have an immutable train assignment")

    try:
        registered = database.get_model(args.version)
    except KeyError:
        database.register_candidate(
            version=args.version,
            artifact_path=model_path,
            metadata=metadata,
        )
    else:
        if Path(registered["artifact_path"]).resolve() != model_path.resolve():
            raise SystemExit("Candidate version is already registered to another artifact")
        verify_model_artifact(model_path, str(registered["metadata_json"]))

    database.save_feedback_split_assignments(assignments)
    cycle_id = uuid.uuid4().hex
    database.create_learning_cycle(
        cycle_id=cycle_id,
        base_model_version=str(active["version"]),
        accepted_feedback_ids=[],
        batch_feedback_ids=sorted(feedback_ids),
        shadow_target_boards=args.shadow_target,
    )
    database.set_learning_candidate(cycle_id, args.version)
    database.start_shadowing(
        cycle_id,
        {
            "fixed_qa": metadata["fixed_qa"],
            "development_replay": metadata.get("development_replay"),
            "candidate_metadata_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "manual_shadow_entry": True,
        },
    )
    print(
        f"Candidate {args.version} entered shadow cycle {cycle_id}; "
        f"needs {args.shadow_target} later diverse boards before promotion"
    )


def _feedback_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
        raise SystemExit("Training feedback file must be a JSON array of IDs")
    values = set(payload)
    if not values or len(values) != len(payload):
        raise SystemExit("Training feedback IDs must be non-empty and unique")
    return values


def _split_assignments(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or any(
        not isinstance(feedback_id, str) or split not in {"train", "selection", "gate"}
        for feedback_id, split in payload.items()
    ):
        raise SystemExit("Split assignments must map feedback IDs to train/selection/gate")
    return {str(feedback_id): str(split) for feedback_id, split in payload.items()}


def _validate_training_snapshot(
    rows: list[dict[str, Any]],
    feedback_ids: set[str],
) -> None:
    if len(rows) != len(feedback_ids):
        found = {str(row["feedback_id"]) for row in rows}
        raise SystemExit(f"Training feedback is unavailable: {sorted(feedback_ids - found)}")
    client_counts: Counter[str] = Counter()
    image_hashes: set[str] = set()
    fingerprints: list[int] = []
    for row in rows:
        client = str(row.get("client_session_id") or "anonymous")
        client_counts[client] += 1
        if client_counts[client] > MAX_BOARDS_PER_CLIENT:
            raise SystemExit(f"Training snapshot exceeds the installation cap: {client}")
        image_hash = str(row["image_sha256"])
        if image_hash in image_hashes:
            raise SystemExit("Training snapshot contains duplicate source images")
        image_hashes.add(image_hash)
        board = cv2.imread(str(row["rectified_image_path"]), cv2.IMREAD_COLOR)
        if board is None:
            raise SystemExit(f"Training board is unavailable: {row['feedback_id']}")
        fingerprint = int(perceptual_hash(board), 16)
        if any(
            (fingerprint ^ previous).bit_count() <= PERCEPTUAL_HASH_MAX_DISTANCE
            for previous in fingerprints
        ):
            raise SystemExit("Training snapshot contains a perceptually duplicate board")
        fingerprints.append(fingerprint)


if __name__ == "__main__":
    main()
