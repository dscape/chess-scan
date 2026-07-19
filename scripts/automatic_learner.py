#!/usr/bin/env python3
"""Continuously train, shadow-evaluate, and promote improving feedback models."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2

from chess_scan.bootstrap import initialize_database
from chess_scan.classifier import DiagramClassifier
from chess_scan.config import Settings
from chess_scan.database import Database
from chess_scan.learning import (
    INITIAL_TRAINING_BOARDS,
    MAX_BOARDS_PER_CLIENT,
    MAX_SHADOW_BOARDS,
    MIN_SHADOW_BOARDS,
    MIN_SHADOW_CLIENTS,
    NEW_TRAINING_BOARDS,
    compare_labels,
    diverse_shadow_rows,
    perceptual_hash,
    promotion_decision,
    summarize_shadow,
)


class BenchmarkUnavailableError(RuntimeError):
    """A benchmark could not run, so the same candidate should be retried later."""


def main() -> None:
    args = parse_args()
    settings = Settings.load()
    if args.once:
        with learner_lock(settings.data_dir):
            advance(settings, args)
        return

    while True:
        try:
            with learner_lock(settings.data_dir):
                advance(settings, args)
        except BlockingIOError:
            print("Another learner holds the production lock; waiting")
        except Exception:
            traceback.print_exc()
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Advance once and exit")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_POLL_SECONDS", 6 * 60 * 60),
    )
    parser.add_argument(
        "--min-total-boards",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_MIN_TOTAL_BOARDS", INITIAL_TRAINING_BOARDS),
    )
    parser.add_argument(
        "--min-new-boards",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_MIN_NEW_BOARDS", NEW_TRAINING_BOARDS),
    )
    parser.add_argument(
        "--min-shadow-boards",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_MIN_SHADOW_BOARDS", MIN_SHADOW_BOARDS),
    )
    parser.add_argument(
        "--max-shadow-boards",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_MAX_SHADOW_BOARDS", MAX_SHADOW_BOARDS),
    )
    parser.add_argument(
        "--min-shadow-clients",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_MIN_SHADOW_CLIENTS", MIN_SHADOW_CLIENTS),
    )
    parser.add_argument(
        "--max-boards-per-client",
        type=int,
        default=_env_int("CHESS_SCAN_LEARNER_MAX_BOARDS_PER_CLIENT", MAX_BOARDS_PER_CLIENT),
    )
    parser.add_argument(
        "--skip-fixed-qa",
        action="store_true",
        help="Development-only: skip official online and photo gates",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to train_candidate.py",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.min_total_boards <= 0 or args.min_new_boards <= 0:
        parser.error("training thresholds must be positive")
    if not 0 < args.min_shadow_boards <= args.max_shadow_boards:
        parser.error("shadow thresholds must be positive and ordered")
    return args


def advance(settings: Settings, args: argparse.Namespace) -> None:
    database = initialize_database(settings)
    cycle = database.active_learning_cycle()
    if cycle is None:
        cycle = start_cycle(database, settings, args)
        if cycle is None:
            return

    if cycle["state"] == "training":
        train_cycle(database, settings, cycle, args)
        cycle = database.get_learning_cycle(str(cycle["id"]))
    if cycle["state"] == "benchmarking":
        benchmark_cycle(database, settings, cycle, args)
        cycle = database.get_learning_cycle(str(cycle["id"]))
    if cycle["state"] == "shadowing":
        evaluate_shadow(database, cycle, args)


def start_cycle(
    database: Database, settings: Settings, args: argparse.Namespace
) -> dict[str, Any] | None:
    snapshot = database.learning_feedback_snapshot(
        min_total_boards=args.min_total_boards,
        min_new_boards=args.min_new_boards,
        max_boards_per_client=args.max_boards_per_client,
    )
    if snapshot is None:
        status = database.learning_status()
        print(
            f"Collecting feedback: {status['learning_progress']}/{status['learning_target']} "
            f"boards for the next candidate"
        )
        return None

    cycle_id = uuid.uuid4().hex
    cycle = database.create_learning_cycle(
        cycle_id=cycle_id,
        base_model_version=str(database.get_active_model()["version"]),
        accepted_feedback_ids=snapshot["accepted"],
        batch_feedback_ids=snapshot["batch"],
        shadow_target_boards=args.min_shadow_boards,
    )
    write_feedback_snapshot(settings, cycle_id, snapshot["accepted"] + snapshot["batch"])
    print(
        f"Started cycle {cycle_id} with {len(snapshot['accepted'])} accepted and "
        f"{len(snapshot['batch'])} new boards"
    )
    return cycle


def train_cycle(
    database: Database,
    settings: Settings,
    cycle: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cycle_id = str(cycle["id"])
    feedback_ids = database.learning_cycle_feedback_ids(cycle_id)
    snapshot_path = write_feedback_snapshot(settings, cycle_id, feedback_ids)
    previous = database.latest_candidate()
    forwarded = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_candidate.py"),
        "--min-boards",
        str(args.min_total_boards),
        "--feedback-ids-file",
        str(snapshot_path),
        "--allow-gate-tie",
        *forwarded,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        database.reject_learning_cycle(
            cycle_id,
            reason=f"training command exited {result.returncode}",
            metrics={"stage": "training", "returncode": result.returncode},
        )
        print(f"Rejected cycle {cycle_id}: training failed")
        return

    candidate = database.latest_candidate()
    if candidate is None or (previous is not None and candidate["version"] == previous["version"]):
        database.reject_learning_cycle(
            cycle_id,
            reason="training did not register a new candidate",
            metrics={"stage": "training"},
        )
        return
    metadata = json.loads(str(candidate["metadata_json"]))
    if metadata.get("active_baseline") != cycle["base_model_version"]:
        database.reject_learning_cycle(
            cycle_id,
            reason="active model changed while the candidate was training",
            metrics={"stage": "training", "candidate": metadata},
        )
        return
    if not bool(metadata.get("eligible_for_promotion")):
        database.reject_learning_cycle(
            cycle_id,
            reason="candidate failed its grouped feedback gate",
            metrics={"stage": "training", "candidate": metadata},
        )
        print(f"Rejected {candidate['version']}: grouped feedback gate failed")
        return
    database.set_learning_candidate(cycle_id, str(candidate["version"]))
    print(f"Candidate {candidate['version']} passed training and selection")


def benchmark_cycle(
    database: Database,
    settings: Settings,
    cycle: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cycle_id = str(cycle["id"])
    candidate = database.get_model(str(cycle["candidate_model_version"]))
    if args.skip_fixed_qa:
        metrics = {"fixed_qa": "skipped"}
    else:
        try:
            metrics = run_fixed_qa(
                model_path=Path(candidate["artifact_path"]),
                output_dir=cycle_directory(settings, cycle_id),
                cache_dir=settings.data_dir / "qa-cache",
            )
        except BenchmarkUnavailableError as exc:
            print(f"Fixed QA unavailable for {candidate['version']}: {exc}; retrying later")
            return
        if not metrics["passed"]:
            database.reject_learning_cycle(
                cycle_id,
                reason="candidate failed an immutable official benchmark",
                metrics=metrics,
            )
            print(f"Rejected {candidate['version']}: fixed QA failed")
            return
    database.start_shadowing(cycle_id, metrics)
    print(f"Candidate {candidate['version']} is now evaluating on fresh confirmations")


def evaluate_shadow(
    database: Database,
    cycle: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    cycle_id = str(cycle["id"])
    active_version = str(database.get_active_model()["version"])
    if active_version != cycle["base_model_version"]:
        database.reject_learning_cycle(
            cycle_id,
            reason=f"active model changed from {cycle['base_model_version']} to {active_version}",
            metrics={"stage": "shadowing"},
        )
        return
    candidate = database.get_model(str(cycle["candidate_model_version"]))
    classifier = DiagramClassifier(
        Path(candidate["artifact_path"]),
        version=str(candidate["version"]),
    )
    for row in database.shadow_examples(cycle_id):
        board = cv2.imread(str(row["rectified_image_path"]), cv2.IMREAD_COLOR)
        if board is None:
            continue
        expected = [int(value) for value in json.loads(row["final_labels_json"])]
        active = [int(value) for value in json.loads(row["predicted_labels_json"])]
        prediction = classifier.predict(board)
        comparison = compare_labels(active, prediction.labels, expected)
        database.record_shadow_evaluation(
            cycle_id=cycle_id,
            feedback_id=str(row["feedback_id"]),
            perceptual_hash=perceptual_hash(board),
            candidate_labels=prediction.labels,
            active_square_errors=comparison.active_square_errors,
            candidate_square_errors=comparison.candidate_square_errors,
            active_non_empty_errors=comparison.active_non_empty_errors,
            candidate_non_empty_errors=comparison.candidate_non_empty_errors,
            active_board_exact=comparison.active_board_exact,
            candidate_board_exact=comparison.candidate_board_exact,
        )

    diverse = diverse_shadow_rows(
        database.shadow_evaluations(cycle_id),
        max_per_client=args.max_boards_per_client,
    )
    summary = summarize_shadow(diverse)
    passed, reason = promotion_decision(
        summary,
        minimum_boards=args.min_shadow_boards,
        minimum_clients=args.min_shadow_clients,
    )
    metrics = json.loads(str(cycle["metrics_json"]))
    metrics["shadow"] = summary.as_dict()
    metrics["shadow_decision"] = reason
    print(
        f"Shadow {cycle['candidate_model_version']}: {summary.boards}/"
        f"{args.min_shadow_boards} diverse boards, {summary.clients} clients; {reason}"
    )
    if passed:
        version = database.promote_learning_cycle(cycle_id, reason=reason, metrics=metrics)
        print(f"Promoted {version}; new scans will load it automatically")
    elif summary.boards >= args.max_shadow_boards:
        database.reject_learning_cycle(cycle_id, reason=reason, metrics=metrics)
        print(f"Rejected {cycle['candidate_model_version']} after shadow evaluation")


def run_fixed_qa(
    *,
    model_path: Path,
    output_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    commands = {
        "online": [
            sys.executable,
            str(scripts / "evaluate_online_examples.py"),
            "--model",
            str(model_path),
            "--cache-dir",
            str(cache_dir),
        ],
        "photo": [
            sys.executable,
            str(scripts / "evaluate_photo_stress.py"),
            "--model",
            str(model_path),
            "--cache-dir",
            str(cache_dir),
        ],
    }
    metrics: dict[str, Any] = {"passed": True}
    for name, command in commands.items():
        output_path = output_dir / f"{name}-qa.json"
        output_path.unlink(missing_ok=True)
        result = subprocess.run([*command, "--output", str(output_path)], check=False)
        if not output_path.exists():
            raise BenchmarkUnavailableError(f"{name} command exited {result.returncode}")
        payload = json.loads(output_path.read_text())
        metrics[name] = _qa_summary(name, payload)
        if result.returncode != 0:
            metrics["passed"] = False
    return metrics


def _qa_summary(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "online":
        return dict(payload["combined"])
    return {
        "classifier": payload.get("classifier"),
        "pipeline": payload.get("pipeline"),
    }


def write_feedback_snapshot(
    settings: Settings,
    cycle_id: str,
    feedback_ids: list[str],
) -> Path:
    path = cycle_directory(settings, cycle_id) / "feedback-ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feedback_ids, indent=2))
    return path


def cycle_directory(settings: Settings, cycle_id: str) -> Path:
    return settings.data_dir / "learning-cycles" / cycle_id


@contextmanager
def learner_lock(data_dir: Path) -> Iterator[None]:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "automatic-learner.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


if __name__ == "__main__":
    main()
