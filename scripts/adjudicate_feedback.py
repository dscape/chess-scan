#!/usr/bin/env python3
"""Append a reviewed correction to an immutable feedback event."""

from __future__ import annotations

import argparse
import json
import uuid

from chess_scan.board import CLASS_NAMES, build_full_fen, validate_full_fen, validate_labels
from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback_id")
    parser.add_argument(
        "--set-square",
        action="append",
        required=True,
        metavar="INDEX=PIECE",
        help="Set an image-order square to empty, P/N/B/R/Q/K, or p/n/b/r/q/k",
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    database = initialize_database(Settings.load())
    feedback = database.feedback_for_adjudication(args.feedback_id)
    labels = [int(value) for value in json.loads(feedback["final_labels_json"])]
    before_fen = str(feedback["final_fen"])
    changes = []
    for assignment in args.set_square:
        index, class_id = parse_assignment(assignment)
        before = CLASS_NAMES[labels[index]]
        labels[index] = class_id
        changes.append({"index": index, "before": before, "after": CLASS_NAMES[class_id]})

    validate_labels(labels)
    full_fen = build_full_fen(
        labels,
        orientation=str(feedback["orientation"]),
        side_to_move=str(feedback["side_to_move"]),
        castling=str(feedback["castling"]),
        en_passant=str(feedback["en_passant"]),
    )
    validate_full_fen(full_fen)
    adjudication_id = uuid.uuid4().hex
    changed_squares = database.append_feedback_adjudication(
        adjudication_id=adjudication_id,
        feedback_id=args.feedback_id,
        labels=labels,
        full_fen=full_fen,
        reason=args.reason,
    )
    print(
        json.dumps(
            {
                "adjudication_id": adjudication_id,
                "feedback_id": args.feedback_id,
                "changes": changes,
                "before_fen": before_fen,
                "corrected_fen": full_fen,
                "changed_squares_from_model": changed_squares,
            },
            indent=2,
        )
    )


def parse_assignment(value: str) -> tuple[int, int]:
    try:
        raw_index, symbol = value.split("=", 1)
        index = int(raw_index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("square assignment must be INDEX=PIECE") from exc
    if not 0 <= index < 64:
        raise argparse.ArgumentTypeError("square index must be between 0 and 63")
    try:
        class_id = CLASS_NAMES.index(symbol)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"unknown piece symbol: {symbol}") from exc
    return index, class_id


if __name__ == "__main__":
    main()
