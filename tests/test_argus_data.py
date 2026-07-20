from __future__ import annotations

from chess_scan.argus_data import argus_pair_decision, labels_from_board_filename


def test_labels_from_chess_positions_filename() -> None:
    labels = labels_from_board_filename("r3k2r-8-8-8-8-8-8-R3K2R.jpeg")

    assert len(labels) == 64
    assert labels[:8] == [10, 0, 0, 0, 12, 0, 0, 10]
    assert labels[-8:] == [4, 0, 0, 0, 6, 0, 0, 4]


def test_argus_gate_rejects_any_held_out_class_regression() -> None:
    per_class = {
        name: {"correct": 10, "total": 10, "accuracy": 1.0}
        for name in ("empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
    }
    active = {
        "chess_positions_test": {
            "accuracy": 1.0,
            "non_empty_accuracy": 1.0,
            "per_class": per_class,
        },
        "synthetic_replay": {"correct": 100},
    }
    candidate = {
        "chess_positions_test": {
            "accuracy": 0.99,
            "non_empty_accuracy": 0.99,
            "per_class": {name: dict(metrics) for name, metrics in per_class.items()},
        },
        "synthetic_replay": {"correct": 100},
    }
    candidate["chess_positions_test"]["per_class"]["q"]["correct"] = 9

    passed, reasons = argus_pair_decision(active, candidate)

    assert passed is False
    assert "chess_positions_test accuracy regressed" in reasons
    assert "chess_positions_test class q regressed" in reasons
