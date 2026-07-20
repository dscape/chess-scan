"""Shared classification metrics for model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from chess_scan.board import CLASS_NAMES


def classification_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    correct = predicted == expected
    occupied = expected != 0
    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = expected == class_index
        per_class[class_name or "empty"] = {
            "correct": int(correct[selected].sum()),
            "total": int(selected.sum()),
            "accuracy": float(correct[selected].mean()),
        }
    return {
        "correct": int(correct.sum()),
        "total": len(expected),
        "accuracy": float(correct.mean()),
        "non_empty_correct": int(correct[occupied].sum()),
        "non_empty_total": int(occupied.sum()),
        "non_empty_accuracy": float(correct[occupied].mean()),
        "per_class": per_class,
    }
