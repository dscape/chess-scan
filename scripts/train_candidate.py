#!/usr/bin/env python3
"""Train and register a gated candidate from confirmed human feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from chess_scan.bootstrap import initialize_database
from chess_scan.classifier import (
    INPUT_SIZE,
    NUM_CLASSES,
    DiagramClassifier,
    preprocess_board,
)
from chess_scan.config import Settings
from chess_scan.model_artifact import sha256_file, verify_model_artifact
from square_model import export_onnx, load_fused_onnx
from training_utils import resolve_device

_MIN_CLASS_EXAMPLES_PER_SPLIT = 5
_PERCEPTUAL_HASH_MAX_DISTANCE = 5
_SELECTION_CACHE_BOARDS = 128


class FeedbackBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], *, augment: bool, seed: int) -> None:
        self.rows = rows
        self.augment = augment
        self.rng = random.Random(seed)
        self.labels = [
            np.asarray(json.loads(row["final_labels_json"]), dtype=np.int64) for row in rows
        ]
        self.rejected = [
            np.asarray(json.loads(row["predicted_labels_json"]), dtype=np.int64) for row in rows
        ]
        self.cached_inputs: OrderedDict[int, np.ndarray] | None = None if augment else OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cached_inputs is None:
            inputs = preprocess_board(augment_board(_read_board(self.rows[index]), self.rng))
            images = torch.from_numpy(inputs)
        else:
            cached = self.cached_inputs.pop(index, None)
            if cached is None:
                cached = _preprocessed_uint8(_read_board(self.rows[index]))
            self.cached_inputs[index] = cached
            if len(self.cached_inputs) > _SELECTION_CACHE_BOARDS:
                self.cached_inputs.popitem(last=False)
            images = torch.from_numpy(cached).float().div_(255.0)
        labels = torch.from_numpy(self.labels[index])
        rejected = torch.from_numpy(self.rejected[index])
        return images, labels, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-fraction", type=float, default=0.15)
    parser.add_argument("--gate-fraction", type=float, default=0.15)
    parser.add_argument("--min-boards", type=int, default=100)
    parser.add_argument(
        "--feedback-ids-file",
        type=Path,
        help="JSON array selecting the immutable feedback snapshot for this run",
    )
    parser.add_argument(
        "--corrected-square-weight",
        type=float,
        default=4.0,
        help="Supervised loss multiplier for squares explicitly corrected by a user",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--preference-weight",
        type=float,
        default=0.0,
        help="Experimental chosen-vs-rejected pairwise loss weight; production default is SFT only",
    )
    parser.add_argument(
        "--allow-gate-tie",
        action="store_true",
        help="Allow a non-regressing candidate to proceed to a fresh shadow promotion gate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Candidate registry directory (default: <data-dir>/model-registry)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    settings = Settings.load()
    output_dir = args.output_dir or settings.data_dir / "model-registry"
    database = initialize_database(settings)
    active = database.get_active_model()
    active_path = Path(active["artifact_path"])
    verify_model_artifact(active_path, str(active["metadata_json"]))

    feedback_ids = load_feedback_ids(args.feedback_ids_file)
    rows = database.training_examples(feedback_ids)
    if feedback_ids is not None and len(rows) != len(feedback_ids):
        found = {str(row["feedback_id"]) for row in rows}
        missing = sorted(feedback_ids - found)
        raise SystemExit(f"Feedback snapshot contains unavailable records: {missing}")
    if args.corrected_square_weight < 1:
        raise SystemExit("--corrected-square-weight must be at least 1")
    if len(rows) < args.min_boards:
        raise SystemExit(
            f"Need at least {args.min_boards} consented confirmed boards; found {len(rows)}"
        )
    ensure_class_coverage(rows)
    train_rows, selection_rows, gate_rows, assignments = split_rows(
        rows,
        selection_fraction=args.selection_fraction,
        gate_fraction=args.gate_fraction,
        existing_assignments=database.feedback_split_assignments(),
    )
    if len(train_rows) < 20 or len(selection_rows) < 10 or len(gate_rows) < 10:
        raise SystemExit(
            "Grouped split needs at least 20 training, 10 selection, and 10 gate boards; "
            f"found {len(train_rows)}, {len(selection_rows)}, and {len(gate_rows)}. "
            "Collect feedback across more capture sessions; groups are never split to fill buckets."
        )

    database.save_feedback_split_assignments(assignments)
    quarantined_boards = len(rows) - len(train_rows) - len(selection_rows) - len(gate_rows)

    run_id = uuid.uuid4().hex
    database.start_training_run(
        run_id=run_id,
        base_model_version=str(active["version"]),
        training_example_count=len(rows),
    )
    print(
        f"Training on {len(train_rows)} boards; selecting on {len(selection_rows)}; "
        f"gating on {len(gate_rows)}"
    )
    print(f"Run: {run_id}; device: {device}; preference weight: {args.preference_weight}")
    train_loader = DataLoader(
        FeedbackBoardDataset(train_rows, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    selection_loader = DataLoader(
        FeedbackBoardDataset(selection_rows, augment=False, seed=args.seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_fused_onnx(active_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = class_weights_for_rows(train_rows).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        board_count = 0
        for images, labels, rejected in train_loader:
            batch_size = images.shape[0]
            logits = model(images.reshape(batch_size * 64, 3, INPUT_SIZE, INPUT_SIZE).to(device))
            logits = logits.reshape(batch_size, 64, NUM_CLASSES)
            labels = labels.to(device)
            rejected = rejected.to(device)
            supervised_losses = functional.cross_entropy(
                logits.reshape(-1, NUM_CLASSES),
                labels.reshape(-1),
                weight=class_weights,
                reduction="none",
            ).reshape(batch_size, 64)
            correction_weights = torch.where(
                labels != rejected,
                args.corrected_square_weight,
                1.0,
            )
            supervised_loss = (supervised_losses * correction_weights).sum() / (
                correction_weights.sum()
            )
            loss = supervised_loss
            if args.preference_weight > 0:
                loss = loss + args.preference_weight * preference_loss(logits, labels, rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_size
            board_count += batch_size
        scheduler.step()

        metrics = evaluate_torch(model, selection_loader, device=device)
        print(
            f"epoch={epoch + 1:02d} loss={epoch_loss / board_count:.4f} "
            f"board_exact={metrics['board_exact']:.4f} square={metrics['square_accuracy']:.4f} "
            f"non_empty={metrics['non_empty_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )
        if best_metrics is None or metric_rank(metrics) > metric_rank(best_metrics):
            best_metrics = metrics
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a candidate")
    model.load_state_dict(best_state)
    model.to(device).eval()

    version = datetime.now(UTC).strftime("steps-%Y%m%d%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{version}.pt"
    onnx_path = output_dir / f"{version}.onnx"
    metadata_path = output_dir / f"{version}.json"
    torch.save(
        {
            "state_dict": best_state,
            "version": version,
            "architecture": "fused_tiny_square_cnn",
            "input_size": INPUT_SIZE,
            "num_classes": NUM_CLASSES,
        },
        checkpoint_path,
    )
    export_onnx(model.cpu(), onnx_path)

    artifact_sha256 = sha256_file(onnx_path)
    active_classifier = DiagramClassifier(active_path, version=str(active["version"]))
    candidate_classifier = DiagramClassifier(onnx_path, version=version)
    active_metrics, candidate_metrics = evaluate_onnx_pair(
        active_classifier,
        candidate_classifier,
        gate_rows,
        board_batch_size=args.batch_size,
    )
    if sha256_file(onnx_path) != artifact_sha256:
        raise RuntimeError("Candidate artifact changed during ONNX Runtime evaluation")
    eligible = passes_gates(
        candidate_metrics,
        active_metrics,
        allow_board_exact_tie=args.allow_gate_tie,
    )
    metadata = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "architecture": "fused_tiny_square_cnn",
        "training_boards": len(train_rows),
        "selection_boards": len(selection_rows),
        "gate_boards": len(gate_rows),
        "quarantined_boards": quarantined_boards,
        "seed": args.seed,
        "preference_weight": args.preference_weight,
        "corrected_square_weight": args.corrected_square_weight,
        "feedback_snapshot_sha256": feedback_snapshot_sha256(rows),
        "artifact_sha256": artifact_sha256,
        "active_baseline": active["version"],
        "active_metrics": active_metrics,
        "candidate_metrics": candidate_metrics,
        "eligible_for_promotion": eligible,
        "allow_gate_tie": args.allow_gate_tie,
        "promotion_gates": (
            "board_exact does not regress before fresh shadow evaluation; "
            "square/non-empty/macro-F1 do not regress"
            if args.allow_gate_tie
            else "board_exact improves; square/non-empty/macro-F1 do not regress"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    database.register_candidate(version=version, artifact_path=onnx_path, metadata=metadata)
    database.complete_training_run(
        run_id=run_id,
        candidate_model_version=version,
        metrics=metadata,
    )

    print(json.dumps(metadata, indent=2))
    print(f"Registered candidate {version} at {onnx_path}")
    if eligible:
        print(
            "Eligible. Review metrics, then run: "
            f"python scripts/promote_model.py {version} --confirm"
        )
    else:
        print("Not eligible for promotion; the active model remains unchanged")


def load_feedback_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
        raise SystemExit("Feedback snapshot must be a JSON array of feedback ids")
    feedback_ids = set(payload)
    if len(feedback_ids) != len(payload):
        raise SystemExit("Feedback snapshot contains duplicate ids")
    return feedback_ids


def feedback_snapshot_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(str(row["feedback_id"]) for row in rows))
    return hashlib.sha256(payload.encode()).hexdigest()


def preference_loss(
    logits: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
) -> torch.Tensor:
    log_probs = functional.log_softmax(logits, dim=-1)
    chosen_scores = log_probs.gather(2, chosen.unsqueeze(-1)).squeeze(-1)
    rejected_scores = log_probs.gather(2, rejected.unsqueeze(-1)).squeeze(-1)
    changed = chosen != rejected
    changed_count = changed.sum(dim=1)
    valid = changed_count > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    score_delta = ((chosen_scores - rejected_scores) * changed).sum(dim=1)
    score_delta = score_delta[valid] / changed_count[valid]
    return functional.softplus(-score_delta).mean()


def evaluate_torch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    with torch.no_grad():
        for images, labels, _rejected in loader:
            batch_size = images.shape[0]
            logits = model(images.reshape(batch_size * 64, 3, INPUT_SIZE, INPUT_SIZE).to(device))
            predicted = logits.argmax(dim=1).reshape(batch_size, 64).cpu().tolist()
            predictions.extend(predicted)
            targets.extend(labels.tolist())
    return classification_metrics(predictions, targets)


def evaluate_onnx_pair(
    active: DiagramClassifier,
    candidate: DiagramClassifier,
    rows: list[dict[str, Any]],
    *,
    board_batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    active_predictions: list[list[int]] = []
    candidate_predictions: list[list[int]] = []
    targets: list[list[int]] = []
    for offset in range(0, len(rows), board_batch_size):
        batch_rows = rows[offset : offset + board_batch_size]
        board_inputs = np.stack([_preprocessed_uint8(_read_board(row)) for row in batch_rows])
        normalized = board_inputs.reshape(-1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        normalized /= 255.0
        active_predictions.extend(
            prediction.labels for prediction in active.predict_preprocessed(normalized)
        )
        candidate_predictions.extend(
            prediction.labels for prediction in candidate.predict_preprocessed(normalized)
        )
        targets.extend(json.loads(row["final_labels_json"]) for row in batch_rows)
    return (
        classification_metrics(active_predictions, targets),
        classification_metrics(candidate_predictions, targets),
    )


def classification_metrics(
    predictions: list[list[int]],
    targets: list[list[int]],
) -> dict[str, float]:
    predicted = np.asarray(predictions, dtype=np.int64)
    expected = np.asarray(targets, dtype=np.int64)
    correct = predicted == expected
    non_empty = expected != 0
    f1_scores: list[float] = []
    for class_id in range(NUM_CLASSES):
        true_positive = int(((predicted == class_id) & (expected == class_id)).sum())
        false_positive = int(((predicted == class_id) & (expected != class_id)).sum())
        false_negative = int(((predicted != class_id) & (expected == class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "board_exact": float(correct.all(axis=1).mean()),
        "square_accuracy": float(correct.mean()),
        "non_empty_accuracy": float(correct[non_empty].mean()) if non_empty.any() else 0.0,
        "macro_f1": float(np.mean(f1_scores)),
    }


def split_rows(
    rows: list[dict[str, Any]],
    *,
    selection_fraction: float,
    gate_fraction: float,
    existing_assignments: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    if selection_fraction <= 0 or gate_fraction <= 0:
        raise ValueError("Selection and gate fractions must be positive")
    if selection_fraction + gate_fraction >= 1:
        raise ValueError("Selection and gate fractions must leave room for training data")

    groups = _connected_feedback_groups(rows)
    has_existing = any(
        str(row["feedback_id"]) in existing_assignments for group in groups for row in group
    )
    if has_existing:
        buckets, assignments = _extend_persisted_splits(
            groups,
            existing_assignments=existing_assignments,
            selection_fraction=selection_fraction,
            gate_fraction=gate_fraction,
        )
        _ensure_split_class_coverage(buckets)
    else:
        buckets = _initial_grouped_split(
            groups,
            row_count=len(rows),
            selection_fraction=selection_fraction,
            gate_fraction=gate_fraction,
        )
        assignments = {
            str(row["feedback_id"]): split
            for split in ("train", "selection", "gate")
            for row in buckets[split]
        }

    return buckets["train"], buckets["selection"], buckets["gate"], assignments


def _extend_persisted_splits(
    groups: list[list[dict[str, Any]]],
    *,
    existing_assignments: dict[str, str],
    selection_fraction: float,
    gate_fraction: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "selection": [],
        "gate": [],
        "quarantine": [],
    }
    assignments: dict[str, str] = {}
    for group in groups:
        persisted = {
            existing_assignments[str(row["feedback_id"])]
            for row in group
            if str(row["feedback_id"]) in existing_assignments
        }
        if not persisted:
            destination = _stable_split(
                _group_key(group),
                selection_fraction=selection_fraction,
                gate_fraction=gate_fraction,
            )
        elif len(persisted) == 1 and "quarantine" not in persisted:
            destination = persisted.pop()
        else:
            destination = "quarantine"

        buckets[destination].extend(group)
        for row in group:
            feedback_id = str(row["feedback_id"])
            if feedback_id not in existing_assignments:
                assignments[feedback_id] = destination
    return buckets, assignments


def _initial_grouped_split(
    groups: list[list[dict[str, Any]]],
    *,
    row_count: int,
    selection_fraction: float,
    gate_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    group_class_counts = [_class_counts(group) for group in groups]
    class_group_support = Counter(
        class_id for counts in group_class_counts for class_id in counts if counts[class_id] > 0
    )
    targets = {
        "train": row_count * (1.0 - selection_fraction - gate_fraction),
        "selection": row_count * selection_fraction,
        "gate": row_count * gate_fraction,
    }
    best: dict[str, list[dict[str, Any]]] | None = None
    best_error = float("inf")
    for attempt in range(256):
        buckets: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "selection": [],
            "gate": [],
            "quarantine": [],
        }
        bucket_counts = {name: Counter() for name in ("train", "selection", "gate")}
        ordered = sorted(
            zip(groups, group_class_counts, strict=True),
            key=lambda item: (
                min(class_group_support[class_id] for class_id in item[1]),
                _stable_hash(f"{attempt}:{_group_key(item[0])}"),
            ),
        )
        for group, counts in ordered:
            candidates: list[tuple[float, int, str]] = []
            for name in ("train", "selection", "gate"):
                coverage_gain = sum(
                    min(
                        counts[class_id],
                        max(0, _MIN_CLASS_EXAMPLES_PER_SPLIT - bucket_counts[name][class_id]),
                    )
                    / (_MIN_CLASS_EXAMPLES_PER_SPLIT * class_group_support[class_id])
                    for class_id in counts
                )
                remaining = targets[name] - len(buckets[name])
                capacity_score = remaining / max(targets[name], 1.0)
                overflow = max(0.0, len(group) - remaining) / max(targets[name], 1.0)
                score = 100.0 * coverage_gain + capacity_score - 10.0 * overflow
                tie_breaker = _stable_hash(f"{attempt}:{name}:{_group_key(group)}")
                candidates.append((score, tie_breaker, name))
            destination = max(candidates)[2]
            buckets[destination].extend(group)
            bucket_counts[destination].update(counts)

        if not _split_has_class_coverage(buckets):
            continue
        error = sum(abs(len(buckets[name]) - targets[name]) for name in targets)
        if error < best_error:
            best = buckets
            best_error = error

    if best is None:
        raise SystemExit(
            "Could not make session- and image-grouped splits with at least "
            f"{_MIN_CLASS_EXAMPLES_PER_SPLIT} examples of every class in each split. "
            "Collect more diverse feedback across independent sessions."
        )
    return best


def _stable_split(
    group_key: str,
    *,
    selection_fraction: float,
    gate_fraction: float,
) -> str:
    bucket = _stable_hash(group_key) / float(2**64)
    if bucket < gate_fraction:
        return "gate"
    if bucket < gate_fraction + selection_fraction:
        return "selection"
    return "train"


def _ensure_split_class_coverage(buckets: dict[str, list[dict[str, Any]]]) -> None:
    insufficient = _split_class_gaps(buckets)
    if any(insufficient.values()):
        raise SystemExit(f"Persisted split has inadequate class support: {insufficient}")


def _split_has_class_coverage(buckets: dict[str, list[dict[str, Any]]]) -> bool:
    return not any(_split_class_gaps(buckets).values())


def _split_class_gaps(
    buckets: dict[str, list[dict[str, Any]]],
) -> dict[str, list[int]]:
    gaps: dict[str, list[int]] = {}
    for name in ("train", "selection", "gate"):
        counts = _class_counts(buckets[name])
        gaps[name] = [
            class_id
            for class_id in range(NUM_CLASSES)
            if counts[class_id] < _MIN_CLASS_EXAMPLES_PER_SPLIT
        ]
    return gaps


def ensure_class_coverage(rows: list[dict[str, Any]]) -> None:
    counts = _class_counts(rows)
    required = _MIN_CLASS_EXAMPLES_PER_SPLIT * 3
    insufficient = [class_id for class_id in range(NUM_CLASSES) if counts[class_id] < required]
    if insufficient:
        raise SystemExit(
            f"Training feedback needs at least {required} examples of every square class; "
            f"insufficient classes: {insufficient}"
        )


def _connected_feedback_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    sessions: dict[str, int] = {}
    source_hashes: dict[str, int] = {}
    perceptual_hashes: list[int] = []
    bands: list[dict[int, list[int]]] = [{} for _ in range(8)]
    for index, row in enumerate(rows):
        session = row.get("client_session_id")
        if session:
            previous = sessions.setdefault(str(session), index)
            union(index, previous)
        source_hash = str(row.get("image_sha256") or "")
        if source_hash:
            previous = source_hashes.setdefault(source_hash, index)
            union(index, previous)

        fingerprint = _perceptual_hash(Path(row["rectified_image_path"]))
        candidates: set[int] = set()
        for band_index, band in enumerate(bands):
            band_value = (fingerprint >> (band_index * 8)) & 0xFF
            candidates.update(band.get(band_value, []))
        for candidate in candidates:
            if (fingerprint ^ perceptual_hashes[candidate]).bit_count() <= (
                _PERCEPTUAL_HASH_MAX_DISTANCE
            ):
                union(index, candidate)
        for band_index, band in enumerate(bands):
            band_value = (fingerprint >> (band_index * 8)) & 0xFF
            band.setdefault(band_value, []).append(index)
        perceptual_hashes.append(fingerprint)

    connected: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        connected.setdefault(find(index), []).append(row)
    return list(connected.values())


def _perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read board crop: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequencies = cv2.dct(resized)[:8, :8].ravel()
    median = float(np.median(low_frequencies[1:]))
    fingerprint = 0
    for index, value in enumerate(low_frequencies):
        if value > median:
            fingerprint |= 1 << index
    return fingerprint


def _class_counts(rows: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in rows:
        counts.update(json.loads(row["final_labels_json"]))
    return counts


def _group_key(rows: list[dict[str, Any]]) -> str:
    return min(str(row["feedback_id"]) for row in rows)


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def _read_board(row: dict[str, Any]) -> np.ndarray:
    path = Path(row["rectified_image_path"])
    board = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if board is None:
        raise ValueError(f"Cannot read board crop: {path}")
    return board


def _preprocessed_uint8(board: np.ndarray) -> np.ndarray:
    inputs = preprocess_board(board)
    return np.rint(inputs * 255.0).astype(np.uint8)


def class_weights_for_rows(rows: list[dict[str, Any]]) -> torch.Tensor:
    counts = _class_counts(rows)
    values = torch.tensor(
        [counts[class_id] for class_id in range(NUM_CLASSES)], dtype=torch.float32
    )
    weights = torch.rsqrt(values.clamp_min(1.0))
    return weights / weights.mean()


def augment_board(board: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = board.shape[:2]
    angle = rng.uniform(-1.2, 1.2)
    scale = rng.uniform(0.985, 1.015)
    center = (width / 2.0, height / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle, scale)
    transform[:, 2] += [rng.uniform(-3, 3), rng.uniform(-3, 3)]
    augmented = cv2.warpAffine(
        board,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    contrast = rng.uniform(0.82, 1.18)
    brightness = rng.uniform(-18, 18)
    augmented = np.clip(augmented.astype(np.float32) * contrast + brightness, 0, 255).astype(
        np.uint8
    )
    if rng.random() < 0.35:
        augmented = cv2.GaussianBlur(augmented, (3, 3), rng.uniform(0.2, 1.0))
    if rng.random() < 0.35:
        quality = rng.randint(55, 92)
        ok, encoded = cv2.imencode(".jpg", augmented, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            augmented = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return augmented


def passes_gates(
    candidate: dict[str, float],
    active: dict[str, float],
    *,
    allow_board_exact_tie: bool = False,
) -> bool:
    board_exact_passes = (
        candidate["board_exact"] >= active["board_exact"]
        if allow_board_exact_tie
        else candidate["board_exact"] > active["board_exact"]
    )
    return (
        board_exact_passes
        and candidate["square_accuracy"] >= active["square_accuracy"]
        and candidate["non_empty_accuracy"] >= active["non_empty_accuracy"]
        and candidate["macro_f1"] >= active["macro_f1"]
    )


def metric_rank(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["board_exact"],
        metrics["non_empty_accuracy"],
        metrics["macro_f1"],
        metrics["square_accuracy"],
    )


if __name__ == "__main__":
    main()
