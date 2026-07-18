#!/usr/bin/env python3
"""Train and register a gated candidate from confirmed human feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from collections import Counter
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
from chess_scan.classifier import DiagramClassifier, preprocess_square_crops, split_board_squares
from chess_scan.config import Settings
from square_model import INPUT_SIZE, NUM_CLASSES, export_onnx, load_fused_onnx


class FeedbackBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], *, augment: bool, seed: int) -> None:
        self.rows = rows
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        board = cv2.imread(str(row["rectified_image_path"]), cv2.IMREAD_COLOR)
        if board is None:
            raise ValueError(f"Cannot read board crop: {row['rectified_image_path']}")
        if self.augment:
            board = augment_board(board, self.rng)
        crops = split_board_squares(board)
        images = torch.from_numpy(preprocess_square_crops(crops))
        labels = torch.tensor(json.loads(row["final_labels_json"]), dtype=torch.long)
        rejected = torch.tensor(json.loads(row["predicted_labels_json"]), dtype=torch.long)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--preference-weight",
        type=float,
        default=0.0,
        help="Experimental chosen-vs-rejected pairwise loss weight; production default is SFT only",
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
    rows = database.training_examples()
    if len(rows) < args.min_boards:
        raise SystemExit(
            f"Need at least {args.min_boards} consented confirmed boards; found {len(rows)}"
        )
    ensure_class_coverage(rows)
    train_rows, selection_rows, gate_rows = split_rows(
        rows,
        selection_fraction=args.selection_fraction,
        gate_fraction=args.gate_fraction,
    )
    if len(train_rows) < 20 or len(selection_rows) < 10 or len(gate_rows) < 10:
        raise SystemExit(
            "Grouped split needs at least 20 training, 10 selection, and 10 gate boards; "
            f"found {len(train_rows)}, {len(selection_rows)}, and {len(gate_rows)}. "
            "Collect feedback across more capture sessions; groups are never split to fill buckets."
        )

    active = database.get_active_model()
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

    model = load_fused_onnx(Path(active["artifact_path"])).to(device)
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
            supervised_loss = functional.cross_entropy(
                logits.reshape(-1, NUM_CLASSES),
                labels.reshape(-1),
                weight=class_weights,
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

    active_classifier = DiagramClassifier(
        Path(active["artifact_path"]), version=str(active["version"])
    )
    gate_loader = DataLoader(
        FeedbackBoardDataset(gate_rows, augment=False, seed=args.seed + 2),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    active_metrics = evaluate_onnx(active_classifier, gate_rows)
    candidate_metrics = evaluate_torch(model, gate_loader, device=device)
    eligible = passes_gates(candidate_metrics, active_metrics)

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
    metadata = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "architecture": "fused_tiny_square_cnn",
        "training_boards": len(train_rows),
        "selection_boards": len(selection_rows),
        "gate_boards": len(gate_rows),
        "seed": args.seed,
        "preference_weight": args.preference_weight,
        "active_baseline": active["version"],
        "active_metrics": active_metrics,
        "candidate_metrics": candidate_metrics,
        "eligible_for_promotion": eligible,
        "promotion_gates": "board_exact improves; square/non-empty/macro-F1 do not regress",
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


def evaluate_onnx(
    classifier: DiagramClassifier,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    for row in rows:
        image = cv2.imread(str(row["rectified_image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read board crop: {row['rectified_image_path']}")
        predictions.append(classifier.predict(image).labels)
        targets.append(json.loads(row["final_labels_json"]))
    return classification_metrics(predictions, targets)


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if selection_fraction <= 0 or gate_fraction <= 0:
        raise ValueError("Selection and gate fractions must be positive")
    if selection_fraction + gate_fraction >= 1:
        raise ValueError("Selection and gate fractions must leave room for training data")

    train: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    gate: list[dict[str, Any]] = []
    gate_threshold = int(gate_fraction * 10_000)
    selection_threshold = gate_threshold + int(selection_fraction * 10_000)
    for row in rows:
        group = str(row["client_session_id"] or row["feedback_id"])
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10_000
        if bucket < gate_threshold:
            gate.append(row)
        elif bucket < selection_threshold:
            selection.append(row)
        else:
            train.append(row)

    return train, selection, gate


def ensure_class_coverage(rows: list[dict[str, Any]]) -> None:
    counts: Counter[int] = Counter()
    for row in rows:
        counts.update(json.loads(row["final_labels_json"]))
    missing = [class_id for class_id in range(NUM_CLASSES) if counts[class_id] == 0]
    if missing:
        raise SystemExit(f"Training feedback is missing square classes: {missing}")


def class_weights_for_rows(rows: list[dict[str, Any]]) -> torch.Tensor:
    counts: Counter[int] = Counter()
    for row in rows:
        counts.update(json.loads(row["final_labels_json"]))
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


def passes_gates(candidate: dict[str, float], active: dict[str, float]) -> bool:
    return (
        candidate["board_exact"] > active["board_exact"]
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


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
