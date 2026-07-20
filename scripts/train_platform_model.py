#!/usr/bin/env python3
"""Adapt the square classifier to platform renderings with domain-balanced replay."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from chess_scan.argus_data import default_data_dir, sha256_file, verify_data_manifest
from chess_scan.classifier import preprocess_board, preprocess_square_crops
from chess_scan.model_artifact import model_version
from chess_scan.platform_data import default_data_dir as default_platform_data_dir
from chess_scan.platform_data import read_records
from chess_scan.platform_data import verify_data_manifest as verify_platform_data_manifest
from image_augmentation import jpeg_round_trip, resize_round_trip
from square_model import (
    WideSquareClassifier,
    export_onnx,
    load_fused_onnx,
    verify_model_matches_onnx,
)
from train_argus_recovery import prepare_official_retention_data
from training_utils import (
    REPLAY_WORKERS,
    ArgusReplayDataset,
    collate_replay_batch,
    distillation_loss,
    repeat_loader,
    resolve_device,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "chess-steps-v3.onnx"
DEFAULT_OFFICIAL_DIR = Path.home() / "chess-scan-training" / "chess-steps-official-v1"


class BoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        records: list[dict[str, Any]],
        *,
        seed: int,
        augment: bool,
    ) -> None:
        self.root = root
        self.records = records
        self.rng = random.Random(seed)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        board = cv2.imread(str(self.root / record["path"]), cv2.IMREAD_COLOR)
        if board is None:
            raise ValueError(f"Cannot read training board: {record['path']}")
        if self.augment:
            board = augment_screen_board(board, self.rng)
        images = torch.from_numpy(preprocess_board(board).copy())
        labels = torch.tensor(record["labels"], dtype=torch.long)
        return images, labels


class SquareDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path, records: list[dict[str, Any]], *, seed: int) -> None:
        self.root = root
        self.records = records
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        crop = cv2.imread(str(self.root / record["path"]), cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError(f"Cannot read training square: {record['path']}")
        if self.rng.random() < 0.6:
            crop = photometric_augmentation(crop, self.rng)
        image = torch.from_numpy(preprocess_square_crops([crop])[0].copy())
        return image, torch.tensor(record["label"], dtype=torch.long)


class OfficialBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], *, seed: int) -> None:
        self.rows = rows
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        board = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        if board is None:
            raise ValueError(f"Cannot read official board: {row['path']}")
        if self.rng.random() < 0.7:
            board = augment_retention_board(board, self.rng)
        return (
            torch.from_numpy(preprocess_board(board).copy()),
            torch.tensor(row["labels"], dtype=torch.long),
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    argus_dir = args.argus_dir.expanduser().resolve()
    platform_dir = args.platform_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    verify_data_manifest(argus_dir)
    platform_manifest = verify_platform_data_manifest(platform_dir)
    platform_records = read_records(platform_dir / "records.jsonl", split="train")
    real_records = read_records(platform_dir / "real" / "records.jsonl", split="train")
    square_records = read_records(platform_dir / "real" / "squares.jsonl", split="train")
    official_rows = prepare_official_retention_data(args.official_dir.expanduser().resolve())
    device = resolve_device(args.device)

    platform_loader = board_loader(
        BoardDataset(platform_dir, platform_records, seed=args.seed, augment=True),
        batch_size=args.board_batch_size,
        seed=args.seed,
    )
    official_loader = board_loader(
        OfficialBoardDataset(official_rows, seed=args.seed + 1),
        batch_size=args.board_batch_size,
        seed=args.seed + 1,
    )
    feedback_loader = board_loader(
        BoardDataset(platform_dir, real_records, seed=args.seed + 2, augment=True),
        batch_size=min(4, len(real_records)),
        seed=args.seed + 2,
    )
    square_loader = board_loader(
        SquareDataset(platform_dir, square_records, seed=args.seed + 3),
        batch_size=len(square_records),
        seed=args.seed + 3,
    )
    replay_loader = prepare_replay_loader(
        argus_dir,
        batch_size=args.square_batch_size,
        seed=args.seed,
        source=args.replay_source,
    )

    teacher_path = args.teacher_model or args.base_model
    teacher = load_fused_onnx(teacher_path).eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    if args.architecture == "wide":
        model = WideSquareClassifier()
        if args.base_checkpoint:
            checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
    else:
        model = load_fused_onnx(args.base_model).to(device)
    configure_trainable_layers(model, blocks=args.trainable_blocks)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": args.classifier_lr},
            {
                "params": [
                    parameter
                    for parameter in model.features.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.feature_lr,
            },
        ],
        weight_decay=1e-5,
    )
    class_weights = torch.tensor(
        [0.35, 1.0, 1.25, 1.25, 1.35, 1.6, 1.6, 1.0, 1.25, 1.25, 1.35, 1.6, 1.6],
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(
            model,
            teacher=teacher,
            platform_loader=platform_loader,
            replay_loader=replay_loader,
            official_loader=official_loader,
            feedback_loader=feedback_loader,
            square_loader=square_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            platform_weight=args.platform_weight,
            replay_weight=args.replay_weight,
            official_weight=args.official_weight,
            feedback_weight=args.feedback_weight,
            retention_weight=args.retention_weight,
            device=device,
        )
        checkpoint = output_dir / f"{args.version}-epoch-{epoch + 1}.onnx"
        export_onnx(model, checkpoint)
        model.to(device)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={loss:.6f} artifact={checkpoint}")

    artifact_path = output_dir / f"{args.version}.onnx"
    checkpoint_path = output_dir / f"{args.version}.pt"
    metadata_path = output_dir / f"{args.version}.json"
    export_onnx(model, artifact_path)
    onnx_max_logit_difference = verify_model_matches_onnx(model, artifact_path)
    model.cpu().eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "version": args.version,
            "architecture": architecture_name(args.architecture),
            "base_model": model_version(args.base_model),
            "input_size": 64,
            "num_classes": 13,
        },
        checkpoint_path,
    )
    metadata = {
        "version": args.version,
        "trained_at": datetime.now(UTC).isoformat(),
        "runtime_format": "onnx",
        "architecture": architecture_name(args.architecture),
        "base_model": model_version(args.base_model),
        "artifact_sha256": sha256_file(artifact_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_max_logit_difference": onnx_max_logit_difference,
        "platform_dataset_version": platform_manifest["version"],
        "platform_training_boards": len(platform_records),
        "real_feedback_boards": len(real_records),
        "eligible_for_promotion": False,
        "requires_fixed_qa": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--teacher-model", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--architecture", choices=("tiny", "wide"), default="tiny")
    parser.add_argument("--argus-dir", type=Path, default=default_data_dir())
    parser.add_argument("--platform-dir", type=Path, default=default_platform_data_dir())
    parser.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "chess-scan-training" / "platform-model-candidates",
    )
    parser.add_argument("--version", default="chess-steps-v4-candidate")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--board-batch-size", type=int, default=8)
    parser.add_argument("--square-batch-size", type=int, default=512)
    parser.add_argument("--replay-source", choices=("all", "argus", "synthetic"), default="all")
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--feature-lr", type=float, default=1e-5)
    parser.add_argument("--trainable-blocks", type=int, default=8)
    parser.add_argument("--platform-weight", type=float, default=1.0)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--official-weight", type=float, default=1.0)
    parser.add_argument("--feedback-weight", type=float, default=2.0)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def board_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )


def prepare_replay_loader(data_dir: Path, *, batch_size: int, seed: int, source: str) -> DataLoader:
    return DataLoader(
        ArgusReplayDataset(data_dir, source=source),
        batch_size=batch_size,
        shuffle=True,
        num_workers=REPLAY_WORKERS,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_replay_batch,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(
    model: torch.nn.Module,
    *,
    teacher: torch.nn.Module,
    platform_loader: DataLoader,
    replay_loader: DataLoader,
    official_loader: DataLoader,
    feedback_loader: DataLoader,
    square_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    platform_weight: float,
    replay_weight: float,
    official_weight: float,
    feedback_weight: float,
    retention_weight: float,
    device: torch.device,
) -> float:
    model.train()
    replay = repeat_loader(replay_loader)
    official = repeat_loader(official_loader)
    feedback = repeat_loader(feedback_loader)
    squares = repeat_loader(square_loader)
    total_loss = 0.0
    for platform_images, platform_labels in platform_loader:
        replay_images, replay_labels = next(replay)
        official_images, official_labels = next(official)
        feedback_images, feedback_labels = next(feedback)
        square_images, square_labels = next(squares)
        platform_images, platform_labels = flatten_boards(platform_images, platform_labels, device)
        official_images, official_labels = flatten_boards(official_images, official_labels, device)
        feedback_images, feedback_labels = flatten_boards(feedback_images, feedback_labels, device)
        replay_images = replay_images.to(device)
        replay_labels = replay_labels.to(device)
        square_images = square_images.to(device)
        square_labels = square_labels.to(device)

        optimizer.zero_grad()
        step_loss = backward_weighted(
            weighted_loss(model(platform_images), platform_labels, class_weights),
            platform_weight,
        )

        replay_logits = model(replay_images)
        with torch.no_grad():
            replay_teacher_logits = teacher(replay_images)
        replay_loss = replay_weight * weighted_loss(
            replay_logits,
            replay_labels,
            class_weights,
        ) + retention_weight * distillation_loss(replay_logits, replay_teacher_logits)
        step_loss = step_loss + backward_weighted(replay_loss, 1.0)

        official_logits = model(official_images)
        with torch.no_grad():
            official_teacher_logits = teacher(official_images)
        official_loss = official_weight * weighted_loss(
            official_logits,
            official_labels,
            class_weights,
        ) + retention_weight * distillation_loss(official_logits, official_teacher_logits)
        step_loss = step_loss + backward_weighted(official_loss, 1.0)

        step_loss = step_loss + backward_weighted(
            weighted_loss(model(feedback_images), feedback_labels, class_weights),
            feedback_weight,
        )
        step_loss = step_loss + backward_weighted(
            weighted_loss(model(square_images), square_labels, class_weights),
            feedback_weight,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += float(step_loss)
    return total_loss / len(platform_loader)


def weighted_loss(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    return functional.cross_entropy(logits, labels, weight=class_weights)


def backward_weighted(loss: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0:
        return loss.detach().new_zeros(())
    weighted = weight * loss
    weighted.backward()
    return weighted.detach()


def flatten_boards(
    images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return images.flatten(0, 1).to(device), labels.flatten().to(device)


def architecture_name(architecture: str) -> str:
    return "wide_square_cnn" if architecture == "wide" else "fused_tiny_square_cnn"


def configure_trainable_layers(model: torch.nn.Module, *, blocks: int) -> None:
    features = list(model.features)
    if blocks < 1 or blocks > len(features):
        raise ValueError(f"trainable-blocks must be between 1 and {len(features)}")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in features[-blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def augment_screen_board(board: np.ndarray, rng: random.Random) -> np.ndarray:
    board = cv2.resize(board, (512, 512), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.25:
        return board
    board = photometric_augmentation(board, rng)
    variant = rng.randrange(5)
    if variant == 1:
        board = resize_round_trip(board, rng.randrange(180, 430))
    elif variant == 2:
        board = jpeg_round_trip(board, rng.randrange(35, 90))
    elif variant == 3:
        board = cv2.GaussianBlur(board, (3, 3), rng.uniform(0.2, 1.2))
    elif variant == 4:
        board = camera_round_trip(board, rng)
    return add_display_artifacts(board, rng)


def augment_retention_board(board: np.ndarray, rng: random.Random) -> np.ndarray:
    board = cv2.resize(board, (512, 512), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.4:
        board = resize_round_trip(board, rng.randrange(220, 430))
    if rng.random() < 0.3:
        board = cv2.GaussianBlur(board, (3, 3), rng.uniform(0.1, 0.7))
    return board


def photometric_augmentation(board: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-12, 12)) % 180
    hsv[:, :, 1] *= rng.uniform(0.25, 1.5)
    hsv[:, :, 2] *= rng.uniform(0.65, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gamma = rng.uniform(0.7, 1.4)
    table = np.asarray([(index / 255) ** gamma * 255 for index in range(256)], dtype=np.uint8)
    return cv2.LUT(adjusted, table)


def camera_round_trip(board: np.ndarray, rng: random.Random) -> np.ndarray:
    canvas_size = 640
    margin = 54
    source = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
    corners = np.float32(
        [
            [margin + rng.uniform(-20, 20), margin + rng.uniform(-20, 20)],
            [canvas_size - margin + rng.uniform(-20, 20), margin + rng.uniform(-20, 20)],
            [
                canvas_size - margin + rng.uniform(-20, 20),
                canvas_size - margin + rng.uniform(-20, 20),
            ],
            [margin + rng.uniform(-20, 20), canvas_size - margin + rng.uniform(-20, 20)],
        ]
    )
    canvas = cv2.warpPerspective(
        board,
        cv2.getPerspectiveTransform(source, corners),
        (canvas_size, canvas_size),
        borderValue=(rng.randrange(15, 80),) * 3,
    )
    return cv2.warpPerspective(
        canvas,
        cv2.getPerspectiveTransform(corners, source),
        (512, 512),
    )


def add_display_artifacts(board: np.ndarray, rng: random.Random) -> np.ndarray:
    output = board.astype(np.float32)
    if rng.random() < 0.45:
        x = np.linspace(0, np.pi * rng.uniform(5, 18), output.shape[1])
        modulation = np.sin(x + rng.uniform(0, np.pi * 2)) * rng.uniform(1, 6)
        output += modulation[None, :, None]
    if rng.random() < 0.35:
        gradient = np.linspace(rng.uniform(-18, 0), rng.uniform(0, 18), output.shape[0])
        output += gradient[:, None, None]
    return np.clip(output, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
