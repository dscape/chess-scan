#!/usr/bin/env python3
"""Recover Argus digital-board accuracy without regressing Chess Steps print behavior."""

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
from torch.utils.data import DataLoader, Dataset, TensorDataset

from chess_scan.argus_data import (
    default_data_dir,
    evaluate_argus_pair,
    labels_from_board_filename,
    load_prepared_split,
    load_synthetic_replay,
    sha256_file,
    validate_source_data,
    verify_data_manifest,
)
from chess_scan.classifier import preprocess_board
from evaluate_photo_stress import make_halftone_screen, rectify_printed_photo, resize_round_trip
from square_model import export_onnx, load_fused_onnx
from train_chess_steps_model import (
    download_official_samples,
    has_exactly_two_kings,
    load_or_label_official_boards,
    render_official_boards,
    verify_benchmark_pdf,
)
from training_utils import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "chess-steps-v2.onnx"
REFERENCE_BENCHMARK = PROJECT_ROOT / "benchmarks" / "chess-steps-step2.json"
KING_BENCHMARK = PROJECT_ROOT / "benchmarks" / "chess-steps-kings.json"
EXCLUDED_OFFICIAL_SOURCES = {"en_lp_2", "en_lp_2e", "en_lp_3"}
FULL_CLASS_COUNTS = np.asarray(
    [
        4_319_574,
        71_992,
        71_094,
        70_363,
        70_981,
        35_528,
        80_000,
        72_061,
        71_088,
        70_857,
        70_924,
        35_538,
        80_000,
    ],
    dtype=np.float64,
)


class RetentionBoardDataset(Dataset[torch.Tensor]):
    def __init__(self, rows: list[dict[str, Any]], *, seed: int) -> None:
        self.rows = rows
        self.rng = random.Random(seed)
        self.halftone_screen = make_halftone_screen(1024)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        board = cv2.imread(str(self.rows[index]["path"]), cv2.IMREAD_COLOR)
        if board is None:
            raise ValueError(f"Cannot read official retention board: {self.rows[index]['path']}")
        variant = self.rng.randrange(6)
        if variant == 1:
            board = resize_round_trip(board, 256)
        elif variant == 2:
            board = resize_round_trip(board, 256, contrast=self.rng.uniform(0.25, 0.65))
        elif variant == 3:
            board = rectify_printed_photo(board, self.halftone_screen)
        elif variant == 4:
            size = self.rng.randrange(180, 350)
            board = cv2.resize(board, (size, size), interpolation=cv2.INTER_AREA)
            board = cv2.resize(board, (512, 512), interpolation=cv2.INTER_CUBIC)
        elif variant == 5:
            board = cv2.GaussianBlur(board, (3, 3), self.rng.uniform(0.2, 0.8))
        return torch.from_numpy(preprocess_board(board))


class ChessPositionsBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: Path) -> None:
        self.paths = sorted((data_dir / "data" / "chess_positions" / "train").glob("*.jpeg"))
        if len(self.paths) != 80_000:
            raise ValueError(f"Expected 80,000 Argus training boards, found {len(self.paths)}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        board = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if board is None or board.shape[:2] != (400, 400):
            raise ValueError(f"Invalid chess-positions board: {path}")
        squares = []
        for row in range(8):
            for column in range(8):
                square = board[row * 50 : (row + 1) * 50, column * 50 : (column + 1) * 50]
                square = cv2.resize(square, (64, 64), interpolation=cv2.INTER_LINEAR)
                squares.append(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        images = np.stack(squares).transpose(0, 3, 1, 2).copy()
        labels = np.asarray(labels_from_board_filename(path.name), dtype=np.int64)
        return torch.from_numpy(images).float().div_(255.0), torch.from_numpy(labels)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = args.data_dir.expanduser().resolve()
    official_dir = args.official_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_source_data(data_dir)
    verify_data_manifest(data_dir)
    retention_rows = prepare_official_retention_data(official_dir)
    device = resolve_device(args.device)
    teacher = load_fused_onnx(args.base_model).eval().to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_fused_onnx(args.base_model).to(device)
    train_compact_stage(
        model,
        teacher=teacher,
        data_dir=data_dir,
        retention_rows=retention_rows,
        epochs=12,
        seed=args.seed,
        device=device,
    )
    stage_one = output_dir / f"{args.version}-stage-one.onnx"
    export_onnx(model, stage_one)

    model = load_fused_onnx(stage_one).to(device)
    train_compact_stage(
        model,
        teacher=teacher,
        data_dir=data_dir,
        retention_rows=retention_rows,
        epochs=8,
        seed=args.seed,
        device=device,
    )
    stage_two = output_dir / f"{args.version}-stage-two.onnx"
    export_onnx(model, stage_two)

    model = load_fused_onnx(stage_two).to(device)
    train_full_corpus_stage(
        model,
        teacher=teacher,
        data_dir=data_dir,
        retention_rows=retention_rows,
        device=device,
    )

    artifact_path = output_dir / f"{args.version}.onnx"
    checkpoint_path = output_dir / f"{args.version}.pt"
    metadata_path = output_dir / f"{args.version}.json"
    export_onnx(model, artifact_path)
    model.cpu().eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "version": args.version,
            "architecture": "fused_tiny_square_cnn",
            "base_model": args.base_model.stem,
            "input_size": 64,
            "num_classes": 13,
        },
        checkpoint_path,
    )
    argus_gate = evaluate_argus_pair(args.base_model, artifact_path, data_dir)
    metadata = {
        "version": args.version,
        "trained_at": datetime.now(UTC).isoformat(),
        "runtime_format": "onnx",
        "architecture": "fused_tiny_square_cnn",
        "base_model": args.base_model.stem,
        "artifact_sha256": sha256_file(artifact_path),
        "eligible_for_promotion": False,
        "requires_fixed_qa": True,
        "external_dataset_manifest": str(data_dir / "MANIFEST.json"),
        "argus_gate": argus_gate,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Candidate written to {artifact_path}; run all fixed QA gates before promotion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path.home() / "chess-scan-training" / "chess-steps-official-v1",
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=default_data_dir() / "candidates")
    parser.add_argument("--version", default="chess-steps-v3-reproduction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def prepare_official_retention_data(work_dir: Path) -> list[dict[str, Any]]:
    pdf_dir = work_dir / "official-pdfs"
    board_dir = work_dir / "official-boards"
    reference = json.loads(REFERENCE_BENCHMARK.read_text())
    king_gate = json.loads(KING_BENCHMARK.read_text())
    download_official_samples(pdf_dir)
    verify_benchmark_pdf(pdf_dir / "en_lp_2.pdf", reference)
    verify_benchmark_pdf(pdf_dir / "en_lp_3.pdf", king_gate)
    render_official_boards(pdf_dir, board_dir)
    rows = load_or_label_official_boards(board_dir, reference, work_dir)
    selected = [
        row
        for row in rows
        if row["source"] not in EXCLUDED_OFFICIAL_SOURCES and has_exactly_two_kings(row)
    ]
    if len(selected) != 934:
        raise ValueError(f"Expected 934 official retention boards, found {len(selected)}")
    return selected


def train_compact_stage(
    model: torch.nn.Module,
    *,
    teacher: torch.nn.Module,
    data_dir: Path,
    retention_rows: list[dict[str, Any]],
    epochs: int,
    seed: int,
    device: torch.device,
) -> None:
    train_split = load_prepared_split(data_dir, "train", mmap=False)
    replay_images, replay_labels = load_synthetic_replay(data_dir)
    images = np.concatenate([train_split.images, replay_images])
    labels = np.concatenate([train_split.labels, replay_labels])
    loader = DataLoader(
        TensorDataset(_image_tensor(images), torch.from_numpy(labels)),
        batch_size=256,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    retention_loader = _retention_loader(retention_rows, seed=seed + 2)
    _configure_trainable_layers(model)
    optimizer = _optimizer(model, classifier_lr=2e-4, feature_lr=2e-5)

    for epoch in range(epochs):
        loss = _train_epoch(
            model,
            teacher=teacher,
            training_loader=loader,
            retention_loader=retention_loader,
            optimizer=optimizer,
            device=device,
        )
        print(f"compact epoch {epoch + 1}/{epochs}: loss={loss:.6f}")


def train_full_corpus_stage(
    model: torch.nn.Module,
    *,
    teacher: torch.nn.Module,
    data_dir: Path,
    retention_rows: list[dict[str, Any]],
    device: torch.device,
) -> None:
    loader = DataLoader(
        ChessPositionsBoardDataset(data_dir),
        batch_size=24,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(47),
    )
    retention_loader = _retention_loader(retention_rows, seed=48)
    class_targets = torch.tensor([6.0] + [1.0] * 12, device=device)
    class_weights = class_targets / torch.from_numpy(FULL_CLASS_COUNTS).to(device)
    class_weights /= class_weights.mean()
    _configure_trainable_layers(model)
    optimizer = _optimizer(model, classifier_lr=7.5e-6, feature_lr=7.5e-7)
    retention = _repeat_loader(retention_loader)
    model.train()
    total_loss = 0.0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.reshape(-1, 3, 64, 64).to(device)
        labels = labels.reshape(-1).to(device)
        retained = next(retention).reshape(-1, 3, 64, 64).to(device)
        logits = model(images)
        supervised = functional.cross_entropy(logits, labels, weight=class_weights)
        loss = supervised + 2 * retention_loss(model, teacher, retained)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        if step % 500 == 0 or step == len(loader):
            print(f"full corpus {step}/{len(loader)}: loss={total_loss / step:.6f}")


def retention_loss(
    model: torch.nn.Module, teacher: torch.nn.Module, images: torch.Tensor
) -> torch.Tensor:
    student_logits = model(images)
    with torch.no_grad():
        teacher_logits = teacher(images)
        hard_targets = teacher_logits.argmax(dim=1)
    hard_loss = functional.cross_entropy(student_logits, hard_targets)
    distillation = (
        functional.kl_div(
            functional.log_softmax(student_logits / 2, dim=1),
            functional.softmax(teacher_logits / 2, dim=1),
            reduction="batchmean",
        )
        * 4
    )
    return hard_loss + distillation


def _train_epoch(
    model: torch.nn.Module,
    *,
    teacher: torch.nn.Module,
    training_loader: DataLoader,
    retention_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    retention = _repeat_loader(retention_loader)
    total_loss = 0.0
    for images, labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)
        retained = next(retention).reshape(-1, 3, 64, 64).to(device)
        supervised = functional.cross_entropy(model(images), labels)
        loss = supervised + 2 * retention_loss(model, teacher, retained)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / len(training_loader)


def _repeat_loader(loader: DataLoader):
    while True:
        yield from loader


def _retention_loader(rows: list[dict[str, Any]], *, seed: int) -> DataLoader:
    return DataLoader(
        RetentionBoardDataset(rows, seed=seed),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _configure_trainable_layers(model: torch.nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in list(model.features)[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def _optimizer(
    model: torch.nn.Module, *, classifier_lr: float, feature_lr: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": classifier_lr},
            {
                "params": [
                    parameter
                    for parameter in model.features.parameters()
                    if parameter.requires_grad
                ],
                "lr": feature_lr,
            },
        ],
        weight_decay=1e-5,
    )


def _image_tensor(images_bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(images_bgr[:, :, :, ::-1])
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)


if __name__ == "__main__":
    main()
