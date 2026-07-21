#!/usr/bin/env python3
"""Recover photographed Chess Steps print accuracy with domain-balanced replay."""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, get_worker_info

from chess_scan.argus_data import default_data_dir as default_argus_data_dir
from chess_scan.argus_data import verify_data_manifest as verify_argus_data_manifest
from chess_scan.classifier import DiagramClassifier, preprocess_board
from chess_scan.model_artifact import model_version, sha256_file
from chess_scan.platform_data import default_data_dir as default_platform_data_dir
from chess_scan.platform_data import load_records as load_platform_records
from chess_scan.platform_data import verify_data_manifest as verify_platform_data_manifest
from chess_scan.print_data import default_data_dir as default_print_data_dir
from chess_scan.print_data import load_records as load_print_records
from chess_scan.print_data import verify_data_manifest as verify_print_data_manifest
from evaluate_photo_stress import make_halftone_screen, rectify_printed_photo
from evaluate_platforms import transform_board
from image_augmentation import contrast_brightness, jpeg_round_trip, resize_round_trip
from square_model import export_onnx, load_fused_wide_onnx, verify_model_matches_onnx
from train_argus_recovery import prepare_official_retention_data
from training_utils import (
    REPLAY_WORKERS,
    ArgusReplayDataset,
    collate_replay_batch,
    distillation_loss,
    resolve_device,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "chess-steps-v4.onnx"
DEFAULT_OFFICIAL_DIR = Path.home() / "chess-scan-training" / "chess-steps-official-v1"

RecoveryBatch = tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


class PlatformBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        records: list[dict[str, Any]],
        *,
        seed: int,
    ) -> None:
        self.root = root
        self.records = records
        self.seed = seed
        self._rng: random.Random | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        board = _read_board(self.root / record["path"])
        rng = self._random()
        if rng.randrange(3) > 0:
            board = transform_board(board, "camera", rng.randrange(1_000_000))
        return (
            torch.from_numpy(preprocess_board(board).copy()),
            torch.tensor(record["labels"], dtype=torch.long),
        )

    def _random(self) -> random.Random:
        if self._rng is None:
            self._rng = _worker_random(self.seed)
        return self._rng


class RecoveryBatchCollator:
    def __init__(
        self,
        *,
        print_examples: list[dict[str, Any]],
        official_rows: list[dict[str, Any]],
        argus: ArgusReplayDataset,
        seed: int,
    ) -> None:
        self.print_examples = print_examples
        self.official_rows = official_rows
        self.argus = argus
        self.seed = seed
        self.halftone_screen = make_halftone_screen(1024)
        self._rng: random.Random | None = None

    def __call__(
        self,
        platform_examples: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> RecoveryBatch:
        platform_inputs, platform_labels = zip(*platform_examples, strict=True)
        return (
            self._target_batch(2),
            (torch.cat(platform_inputs), torch.cat(platform_labels)),
            self._official_batch(4),
            self._argus_batch(256),
        )

    def _target_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = []
        labels = []
        weights = []
        rng = self._random()
        for _ in range(count):
            example = rng.choice(self.print_examples)
            inputs.append(preprocess_board(_augment_print(example["board"], rng)))
            labels.append(example["labels"])
            weights.append(example["weights"])
        return (
            torch.from_numpy(np.concatenate(inputs).copy()),
            torch.from_numpy(np.concatenate(labels)).long(),
            torch.from_numpy(np.concatenate(weights)).float(),
        )

    def _official_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = []
        labels = []
        rng = self._random()
        for record in rng.sample(self.official_rows, count):
            board = _augment_official(
                _read_board(Path(record["path"])),
                rng,
                self.halftone_screen,
            )
            inputs.append(preprocess_board(board))
            labels.extend(record["labels"])
        return (
            torch.from_numpy(np.concatenate(inputs).copy()),
            torch.tensor(labels, dtype=torch.long),
        )

    def _argus_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._random()
        examples = [self.argus[rng.randrange(len(self.argus))] for _ in range(count)]
        return collate_replay_batch(examples)

    def _random(self) -> random.Random:
        if self._rng is None:
            self._rng = _worker_random(self.seed)
        return self._rng


class PrintRecoveryTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.print_dir = args.print_data_dir.expanduser().resolve()
        self.platform_dir = args.platform_data_dir.expanduser().resolve()
        self.argus_dir = args.argus_data_dir.expanduser().resolve()
        self.output_dir = args.output_dir.expanduser().resolve()

        print_manifest = verify_print_data_manifest(self.print_dir)
        platform_manifest = verify_platform_data_manifest(self.platform_dir)
        verify_argus_data_manifest(self.argus_dir)
        self.print_manifest = print_manifest
        self.platform_manifest = platform_manifest
        self.print_examples = self._load_print_examples()
        self.platform_records = load_platform_records(self.platform_dir, split="train")
        self.official_rows = prepare_official_retention_data(
            args.official_dir.expanduser().resolve()
        )
        if not self.platform_records:
            raise ValueError("The print recovery trainer requires platform records")
        self.training_loader = DataLoader(
            PlatformBoardDataset(
                self.platform_dir,
                self.platform_records,
                seed=args.seed,
            ),
            batch_size=args.platform_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=REPLAY_WORKERS,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=RecoveryBatchCollator(
                print_examples=self.print_examples,
                official_rows=self.official_rows,
                argus=ArgusReplayDataset(self.argus_dir, source="all"),
                seed=args.seed + 1,
            ),
            generator=torch.Generator().manual_seed(args.seed),
        )

        self.model = load_fused_wide_onnx(args.base_model)
        self.teacher = copy.deepcopy(self.model).eval().to(self.device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        for parameter in self.model.features.parameters():
            parameter.requires_grad = True
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.classifier.parameters(), "lr": args.classifier_lr},
                {"params": self.model.features.parameters(), "lr": args.feature_lr},
            ],
            weight_decay=1e-5,
        )

    def train(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(UTC)
        for epoch in range(1, self.args.epochs + 1):
            loss = self._train_epoch()
            epoch_path = self.output_dir / f"{self.args.version}-epoch-{epoch}.onnx"
            export_onnx(self.model, epoch_path)
            self.model.to(self.device)
            exact = self._print_exact_boards(epoch_path)
            print(
                f"epoch {epoch}/{self.args.epochs}: loss={loss:.6f} "
                f"print_exact={exact}/{len(self.print_examples)} artifact={epoch_path}"
            )

        artifact_path = self.output_dir / f"{self.args.version}.onnx"
        checkpoint_path = self.output_dir / f"{self.args.version}.pt"
        metadata_path = self.output_dir / f"{self.args.version}.json"
        export_onnx(self.model, artifact_path)
        exact = self._print_exact_boards(artifact_path)
        if exact != len(self.print_examples):
            raise RuntimeError(
                f"Final candidate retained only {exact}/{len(self.print_examples)} "
                "photographed-print boards"
            )
        maximum_difference = verify_model_matches_onnx(self.model, artifact_path)
        state_dict = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state_dict,
                "version": self.args.version,
                "architecture": "fused_wide_square_cnn",
                "base_model": model_version(self.args.base_model),
                "input_size": 64,
                "num_classes": 13,
            },
            checkpoint_path,
        )
        metadata = {
            "version": self.args.version,
            "trained_at": started_at.isoformat(),
            "runtime_format": "onnx",
            "architecture": "fused_wide_square_cnn",
            "base_model": model_version(self.args.base_model),
            "input_size": 64,
            "num_classes": 13,
            "artifact_sha256": sha256_file(artifact_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_max_logit_difference": maximum_difference,
            "eligible_for_promotion": False,
            "requires_fixed_qa": True,
            "adaptation": {
                "purpose": "Recover real photographed Chess Steps print accuracy without "
                "regressing platform, Argus, synthetic, or official-print coverage",
                "script": "scripts/train_print_recovery.py",
                "print_dataset_version": self.print_manifest["version"],
                "print_training_boards": len(self.print_examples),
                "print_training_groups": self.print_manifest["groups"],
                "platform_dataset_version": self.platform_manifest["version"],
                "platform_training_boards": len(self.platform_records),
                "official_retention_boards": len(self.official_rows),
                "source_images_redistributed": False,
            },
            "reproduction": {
                "device": str(self.device),
                "seed": self.args.seed,
                "epochs": self.args.epochs,
                "byte_identical_expected": False,
                "note": "MPS training is not byte-deterministic. Promote only an artifact "
                "that independently passes every recorded gate.",
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    def _train_epoch(self) -> float:
        total_loss = torch.zeros((), device=self.device)
        steps = 0
        for target, platform, official, argus in self.training_loader:
            self.model.train()
            target_inputs, target_labels, target_weights = (
                tensor.to(self.device) for tensor in target
            )
            target_logits = self.model(target_inputs)
            target_loss = (
                functional.cross_entropy(
                    target_logits,
                    target_labels,
                    reduction="none",
                )
                * target_weights
            ).sum() / target_weights.sum()

            platform_loss = self._retention_loss(*platform, distillation_weight=5.0)
            official_loss = self._retention_loss(*official, distillation_weight=2.0)
            argus_loss = self._retention_loss(*argus, distillation_weight=3.0)
            loss = 2.0 * target_loss + 4.0 * platform_loss + 0.5 * official_loss + 0.25 * argus_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            total_loss += loss.detach()
            steps += 1
        if steps == 0:
            raise RuntimeError("The print recovery trainer produced no platform batches")
        return (total_loss / steps).item()

    def _retention_loss(
        self,
        inputs: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        *,
        distillation_weight: float,
    ) -> torch.Tensor:
        images = torch.as_tensor(inputs).to(self.device)
        targets = torch.as_tensor(labels, dtype=torch.long).to(self.device)
        logits = self.model(images)
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        return functional.cross_entropy(logits, targets) + distillation_weight * distillation_loss(
            logits,
            teacher_logits,
        )

    def _load_print_examples(self) -> list[dict[str, Any]]:
        classifier = DiagramClassifier(
            self.args.base_model,
            version=model_version(self.args.base_model),
        )
        examples = []
        for record in load_print_records(self.print_dir):
            board = _read_board(self.print_dir / record["path"])
            labels = np.asarray(record["labels"], dtype=np.int64)
            prediction = np.asarray(classifier.predict(board).labels, dtype=np.int64)
            weights = np.where(labels == 0, 0.15, 1.0).astype(np.float32)
            weights[prediction != labels] = 32.0
            examples.append({"board": board, "labels": labels, "weights": weights})
        return examples

    def _print_exact_boards(self, model_path: Path) -> int:
        classifier = DiagramClassifier(model_path, version=model_version(model_path))
        return sum(
            classifier.predict(example["board"]).labels == example["labels"].tolist()
            for example in self.print_examples
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--print-data-dir", type=Path, default=default_print_data_dir())
    parser.add_argument("--platform-data-dir", type=Path, default=default_platform_data_dir())
    parser.add_argument("--argus-data-dir", type=Path, default=default_argus_data_dir())
    parser.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "chess-scan-training" / "print-model-candidates",
    )
    parser.add_argument("--version", default="chess-steps-v5-candidate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--platform-batch-size", type=int, default=8)
    parser.add_argument("--classifier-lr", type=float, default=1e-5)
    parser.add_argument("--feature-lr", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.platform_batch_size <= 0:
        parser.error("epochs and platform batch size must be positive")
    return args


def _augment_print(board: np.ndarray, rng: random.Random) -> np.ndarray:
    variant = rng.randrange(8)
    if variant == 1:
        return cv2.GaussianBlur(board, (0, 0), rng.uniform(0.2, 1.3))
    if variant == 2:
        return resize_round_trip(board, rng.randrange(128, 430))
    if variant == 3:
        return jpeg_round_trip(board, rng.randrange(45, 96))
    if variant == 4:
        return contrast_brightness(
            board,
            contrast=rng.uniform(0.65, 1.35),
            brightness=rng.uniform(-20, 20),
        )
    if variant == 5:
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if variant == 6:
        compressed = jpeg_round_trip(board, rng.randrange(55, 90))
        return cv2.GaussianBlur(compressed, (0, 0), rng.uniform(0.2, 0.9))
    if variant == 7:
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.3, 1.3)
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] * rng.uniform(0.8, 1.15),
            0,
            255,
        )
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return board


def _augment_official(
    board: np.ndarray,
    rng: random.Random,
    halftone_screen: np.ndarray,
) -> np.ndarray:
    variant = rng.randrange(5)
    if variant == 1:
        return resize_round_trip(board, 256, contrast=rng.uniform(0.3, 0.7))
    if variant == 2:
        return rectify_printed_photo(board, halftone_screen)
    if variant == 3:
        return cv2.GaussianBlur(board, (0, 0), rng.uniform(0.2, 0.8))
    if variant == 4:
        return resize_round_trip(board, rng.randrange(160, 420))
    return board


def _worker_random(seed: int) -> random.Random:
    worker = get_worker_info()
    return random.Random(seed if worker is None else seed + worker.seed)


def _read_board(path: Path) -> np.ndarray:
    board = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if board is None:
        raise ValueError(f"Cannot read training board: {path}")
    return board


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata = PrintRecoveryTrainer(args).train()
    print(json.dumps(metadata, indent=2))
    print("Run every fixed print, platform, Argus, online, and photo gate before promotion.")


if __name__ == "__main__":
    main()
