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

from chess_scan.argus_data import default_data_dir as default_argus_data_dir
from chess_scan.argus_data import verify_data_manifest as verify_argus_data_manifest
from chess_scan.classifier import DiagramClassifier, preprocess_board, preprocess_square_crops
from chess_scan.model_artifact import model_version, sha256_file
from chess_scan.platform_data import default_data_dir as default_platform_data_dir
from chess_scan.platform_data import load_records as load_platform_records
from chess_scan.platform_data import verify_data_manifest as verify_platform_data_manifest
from chess_scan.print_data import default_data_dir as default_print_data_dir
from chess_scan.print_data import load_records as load_print_records
from chess_scan.print_data import verify_data_manifest as verify_print_data_manifest
from evaluate_photo_stress import make_halftone_screen, rectify_printed_photo
from evaluate_platforms import transform_board
from image_augmentation import jpeg_round_trip, resize_round_trip
from square_model import export_onnx, load_fused_wide_onnx, verify_model_matches_onnx
from train_argus_recovery import prepare_official_retention_data
from training_utils import ArgusReplayDataset, distillation_loss, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "chess-steps-v4.onnx"
DEFAULT_OFFICIAL_DIR = Path.home() / "chess-scan-training" / "chess-steps-official-v1"


class PrintRecoveryTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
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
        self.argus = ArgusReplayDataset(self.argus_dir, source="all")
        self.halftone_screen = make_halftone_screen(1024)

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
        order = list(self.platform_records)
        self.rng.shuffle(order)
        total_loss = 0.0
        steps = 0
        for offset in range(
            0, len(order) - self.args.platform_batch_size + 1, self.args.platform_batch_size
        ):
            self.model.train()
            target_inputs, target_labels, target_weights = self._target_batch(2)
            target_logits = self.model(target_inputs)
            target_loss = (
                functional.cross_entropy(
                    target_logits,
                    target_labels,
                    reduction="none",
                )
                * target_weights
            ).sum() / target_weights.sum()

            platform_loss = self._retention_loss(
                *self._platform_batch(order[offset : offset + self.args.platform_batch_size]),
                distillation_weight=5.0,
            )
            official_loss = self._retention_loss(
                *self._official_batch(4),
                distillation_weight=2.0,
            )
            argus_loss = self._retention_loss(
                *self._argus_batch(256),
                distillation_weight=3.0,
            )
            loss = 2.0 * target_loss + 4.0 * platform_loss + 0.5 * official_loss + 0.25 * argus_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        return total_loss / steps

    def _target_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = []
        labels = []
        weights = []
        for _ in range(count):
            example = self.rng.choice(self.print_examples)
            inputs.append(preprocess_board(self._augment_print(example["board"])))
            labels.append(example["labels"])
            weights.append(example["weights"])
        return (
            torch.from_numpy(np.concatenate(inputs).copy()).to(self.device),
            torch.from_numpy(np.concatenate(labels)).long().to(self.device),
            torch.from_numpy(np.concatenate(weights)).float().to(self.device),
        )

    def _platform_batch(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs = []
        labels = []
        for record in records:
            board = _read_board(self.platform_dir / record["path"])
            if self.rng.randrange(3) > 0:
                board = transform_board(board, "camera", self.rng.randrange(1_000_000))
            inputs.append(preprocess_board(board))
            labels.extend(record["labels"])
        return np.concatenate(inputs), np.asarray(labels, dtype=np.int64)

    def _official_batch(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = []
        labels = []
        for record in self.rng.sample(self.official_rows, count):
            board = _read_board(Path(record["path"]))
            variant = self.rng.randrange(5)
            if variant == 1:
                board = resize_round_trip(board, 256, contrast=self.rng.uniform(0.3, 0.7))
            elif variant == 2:
                board = rectify_printed_photo(board, self.halftone_screen)
            elif variant == 3:
                board = cv2.GaussianBlur(board, (0, 0), self.rng.uniform(0.2, 0.8))
            elif variant == 4:
                board = resize_round_trip(board, self.rng.randrange(160, 420))
            inputs.append(preprocess_board(board))
            labels.extend(record["labels"])
        return np.concatenate(inputs), np.asarray(labels, dtype=np.int64)

    def _argus_batch(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        examples = [self.argus[self.rng.randrange(len(self.argus))] for _ in range(count)]
        return (
            preprocess_square_crops([image for image, _label in examples]),
            np.asarray([label for _image, label in examples], dtype=np.int64),
        )

    def _retention_loss(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        distillation_weight: float,
    ) -> torch.Tensor:
        images = torch.from_numpy(inputs.copy()).to(self.device)
        targets = torch.from_numpy(labels).long().to(self.device)
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

    def _augment_print(self, board: np.ndarray) -> np.ndarray:
        variant = self.rng.randrange(8)
        if variant == 1:
            return cv2.GaussianBlur(board, (0, 0), self.rng.uniform(0.2, 1.3))
        if variant == 2:
            return resize_round_trip(board, self.rng.randrange(128, 430))
        if variant == 3:
            return jpeg_round_trip(board, self.rng.randrange(45, 96))
        if variant == 4:
            return np.clip(
                board.astype(np.float32) * self.rng.uniform(0.65, 1.35) + self.rng.uniform(-20, 20),
                0,
                255,
            ).astype(np.uint8)
        if variant == 5:
            gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if variant == 6:
            compressed = jpeg_round_trip(board, self.rng.randrange(55, 90))
            return cv2.GaussianBlur(compressed, (0, 0), self.rng.uniform(0.2, 0.9))
        if variant == 7:
            hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= self.rng.uniform(0.3, 1.3)
            hsv[:, :, 2] = np.clip(
                hsv[:, :, 2] * self.rng.uniform(0.8, 1.15),
                0,
                255,
            )
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return board

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
