#!/usr/bin/env python3
"""Reproduce the queen-color fine-tuning stage from hash-verified official sources."""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from evaluate_photo_stress import find_board_boxes
from qa_common import download_verified, sha256_file
from square_model import export_onnx, load_fused_onnx
from train_chess_steps_model import extract_replay_dataset
from training_utils import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "chess-steps-queen-colors.json"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "chess-steps-v1r1.onnx"


class AdaptationDataset(Dataset[tuple[torch.Tensor, int, bool]]):
    def __init__(
        self,
        replay_dir: Path,
        examples: list[dict[str, Any]],
        board_paths: dict[tuple[str, int, int], Path],
        *,
        augmentation: dict[str, float],
    ) -> None:
        self.replay_images = np.load(replay_dir / "images.npy", mmap_mode="r")
        self.replay_labels = np.load(replay_dir / "labels.npy", mmap_mode="r")
        self.examples = examples
        self.board_paths = board_paths
        self.augmentation = augmentation
        self._board_cache: OrderedDict[Path, list[np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.replay_images) + len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, bool]:
        replay_count = len(self.replay_images)
        if index < replay_count:
            square = np.asarray(self.replay_images[index]).copy()
            label = int(self.replay_labels[index])
            is_official = False
        else:
            example = self.examples[index - replay_count]
            locator = (
                str(example["source_id"]),
                int(example["pdf_page_index"]),
                int(example["board_index"]),
            )
            square = self._board_squares(self.board_paths[locator])[int(example["square_index"])]
            label = int(example["label"])
            is_official = True
        if is_official:
            square = augment_square(square, self.augmentation)
        return square_tensor(square), label, is_official

    def _board_squares(self, path: Path) -> list[np.ndarray]:
        squares = self._board_cache.pop(path, None)
        if squares is None:
            board = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if board is None:
                raise ValueError(f"Cannot read rendered board: {path}")
            boundaries = [round(index * board.shape[0] / 8) for index in range(9)]
            squares = [
                board[
                    boundaries[row] : boundaries[row + 1],
                    boundaries[column] : boundaries[column + 1],
                ]
                for row in range(8)
                for column in range(8)
            ]
        self._board_cache[path] = squares
        if len(self._board_cache) > 64:
            self._board_cache.popitem(last=False)
        return squares


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--backup-archive",
        type=Path,
        default=Path.home() / "argus-backups/2026-03-29/argus_data.tar.gz",
    )
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "data/queen-adaptation")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/model-registry")
    parser.add_argument("--version", default="chess-steps-v2-reproduction")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    replay_dir = args.work_dir / "argus-replay"
    extract_replay_dataset(args.backup_archive.expanduser(), replay_dir)
    board_paths = prepare_official_boards(manifest, args.work_dir)
    device = resolve_device(args.device)
    teacher = load_fused_onnx(args.base_model).eval().to(device)
    model_path = args.base_model

    for stage_index, stage in enumerate(manifest["training_stages"], start=1):
        examples = examples_for_stage(manifest, stage)
        model = load_fused_onnx(model_path).to(device)
        stage_path = args.work_dir / f"stage-{stage_index}.onnx"
        train_stage(
            model,
            teacher=teacher,
            replay_dir=replay_dir,
            examples=examples,
            board_paths=board_paths,
            stage=stage,
            batch_size=args.batch_size,
            device=device,
            checkpoint_path=stage_path,
        )
        model_path = stage_path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"{args.version}.onnx"
    checkpoint_path = args.output_dir / f"{args.version}.pt"
    metadata_path = args.output_dir / f"{args.version}.json"
    artifact_path.write_bytes(model_path.read_bytes())
    final_model = load_fused_onnx(artifact_path)
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "version": args.version,
            "architecture": "fused_tiny_square_cnn",
            "base_model": args.base_model.stem,
            "input_size": 64,
            "num_classes": 13,
        },
        checkpoint_path,
    )
    metadata = {
        "version": args.version,
        "base_model": args.base_model.stem,
        "artifact_sha256": sha256_file(artifact_path),
        "manifest": str(args.manifest),
        "training_stages": manifest["training_stages"],
        "training_examples": len(manifest["examples"]),
        "eligible_for_promotion": False,
        "requires_gate_evaluation": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Candidate written to {artifact_path}")
    print(
        "Run make qa-online, make qa-stress, and the fixed replay/holdout gates before promotion."
    )


def prepare_official_boards(
    manifest: dict[str, Any],
    work_dir: Path,
) -> dict[tuple[str, int, int], Path]:
    requested: dict[str, set[tuple[int, int]]] = {}
    for example in manifest["examples"]:
        requested.setdefault(str(example["source_id"]), set()).add(
            (int(example["pdf_page_index"]), int(example["board_index"]))
        )

    sources = {str(source["id"]): source for source in manifest["sources"]}
    pdf_dir = work_dir / "official-pdfs"
    board_dir = work_dir / "official-boards"
    board_paths: dict[tuple[str, int, int], Path] = {}
    for source_id, locators in requested.items():
        source = sources[source_id]
        pdf_path = pdf_dir / f"{source_id}.pdf"
        download_verified(source["url"], source["sha256"], pdf_path)
        board_paths.update(render_requested_boards(pdf_path, source_id, locators, board_dir))
    return board_paths


def render_requested_boards(
    pdf_path: Path,
    source_id: str,
    requested: set[tuple[int, int]],
    output_dir: Path,
) -> dict[tuple[str, int, int], Path]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required. Run with `uv run --extra ml --with "
            "'pymupdf>=1.25,<2' python scripts/train_queen_adaptation.py`."
        ) from exc

    output: dict[tuple[str, int, int], Path] = {}
    with fitz.open(pdf_path) as document:
        for page_index in sorted({page for page, _ in requested}):
            page = document[page_index]
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(3, 3),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            rgb = np.frombuffer(pixmap.samples, np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )
            page_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for board_index, (x, y, width, height) in enumerate(find_board_boxes(page_bgr)):
                if (page_index, board_index) not in requested:
                    continue
                path = output_dir / source_id / f"p{page_index:02d}-b{board_index:02d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    board = page_bgr[y : y + height, x : x + width]
                    board = cv2.resize(board, (512, 512), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(path), board, [cv2.IMWRITE_JPEG_QUALITY, 98])
                output[(source_id, page_index, board_index)] = path
    missing = {(source_id, *locator) for locator in requested} - output.keys()
    if missing:
        raise ValueError(f"Could not render official boards at: {sorted(missing)}")
    return output


def examples_for_stage(manifest: dict[str, Any], stage: dict[str, Any]) -> list[dict[str, Any]]:
    queen_stages = set(stage["queen_stages"])
    selected = [
        example
        for example in manifest["examples"]
        if example["kind"] == "king" or example["stage"] in queen_stages
    ]
    selected *= int(stage["repeat"])
    hard_repeat = int(stage.get("hard_example_repeat", 0))
    if hard_repeat:
        hard_keys = {
            (
                example["source_id"],
                example["pdf_page_index"],
                example["board_index"],
                example["square_index"],
                example["label"],
            )
            for example in manifest["stage3_hard_examples"]
        }
        hard = [
            example
            for example in manifest["examples"]
            if (
                example["source_id"],
                example["pdf_page_index"],
                example["board_index"],
                example["square_index"],
                example["label"],
            )
            in hard_keys
        ]
        selected.extend(hard * hard_repeat)
    return selected


def train_stage(
    model: torch.nn.Module,
    *,
    teacher: torch.nn.Module,
    replay_dir: Path,
    examples: list[dict[str, Any]],
    board_paths: dict[tuple[str, int, int], Path],
    stage: dict[str, Any],
    batch_size: int,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    seed = int(stage["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = AdaptationDataset(
        replay_dir,
        examples,
        board_paths,
        augmentation=stage["augmentation"],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(stage.get("loader_seed", seed))),
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in list(model.features)[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": float(stage["classifier_lr"])},
            {
                "params": [p for p in model.features.parameters() if p.requires_grad],
                "lr": float(stage["feature_lr"]),
            },
        ],
        weight_decay=1e-5,
    )

    for epoch in range(int(stage["epochs"])):
        model.train()
        total_loss = 0.0
        item_count = 0
        for images, labels, is_official in loader:
            images = images.to(device)
            labels = labels.to(device)
            is_official = is_official.to(device)
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
            replay = ~is_official
            if replay.any():
                with torch.no_grad():
                    teacher_logits = teacher(images[replay])
                loss += (
                    functional.kl_div(
                        functional.log_softmax(logits[replay] / 2, dim=1),
                        functional.softmax(teacher_logits / 2, dim=1),
                        reduction="batchmean",
                    )
                    * 4
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            item_count += len(labels)
        print(f"{stage['name']} epoch {epoch + 1}: loss={total_loss / item_count:.6f}")
        export_onnx(model, checkpoint_path)
        model.to(device)


def augment_square(square: np.ndarray, settings: dict[str, float]) -> np.ndarray:
    height, width = square.shape[:2]
    transform = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        random.uniform(-settings["rotation"], settings["rotation"]),
        random.uniform(settings["scale_min"], settings["scale_max"]),
    )
    translation = settings["translation"]
    transform[:, 2] += [
        random.uniform(-translation, translation),
        random.uniform(-translation, translation),
    ]
    augmented = cv2.warpAffine(
        square,
        transform,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.clip(
        augmented.astype(np.float32) * random.uniform(settings["gain_min"], settings["gain_max"])
        + random.uniform(-settings["offset"], settings["offset"]),
        0,
        255,
    ).astype(np.uint8)


def square_tensor(square: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= 64 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (64, 64), interpolation=interpolation)
    return torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1).copy())


if __name__ == "__main__":
    main()
