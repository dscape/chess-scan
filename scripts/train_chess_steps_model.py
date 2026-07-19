#!/usr/bin/env python3
"""Reproduce the Chess Steps domain-adapted base model from official samples and Argus replay."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from chess_scan.board import CLASS_NAMES
from chess_scan.classifier import (
    INPUT_SIZE,
    NUM_CLASSES,
    preprocess_square_crops,
    split_board_squares,
)
from chess_scan.config import PROJECT_ROOT
from chess_scan.model_artifact import sha256_file
from square_model import export_onnx, load_fused_onnx
from training_utils import resolve_device

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "argus-v2r5.onnx"
REFERENCE_BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "chess-steps-step2.json"
KING_GATE_BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "chess-steps-kings.json"
OFFICIAL_BASE_URL = "https://www.stappenmethode.nl/en/"
OFFICIAL_SAMPLE_PATHS = (
    "lp/en_lp_1.pdf",
    "lp/en_lp_1e.pdf",
    "lp/en_lp_1m.pdf",
    "lp/en_lp_1p.pdf",
    "lp/en_lp_2.pdf",
    "lp/en_lp_2e.pdf",
    "lp/en_lp_2m.pdf",
    "lp/en_lp_2p.pdf",
    "lp/en_lp_3.pdf",
    "lp/en_lp_3e.pdf",
    "lp/en_lp_3m.pdf",
    "lp/en_lp_3p.pdf",
    "lp/en_lp_4.pdf",
    "lp/en_lp_4e.pdf",
    "lp/en_lp_4m.pdf",
    "lp/en_lp_4p.pdf",
    "lp/en_lp_5m.pdf",
    "lp/en_lp_5p.pdf",
    "lp/en_lp_6.pdf",
    "lp/en_lp_6e.pdf",
    "lp/in_lp_2vd.pdf",
    "lp/in_lp_3vd.pdf",
    "lp/in_lp_5.pdf",
    "lp/in_lp_5e.pdf",
)
REPLAY_ARCHIVE_MEMBERS = {
    "data/piece_classifier_dataset/images.npy": "images.npy",
    "data/piece_classifier_dataset/labels.npy": "labels.npy",
    "data/piece_classifier_dataset/metadata.json": "metadata.json",
}


@dataclass(frozen=True, slots=True)
class TemplateReferences:
    features: tuple[tuple[np.ndarray, ...], ...]
    squared_norms: tuple[tuple[np.ndarray, ...], ...]


class MixedSquareDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        replay_dir: Path,
        replay_indices: list[int],
        official_examples: list[tuple[tuple[str, int], int]],
        *,
        augment: bool,
    ) -> None:
        self.replay_images = np.load(replay_dir / "images.npy", mmap_mode="r")
        self.replay_labels = np.load(replay_dir / "labels.npy", mmap_mode="r")
        self.items: list[tuple[str, int | tuple[str, int], int]] = [
            ("replay", index, int(self.replay_labels[index])) for index in replay_indices
        ]
        self.items.extend(("official", key, label) for key, label in official_examples)
        self.augment = augment
        self._board_cache: OrderedDict[str, list[np.ndarray]] = OrderedDict()

    @property
    def targets(self) -> list[int]:
        return [item[2] for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        source, key, label = self.items[index]
        if source == "replay":
            image = np.asarray(self.replay_images[int(key)]).copy()
        else:
            path, square_index = key  # type: ignore[misc]
            image = self._official_squares(path)[square_index]
        return square_tensor(image, augment=self.augment), label

    def _official_squares(self, path: str) -> list[np.ndarray]:
        cached = self._board_cache.pop(path, None)
        if cached is None:
            board = cv2.imread(path, cv2.IMREAD_COLOR)
            if board is None:
                raise ValueError(f"Cannot read official board crop: {path}")
            cached = split_board_squares(board)
        self._board_cache[path] = cached
        if len(self._board_cache) > 64:
            self._board_cache.popitem(last=False)
        return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backup-archive",
        type=Path,
        default=Path.home() / "argus-backups/2026-03-29/argus_data.tar.gz",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "data/base-model-training",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--version", default="chess-steps-v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Preparing data under {args.work_dir}; device={device}")

    reference_benchmark = json.loads(REFERENCE_BENCHMARK_PATH.read_text())
    king_gate_benchmark = json.loads(KING_GATE_BENCHMARK_PATH.read_text())
    replay_dir, board_dir = prepare_data(
        backup_archive=args.backup_archive.expanduser(),
        work_dir=args.work_dir,
        reference_benchmark=reference_benchmark,
        king_gate_benchmark=king_gate_benchmark,
    )
    label_rows = load_or_label_official_boards(board_dir, reference_benchmark, args.work_dir)
    train_indices, replay_gate_indices = split_replay(replay_dir, seed=args.seed)
    official = select_official_examples(
        label_rows,
        excluded_sources={"en_lp_2", "en_lp_2e", "en_lp_3"},
        seed=args.seed,
    )

    train_dataset = MixedSquareDataset(
        replay_dir,
        train_indices,
        official * 2,
        augment=True,
    )
    replay_gate = MixedSquareDataset(
        replay_dir,
        replay_gate_indices,
        [],
        augment=False,
    )
    train_loader = balanced_loader(
        train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    replay_gate_loader = DataLoader(replay_gate, batch_size=512, shuffle=False, num_workers=0)

    model = load_fused_onnx(BASE_MODEL_PATH)
    verify_reconstruction(model, BASE_MODEL_PATH)
    model.to(device)
    baseline_metrics = evaluate_model(
        model,
        replay_gate_loader=replay_gate_loader,
        label_rows=label_rows,
        reference_benchmark=reference_benchmark,
        king_gate_benchmark=king_gate_benchmark,
        device=device,
    )
    print("Baseline:", json.dumps(baseline_metrics, indent=2))

    best_state = train(
        model,
        train_loader=train_loader,
        replay_gate_loader=replay_gate_loader,
        label_rows=label_rows,
        benchmark=reference_benchmark,
        epochs=args.epochs,
        device=device,
    )
    model.load_state_dict(best_state)
    model.to(device).eval()
    candidate_metrics = evaluate_model(
        model,
        replay_gate_loader=replay_gate_loader,
        label_rows=label_rows,
        reference_benchmark=reference_benchmark,
        king_gate_benchmark=king_gate_benchmark,
        device=device,
    )
    eligible = passes_gates(candidate_metrics, baseline_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / f"{args.version}.onnx"
    checkpoint_path = args.output_dir / f"{args.version}.pt"
    metadata_path = args.output_dir / f"{args.version}.json"
    export_onnx(model, onnx_path)
    verify_reconstruction(model, onnx_path)
    torch.save(
        {
            "state_dict": best_state,
            "version": args.version,
            "architecture": "fused_tiny_square_cnn",
            "base_model": "argus-v2r5",
            "input_size": INPUT_SIZE,
            "num_classes": NUM_CLASSES,
        },
        checkpoint_path,
    )

    metadata = build_metadata(
        version=args.version,
        args=args,
        reference_benchmark=reference_benchmark,
        king_gate_benchmark=king_gate_benchmark,
        label_rows=label_rows,
        train_dataset=train_dataset,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        eligible=eligible,
        onnx_path=onnx_path,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Candidate:", json.dumps(candidate_metrics, indent=2))
    print(f"Artifact: {onnx_path}")
    if not eligible:
        raise SystemExit("Candidate failed one or more promotion gates")
    print("All Chess Steps king and replay gates passed")


def prepare_data(
    *,
    backup_archive: Path,
    work_dir: Path,
    reference_benchmark: dict[str, Any],
    king_gate_benchmark: dict[str, Any],
) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = work_dir / "argus-replay"
    pdf_dir = work_dir / "official-pdfs"
    board_dir = work_dir / "official-boards"
    extract_replay_dataset(backup_archive, replay_dir)
    download_official_samples(pdf_dir)
    verify_benchmark_pdf(pdf_dir / "en_lp_2.pdf", reference_benchmark)
    verify_benchmark_pdf(pdf_dir / "en_lp_3.pdf", king_gate_benchmark)
    render_official_boards(pdf_dir, board_dir)
    return replay_dir, board_dir


def extract_replay_dataset(archive_path: Path, output_dir: Path) -> None:
    expected = [output_dir / name for name in REPLAY_ARCHIVE_MEMBERS.values()]
    if all(path.exists() for path in expected):
        return
    if not archive_path.exists():
        raise FileNotFoundError(f"Argus backup archive not found: {archive_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member_name, output_name in REPLAY_ARCHIVE_MEMBERS.items():
            source = archive.extractfile(member_name)
            if source is None:
                raise FileNotFoundError(f"Backup is missing {member_name}")
            with source, (output_dir / output_name).open("wb") as destination:
                shutil.copyfileobj(source, destination)


def download_official_samples(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in OFFICIAL_SAMPLE_PATHS:
        destination = output_dir / Path(relative_path).name
        if destination.exists():
            continue
        url = OFFICIAL_BASE_URL + relative_path
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, destination)


def verify_benchmark_pdf(path: Path, benchmark: dict[str, Any]) -> None:
    actual = sha256_file(path)
    expected = str(benchmark["source_sha256"])
    if actual != expected:
        raise ValueError(f"Official benchmark PDF hash changed: expected {expected}, got {actual}")


def render_official_boards(pdf_dir: Path, output_dir: Path) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required only for dataset preparation. Run with "
            "`uv run --extra ml --with pymupdf scripts/train_chess_steps_model.py`."
        ) from exc

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pdf_output = output_dir / pdf_path.stem
        marker = pdf_output / ".complete"
        if _render_is_complete(pdf_output, marker):
            continue
        pdf_output.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)
        for stale_image in pdf_output.glob("*.jpg"):
            stale_image.unlink()

        written = 0
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document):
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
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for board_index, (x, y, width, height) in enumerate(find_board_boxes(bgr)):
                    board = bgr[y : y + height, x : x + width]
                    board = cv2.resize(board, (512, 512), interpolation=cv2.INTER_AREA)
                    destination = pdf_output / f"p{page_index:02d}-b{board_index:02d}.jpg"
                    if not cv2.imwrite(
                        str(destination),
                        board,
                        [cv2.IMWRITE_JPEG_QUALITY, 98],
                    ):
                        raise OSError(f"Failed to write rendered board: {destination}")
                    written += 1

        rendered = sorted(pdf_output.glob("*.jpg"))
        if written == 0 or len(rendered) != written:
            raise RuntimeError(f"Incomplete board rendering for {pdf_path.name}")
        if any(path.stat().st_size == 0 for path in rendered):
            raise RuntimeError(f"Rendered board is empty for {pdf_path.name}")
        marker.write_text(f"{written}\n")
        print(f"Rendered {written} candidate diagrams from {pdf_path.name}")


def _render_is_complete(output_dir: Path, marker: Path) -> bool:
    if not marker.exists():
        return False
    try:
        expected = int(marker.read_text().strip())
    except ValueError:
        return False
    images = sorted(output_dir.glob("*.jpg"))
    return (
        expected > 0 and len(images) == expected and all(path.stat().st_size > 0 for path in images)
    )


def find_board_boxes(page_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if 300 <= width <= 600 and 300 <= height <= 600 and 0.96 <= width / height <= 1.04:
            candidates.append((x, y, width, height))

    unique: list[tuple[int, int, int, int]] = []
    for box in sorted(candidates, key=lambda item: (item[1], item[0])):
        x, y, width, height = box
        duplicate = any(
            abs(x - prior_x) < 5
            and abs(y - prior_y) < 5
            and abs(width - prior_width) < 5
            and abs(height - prior_height) < 5
            for prior_x, prior_y, prior_width, prior_height in unique
        )
        if not duplicate:
            unique.append(box)
    return unique


def load_or_label_official_boards(
    board_dir: Path,
    benchmark: dict[str, Any],
    work_dir: Path,
) -> list[dict[str, Any]]:
    manifest_path = work_dir / "official-labels.json"
    if manifest_path.exists():
        rows: list[dict[str, Any]] = json.loads(manifest_path.read_text())
        if rows and all(cv2.imread(str(row["path"]), cv2.IMREAD_COLOR) is not None for row in rows):
            return rows

    references = build_glyph_references(board_dir, benchmark)
    rows = []
    for path in sorted(board_dir.glob("*/*.jpg")):
        board = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if board is None:
            raise ValueError(f"Cannot read rendered board: {path}")
        labels = classify_template_squares(split_board_squares(board), references=references)
        rows.append(
            {
                "path": str(path.resolve()),
                "source": path.parent.name,
                "labels": labels,
                "white_king_squares": [index for index, label in enumerate(labels) if label == 6],
                "black_king_squares": [index for index, label in enumerate(labels) if label == 12],
            }
        )
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Template-labeled {len(rows)} candidate diagrams")
    return rows


def build_glyph_references(
    board_dir: Path,
    benchmark: dict[str, Any],
) -> TemplateReferences:
    grouped: list[list[list[np.ndarray]]] = [[[] for _ in range(NUM_CLASSES)] for _ in range(2)]
    for position in benchmark["positions"]:
        board_index = int(position["board_index"])
        path = board_dir / "en_lp_2" / f"p01-b{board_index:02d}.jpg"
        board = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if board is None:
            raise FileNotFoundError(f"Missing benchmark board crop: {path}")
        expected = labels_from_fen(str(position["fen"]))
        for square_index, square in enumerate(split_board_squares(board)):
            grouped[square_parity(square_index)][expected[square_index]].append(
                template_feature(square)
            )

    feature_groups: list[tuple[np.ndarray, ...]] = []
    norm_groups: list[tuple[np.ndarray, ...]] = []
    for parity_groups in grouped:
        if any(not class_features for class_features in parity_groups):
            raise ValueError("Template references do not cover every class and square parity")
        arrays = tuple(np.stack(class_features) for class_features in parity_groups)
        feature_groups.append(arrays)
        norm_groups.append(tuple(np.einsum("ij,ij->i", array, array) for array in arrays))
    return TemplateReferences(tuple(feature_groups), tuple(norm_groups))


def classify_template_squares(
    squares: list[np.ndarray],
    *,
    references: TemplateReferences,
) -> list[int]:
    target_features = np.stack([template_feature(square) for square in squares])
    scores = np.full((len(squares), NUM_CLASSES), np.inf, dtype=np.float32)
    parities = np.asarray([square_parity(index) for index in range(len(squares))])
    for parity in range(2):
        indices = np.flatnonzero(parities == parity)
        targets = target_features[indices]
        target_norms = np.einsum("ij,ij->i", targets, targets)[:, None]
        for class_id in range(NUM_CLASSES):
            candidates = references.features[parity][class_id]
            distances = (
                target_norms
                + references.squared_norms[parity][class_id][None, :]
                - 2.0 * targets @ candidates.T
            )
            scores[indices, class_id] = distances.min(axis=1)
    return scores.argmin(axis=1).astype(int).tolist()


def template_feature(square_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
    smoothed = cv2.GaussianBlur(histogram.reshape(-1, 1), (1, 9), 0).ravel()
    background = float(np.argmax(smoothed))
    normalized = np.where(
        gray < background,
        (gray - background) / max(background, 1.0),
        (gray - background) / max(255.0 - background, 1.0),
    )
    normalized[np.abs(normalized) < 0.04] = 0
    return cv2.resize(normalized, (32, 32), interpolation=cv2.INTER_AREA).ravel()


def select_official_examples(
    rows: list[dict[str, Any]],
    *,
    excluded_sources: set[str],
    seed: int,
) -> list[tuple[tuple[str, int], int]]:
    per_class: list[list[tuple[tuple[str, int], int]]] = [[] for _ in range(NUM_CLASSES)]
    for row in rows:
        if row["source"] in excluded_sources or not has_exactly_two_kings(row):
            continue
        for square_index, label in enumerate(row["labels"]):
            per_class[label].append(((row["path"], square_index), label))

    rng = random.Random(seed)
    selected: list[tuple[tuple[str, int], int]] = []
    for examples in per_class:
        rng.shuffle(examples)
        selected.extend(examples[:1200])
    return selected


def split_replay(replay_dir: Path, *, seed: int) -> tuple[list[int], list[int]]:
    labels = np.load(replay_dir / "labels.npy")
    rng = np.random.RandomState(seed)
    train: list[int] = []
    gate: list[int] = []
    for class_id in range(NUM_CLASSES):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        train.extend(indices[:1300].tolist())
        gate.extend(indices[1300:].tolist())
    return train, gate


def balanced_loader(
    dataset: MixedSquareDataset,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, int]]:
    targets = np.asarray(dataset.targets)
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    sample_weights = 1.0 / counts[targets]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights).double(),
        num_samples=26_000,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Training class counts before balanced sampling: {counts.tolist()}")
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def train(
    model: nn.Module,
    *,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    replay_gate_loader: DataLoader[tuple[torch.Tensor, int]],
    label_rows: list[dict[str, Any]],
    benchmark: dict[str, Any],
    epochs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_rank: tuple[float, ...] | None = None

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            loss = functional.cross_entropy(
                model(images),
                labels,
                label_smoothing=0.01,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            example_count += len(labels)
        scheduler.step()

        replay_accuracy = evaluate_replay(model, replay_gate_loader, device=device)
        evaluation_rows = _training_evaluation_rows(label_rows, benchmark)
        predictions = predict_board_rows(model, evaluation_rows, device=device)
        selection = evaluate_king_positions(
            label_rows,
            predictions,
            source="en_lp_2e",
        )
        manual = evaluate_manual_benchmark(benchmark, label_rows, predictions)
        print(
            f"epoch={epoch + 1:02d} loss={loss_sum / example_count:.4f} "
            f"replay={replay_accuracy:.4f} selection_kings={selection['king_accuracy']:.4f} "
            f"selection_exact={selection['exact_position_rate']:.4f} "
            f"manual_board_exact={manual['board_exact']:.4f}"
        )
        rank = (
            selection["exact_position_rate"],
            selection["king_accuracy"],
            manual["king_accuracy"],
            replay_accuracy,
            manual["square_accuracy"],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_state


def evaluate_model(
    model: nn.Module,
    *,
    replay_gate_loader: DataLoader[tuple[torch.Tensor, int]],
    label_rows: list[dict[str, Any]],
    reference_benchmark: dict[str, Any],
    king_gate_benchmark: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    evaluation_rows = _model_evaluation_rows(
        label_rows,
        reference_benchmark=reference_benchmark,
        king_gate_benchmark=king_gate_benchmark,
    )
    predictions = predict_board_rows(model, evaluation_rows, device=device)
    return {
        "replay_square_accuracy": evaluate_replay(model, replay_gate_loader, device=device),
        "selection_en_lp_2e": evaluate_king_positions(
            label_rows,
            predictions,
            source="en_lp_2e",
        ),
        "independent_manual_king_gate": evaluate_manual_king_gate(
            king_gate_benchmark,
            label_rows,
            predictions,
        ),
        "official_standard": evaluate_king_positions(
            label_rows,
            predictions,
            source=None,
        ),
        "template_reference_step2_page": evaluate_manual_benchmark(
            reference_benchmark,
            label_rows,
            predictions,
        ),
    }


def evaluate_replay(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    *,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += len(labels)
    return correct / total


def evaluate_king_positions(
    rows: list[dict[str, Any]],
    predictions_by_path: dict[str, list[int]],
    *,
    source: str | None,
) -> dict[str, float]:
    correct = 0
    total = 0
    exact = 0
    board_count = 0
    for row in rows:
        if (source is not None and row["source"] != source) or not has_exactly_two_kings(row):
            continue
        predictions = predictions_by_path[row["path"]]
        expected = {
            row["white_king_squares"][0]: 6,
            row["black_king_squares"][0]: 12,
        }
        correct += sum(predictions[index] == label for index, label in expected.items())
        total += 2
        predicted_king_squares = {
            index for index, label in enumerate(predictions) if label in (6, 12)
        }
        exact += all(predictions[index] == label for index, label in expected.items()) and (
            predicted_king_squares == set(expected)
        )
        board_count += 1
    return {
        "boards": float(board_count),
        "king_accuracy": correct / total,
        "exact_position_rate": exact / board_count,
    }


def evaluate_manual_benchmark(
    benchmark: dict[str, Any],
    rows: list[dict[str, Any]],
    predictions_by_path: dict[str, list[int]],
) -> dict[str, float]:
    by_name = {Path(row["path"]).name: row for row in rows if row["source"] == "en_lp_2"}
    square_correct = 0
    king_correct = 0
    board_exact = 0
    for position in benchmark["positions"]:
        name = f"p01-b{int(position['board_index']):02d}.jpg"
        row = by_name[name]
        predicted = predictions_by_path[row["path"]]
        expected = labels_from_fen(str(position["fen"]))
        square_correct += sum(
            left == right for left, right in zip(predicted, expected, strict=True)
        )
        king_correct += sum(
            predicted[index] == expected[index] for index in range(64) if expected[index] in (6, 12)
        )
        board_exact += predicted == expected
    count = len(benchmark["positions"])
    return {
        "boards": float(count),
        "square_accuracy": square_correct / (count * 64),
        "king_accuracy": king_correct / (count * 2),
        "board_exact": board_exact / count,
    }


def evaluate_manual_king_gate(
    benchmark: dict[str, Any],
    rows: list[dict[str, Any]],
    predictions_by_path: dict[str, list[int]],
) -> dict[str, float]:
    source = Path(str(benchmark["source_url"])).stem
    by_name = {Path(row["path"]).name: row for row in rows if row["source"] == source}
    correct = 0
    exact = 0
    for position in benchmark["positions"]:
        name = f"p01-b{int(position['board_index']):02d}.jpg"
        row = by_name[name]
        predicted = predictions_by_path[row["path"]]
        expected = {
            int(position["white_king_square"]): 6,
            int(position["black_king_square"]): 12,
        }
        correct += sum(predicted[index] == label for index, label in expected.items())
        predicted_king_squares = {
            index for index, label in enumerate(predicted) if label in (6, 12)
        }
        exact += all(predicted[index] == label for index, label in expected.items()) and (
            predicted_king_squares == set(expected)
        )
    board_count = len(benchmark["positions"])
    return {
        "boards": float(board_count),
        "king_accuracy": correct / (board_count * 2),
        "exact_position_rate": exact / board_count,
    }


def passes_gates(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        candidate["replay_square_accuracy"] >= baseline["replay_square_accuracy"]
        and candidate["selection_en_lp_2e"]["exact_position_rate"] == 1.0
        and candidate["independent_manual_king_gate"]["exact_position_rate"] == 1.0
        and candidate["official_standard"]["exact_position_rate"] == 1.0
        and candidate["template_reference_step2_page"]["king_accuracy"] == 1.0
        and candidate["template_reference_step2_page"]["board_exact"] == 1.0
    )


def build_metadata(
    *,
    version: str,
    args: argparse.Namespace,
    reference_benchmark: dict[str, Any],
    king_gate_benchmark: dict[str, Any],
    label_rows: list[dict[str, Any]],
    train_dataset: MixedSquareDataset,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    eligible: bool,
    onnx_path: Path,
) -> dict[str, Any]:
    standard_count = sum(has_exactly_two_kings(row) for row in label_rows)
    return {
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "runtime_format": "onnx",
        "architecture": "fused_tiny_square_cnn",
        "base_model": "argus-v2r5",
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "seed": args.seed,
        "epochs": args.epochs,
        "training_items_before_sampling": len(train_dataset),
        "eligible_for_promotion": eligible,
        "artifact_sha256": sha256_file(onnx_path),
        "sources": {
            "argus_backup": "/".join(args.backup_archive.parts[-2:]),
            "argus_replay_squares": 19_500,
            "official_sample_base_url": OFFICIAL_BASE_URL,
            "official_sample_pdfs": len(OFFICIAL_SAMPLE_PATHS),
            "detected_candidate_diagrams": len(label_rows),
            "standard_two_king_diagrams": standard_count,
            "official_images_redistributed": False,
            "labeling": (
                "Class templates from 12 manually audited Step 2 diagrams; only diagrams with "
                "exactly one template-matched king of each color enter training"
            ),
        },
        "benchmarks": {
            "template_reference_manifest": str(REFERENCE_BENCHMARK_PATH.relative_to(PROJECT_ROOT)),
            "template_reference_url": reference_benchmark["source_url"],
            "independent_king_gate_manifest": str(
                KING_GATE_BENCHMARK_PATH.relative_to(PROJECT_ROOT)
            ),
            "independent_king_gate_url": king_gate_benchmark["source_url"],
            "training_pixel_exclusion": ["en_lp_2", "en_lp_2e", "en_lp_3"],
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "promotion_gates": (
            "100% independent manually audited king gate; 100% king positions on official "
            "standard diagrams; 100% exact template-reference page; Argus replay accuracy "
            "does not regress"
        ),
    }


def square_tensor(image_bgr: np.ndarray, *, augment: bool = False) -> torch.Tensor:
    image = augment_square(image_bgr) if augment else image_bgr
    return torch.from_numpy(preprocess_square_crops([image])[0])


def augment_square(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    transform = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        random.uniform(-3.0, 3.0),
        random.uniform(0.91, 1.09),
    )
    transform[:, 2] += [random.uniform(-3, 3), random.uniform(-3, 3)]
    image = cv2.warpAffine(
        image_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    image = np.clip(
        image.astype(np.float32) * random.uniform(0.68, 1.28) + random.uniform(-24, 24),
        0,
        255,
    ).astype(np.uint8)
    if random.random() < 0.45:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 1.1))
    if random.random() < 0.35:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, random.randint(42, 92)],
        )
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image


def predict_board_rows(
    model: nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    board_batch_size: int = 16,
) -> dict[str, list[int]]:
    unique_paths = list(dict.fromkeys(str(row["path"]) for row in rows))
    predictions: dict[str, list[int]] = {}
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(unique_paths), board_batch_size):
            paths = unique_paths[offset : offset + board_batch_size]
            board_inputs: list[np.ndarray] = []
            for path in paths:
                board = cv2.imread(path, cv2.IMREAD_COLOR)
                if board is None:
                    raise ValueError(f"Cannot read official board crop: {path}")
                board_inputs.append(preprocess_square_crops(split_board_squares(board)))
            inputs = torch.from_numpy(np.concatenate(board_inputs)).to(device)
            batch_predictions = model(inputs).argmax(dim=1).reshape(len(paths), 64).cpu().tolist()
            predictions.update(zip(paths, batch_predictions, strict=True))
    return predictions


def _model_evaluation_rows(
    rows: list[dict[str, Any]],
    *,
    reference_benchmark: dict[str, Any],
    king_gate_benchmark: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_names = {
        f"p01-b{int(position['board_index']):02d}.jpg"
        for position in reference_benchmark["positions"]
    }
    king_source = Path(str(king_gate_benchmark["source_url"])).stem
    king_names = {
        f"p01-b{int(position['board_index']):02d}.jpg"
        for position in king_gate_benchmark["positions"]
    }
    return [
        row
        for row in rows
        if has_exactly_two_kings(row)
        or (row["source"] == "en_lp_2" and Path(row["path"]).name in reference_names)
        or (row["source"] == king_source and Path(row["path"]).name in king_names)
    ]


def _training_evaluation_rows(
    rows: list[dict[str, Any]],
    benchmark: dict[str, Any],
) -> list[dict[str, Any]]:
    benchmark_names = {
        f"p01-b{int(position['board_index']):02d}.jpg" for position in benchmark["positions"]
    }
    return [
        row
        for row in rows
        if (row["source"] == "en_lp_2e" and has_exactly_two_kings(row))
        or (row["source"] == "en_lp_2" and Path(row["path"]).name in benchmark_names)
    ]


def labels_from_fen(fen: str) -> list[int]:
    labels: list[int] = []
    for rank in fen.split("/"):
        for character in rank:
            if character.isdigit():
                labels.extend([0] * int(character))
            else:
                labels.append(CLASS_NAMES.index(character))
    if len(labels) != 64:
        raise ValueError(f"Expected 64 labels in FEN, got {len(labels)}: {fen}")
    return labels


def square_parity(square_index: int) -> int:
    return (square_index // 8 + square_index % 8) % 2


def has_exactly_two_kings(row: dict[str, Any]) -> bool:
    return len(row["white_king_squares"]) == 1 and len(row["black_king_squares"]) == 1


def verify_reconstruction(model: nn.Module, onnx_path: Path) -> None:
    inputs = np.random.RandomState(0).randn(8, 3, 64, 64).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    model.eval()
    with torch.no_grad():
        actual = model(torch.from_numpy(inputs)).numpy()
    maximum_delta = float(np.max(np.abs(expected - actual)))
    if maximum_delta >= 2e-4:
        raise ValueError(f"ONNX reconstruction changed logits by {maximum_delta}")
    print(f"ONNX reconstruction maximum logit delta: {maximum_delta:.8f}")


if __name__ == "__main__":
    main()
