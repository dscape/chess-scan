"""Shared helpers for model-training scripts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import Dataset

from chess_scan.classifier import preprocess_square_crops

Batch = TypeVar("Batch")
REPLAY_WORKERS = 2


class ArgusReplayDataset(Dataset[tuple[np.ndarray, int]]):
    """Lazily read prepared and synthetic Argus squares from memory-mapped arrays."""

    def __init__(self, data_dir: Path, *, source: str = "all") -> None:
        if source not in {"all", "argus", "synthetic"}:
            raise ValueError(f"Unknown Argus replay source: {source}")
        prepared_dir = data_dir / "prepared"
        replay_dir = data_dir / "data" / "piece_classifier_dataset"
        self.position_images_path = prepared_dir / "train-images.npy"
        self.position_labels_path = prepared_dir / "train-labels.npy"
        self.synthetic_images_path = replay_dir / "images.npy"
        self.synthetic_labels_path = replay_dir / "labels.npy"
        self.position_count = len(np.load(self.position_labels_path, mmap_mode="r"))
        self.synthetic_count = len(np.load(self.synthetic_labels_path, mmap_mode="r"))
        self.source = source
        self._arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        if self.source == "argus":
            return self.position_count
        if self.source == "synthetic":
            return self.synthetic_count
        return self.position_count + self.synthetic_count

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        position_images, position_labels, synthetic_images, synthetic_labels = self._load_arrays()
        if self.source == "synthetic":
            image = np.asarray(synthetic_images[index])
            label = int(synthetic_labels[index])
        elif self.source == "argus" or index < self.position_count:
            image = np.asarray(position_images[index])
            label = int(position_labels[index])
        else:
            replay_index = index - self.position_count
            image = np.asarray(synthetic_images[replay_index])
            label = int(synthetic_labels[replay_index])
        return image, label

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._arrays is None:
            self._arrays = (
                np.load(self.position_images_path, mmap_mode="r"),
                np.load(self.position_labels_path, mmap_mode="r"),
                np.load(self.synthetic_images_path, mmap_mode="r"),
                np.load(self.synthetic_labels_path, mmap_mode="r"),
            )
        return self._arrays


def collate_replay_batch(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch, strict=True)
    inputs = preprocess_square_crops(list(images))
    return torch.from_numpy(inputs.copy()), torch.tensor(labels, dtype=torch.long)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    return (
        functional.kl_div(
            functional.log_softmax(student_logits / temperature, dim=1),
            functional.softmax(teacher_logits.detach() / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )


def repeat_loader(loader: torch.utils.data.DataLoader[Batch]) -> Iterator[Batch]:
    while True:
        yield from loader
