"""Trainable square-classifier architecture reconstructed from an optimized ONNX artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import numpy_helper
from torch.nn import functional

from chess_scan.classifier import INPUT_SIZE, NUM_CLASSES

_CONVOLUTION_SPECS = (
    (3, 24, 3, 2, 1),
    (24, 24, 3, 1, 24),
    (24, 32, 1, 1, 1),
    (32, 32, 3, 2, 32),
    (32, 64, 1, 1, 1),
    (64, 64, 3, 1, 64),
    (64, 64, 1, 1, 1),
    (64, 64, 3, 2, 64),
    (64, 96, 1, 1, 1),
    (96, 96, 3, 1, 96),
    (96, 96, 1, 1, 1),
    (96, 96, 3, 2, 96),
    (96, 160, 1, 1, 1),
    (160, 192, 1, 1, 1),
)
_WIDE_LAYER_SPECS = (
    (3, 32, 5, 2),
    (32, 64, 3, 2),
    (64, 96, 3, 2),
    (96, 160, 3, 2),
    (160, 256, 3, 1),
    (256, 384, 1, 1),
)


class ConvAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.SiLU(inplace=True),
        )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class FusedTinySquareClassifier(nn.Module):
    """Inference-equivalent form of Argus's batch-normalized tiny CNN.

    ONNX optimization folds each batch-normalization layer into its preceding
    convolution. Keeping that fused representation lets later training runs
    initialize from the deployed artifact instead of restarting from random weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(*[ConvAct(*spec) for spec in _CONVOLUTION_SPECS])
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(192, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(self.dropout(pooled))


class FusedWideSquareClassifier(nn.Module):
    """Trainable form of a deployed wide model with batch normalization folded."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(*[ConvAct(*spec, 1) for spec in _WIDE_LAYER_SPECS])
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(384, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(self.dropout(pooled))


class WideSquareClassifier(nn.Module):
    """Higher-capacity square classifier for diverse platform artwork."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(*[ConvNormAct(*spec) for spec in _WIDE_LAYER_SPECS])
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(384, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(self.dropout(pooled))


def load_fused_onnx(path: Path) -> FusedTinySquareClassifier:
    """Reconstruct a trainable tiny model from the deployed ONNX weights."""
    model = FusedTinySquareClassifier()
    _load_fused_weights(path, model)
    return model


def load_fused_wide_onnx(path: Path) -> FusedWideSquareClassifier:
    """Reconstruct a trainable wide model from the deployed ONNX weights."""
    model = FusedWideSquareClassifier()
    _load_fused_weights(path, model)
    return model


def _load_fused_weights(
    path: Path,
    model: FusedTinySquareClassifier | FusedWideSquareClassifier,
) -> None:
    graph = onnx.load(path).graph
    initializers = {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in graph.initializer
    }
    convolution_nodes = [node for node in graph.node if node.op_type == "Conv"]
    gemm_nodes = [node for node in graph.node if node.op_type == "Gemm"]

    convolutions = [block[0] for block in model.features]
    if len(convolution_nodes) != len(convolutions) or len(gemm_nodes) != 1:
        raise ValueError(
            f"Unsupported ONNX graph: found {len(convolution_nodes)} convolutions and "
            f"{len(gemm_nodes)} linear classifiers"
        )

    for node, convolution in zip(convolution_nodes, convolutions, strict=True):
        convolution.weight.data.copy_(torch.from_numpy(initializers[node.input[1]]))
        convolution.bias.data.copy_(torch.from_numpy(initializers[node.input[2]]))

    gemm = gemm_nodes[0]
    model.classifier.weight.data.copy_(torch.from_numpy(initializers[gemm.input[1]]))
    model.classifier.bias.data.copy_(torch.from_numpy(initializers[gemm.input[2]]))


def verify_model_matches_onnx(model: nn.Module, path: Path) -> float:
    """Verify that a trainable checkpoint and its deployed ONNX artifact are equivalent."""
    values = np.linspace(0.0, 1.0, 8 * 3 * INPUT_SIZE * INPUT_SIZE, dtype=np.float32)
    inputs = values.reshape(8, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        expected = model.cpu().eval()(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(None, {session.get_inputs()[0].name: inputs})[0])
    maximum_difference = float(np.max(np.abs(expected - actual)))
    if not np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1)) or not np.allclose(
        expected,
        actual,
        rtol=1e-4,
        atol=5e-5,
    ):
        raise ValueError(
            f"Checkpoint does not match ONNX artifact {path}; "
            f"maximum logit difference is {maximum_difference}"
        )
    return maximum_difference


def export_onnx(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model.cpu().eval(),
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
