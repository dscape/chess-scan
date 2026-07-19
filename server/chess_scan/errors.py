"""Stable domain errors for scan state and model integrity failures."""

from __future__ import annotations


class ScanStateError(ValueError):
    """A scan operation is incompatible with its persisted lifecycle state."""


class ScanAlreadyConfirmedError(ScanStateError):
    pass


class ScanExpiredError(ScanStateError):
    pass


class ModelArtifactError(ValueError):
    """A registered model artifact does not match its integrity metadata."""


class MissingArtifactHashError(ModelArtifactError):
    pass


class ArtifactHashMismatchError(ModelArtifactError):
    pass
