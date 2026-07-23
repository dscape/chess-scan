"""Stable domain errors for scan state and model integrity failures."""

from __future__ import annotations


class ScanStateError(ValueError):
    """A scan operation is incompatible with its persisted lifecycle state."""


class ScanAlreadyConfirmedError(ScanStateError):
    pass


class ScanExpiredError(ScanStateError):
    pass


class PositionReviewNotFoundError(LookupError):
    """A requested immutable position review does not exist."""


class StoredDataIntegrityError(RuntimeError):
    """Persisted immutable data cannot be interpreted safely."""


class PositionReviewFeedbackConflictError(ValueError):
    """Feedback does not match the coaching presentation being rated."""


class CommentaryBusyError(RuntimeError):
    """A coaching run is already active for this review."""


class CommentaryBudgetExceededError(RuntimeError):
    """A confirmed position has exhausted its external coaching budget."""


class ModelArtifactError(ValueError):
    """A registered model artifact does not match its integrity metadata."""


class MissingArtifactHashError(ModelArtifactError):
    pass


class ArtifactHashMismatchError(ModelArtifactError):
    pass
