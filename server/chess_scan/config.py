"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from chess_scan.commentary_limits import (
    COMMENTARY_MODEL_MAX_LENGTH,
    COMMENTARY_PROVIDER_MAX_LENGTH,
    normalize_commentary_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    model_dir: Path
    web_dist: Path
    max_upload_bytes: int
    max_image_dimension: int
    cors_origins: tuple[str, ...]
    commentary_planner_enabled: bool = False
    commentary_planner_endpoint: str | None = None
    commentary_planner_provider: str = "openai-compatible"
    commentary_planner_model: str | None = None
    commentary_planner_api_key: str | None = field(default=None, repr=False)
    commentary_planner_timeout_seconds: float = 6.0
    commentary_planner_max_output_tokens: int = 180
    commentary_planner_max_concurrent: int = 2
    commentary_planner_max_runs_per_feedback: int = 8
    commentary_planner_max_runs_per_hour: int = 60

    def __post_init__(self) -> None:
        provider = normalize_commentary_identity(
            self.commentary_planner_provider,
            label="provider name",
            max_length=COMMENTARY_PROVIDER_MAX_LENGTH,
        )
        object.__setattr__(self, "commentary_planner_provider", provider)
        if self.commentary_planner_model is not None:
            model = normalize_commentary_identity(
                self.commentary_planner_model,
                label="model name",
                max_length=COMMENTARY_MODEL_MAX_LENGTH,
            )
            object.__setattr__(self, "commentary_planner_model", model)
        if self.commentary_planner_enabled and not (
            self.commentary_planner_endpoint and self.commentary_planner_model
        ):
            raise ValueError("Enabled commentary planning requires an endpoint and model")
        if not 0.25 <= self.commentary_planner_timeout_seconds <= 15:
            raise ValueError("Commentary planner timeout must be between 0.25 and 15 seconds")
        if not 32 <= self.commentary_planner_max_output_tokens <= 512:
            raise ValueError("Commentary planner output limit must be between 32 and 512 tokens")
        if not 1 <= self.commentary_planner_max_concurrent <= 8:
            raise ValueError("Commentary planner concurrency must be between 1 and 8")
        if not 1 <= self.commentary_planner_max_runs_per_feedback <= 50:
            raise ValueError("Commentary planner feedback budget must be between 1 and 50")
        if not 1 <= self.commentary_planner_max_runs_per_hour <= 1000:
            raise ValueError("Commentary planner hourly budget must be between 1 and 1000")

    @classmethod
    def load(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv("CHESS_SCAN_CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        )
        return cls(
            data_dir=_path_from_env("CHESS_SCAN_DATA_DIR", "data"),
            model_dir=_path_from_env("CHESS_SCAN_MODEL_DIR", "models"),
            web_dist=_path_from_env("CHESS_SCAN_WEB_DIST", "web/dist"),
            max_upload_bytes=int(os.getenv("CHESS_SCAN_MAX_UPLOAD_MB", "12")) * 1024 * 1024,
            max_image_dimension=int(os.getenv("CHESS_SCAN_MAX_IMAGE_DIMENSION", "2400")),
            cors_origins=origins,
            commentary_planner_enabled=_bool_from_env("CHESS_SCAN_COMMENTARY_PLANNER_ENABLED"),
            commentary_planner_endpoint=(
                os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_ENDPOINT") or None
            ),
            commentary_planner_provider=os.getenv(
                "CHESS_SCAN_COMMENTARY_PLANNER_PROVIDER",
                "openai-compatible",
            ),
            commentary_planner_model=(os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_MODEL") or None),
            commentary_planner_api_key=(os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_API_KEY") or None),
            commentary_planner_timeout_seconds=float(
                os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_TIMEOUT_SECONDS", "6")
            ),
            commentary_planner_max_output_tokens=int(
                os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_MAX_OUTPUT_TOKENS", "180")
            ),
            commentary_planner_max_concurrent=int(
                os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_MAX_CONCURRENT", "2")
            ),
            commentary_planner_max_runs_per_feedback=int(
                os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_MAX_RUNS_PER_FEEDBACK", "8")
            ),
            commentary_planner_max_runs_per_hour=int(
                os.getenv("CHESS_SCAN_COMMENTARY_PLANNER_MAX_RUNS_PER_HOUR", "60")
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "source-temp").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "rectified").mkdir(parents=True, exist_ok=True)
