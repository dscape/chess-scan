"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    model_dir: Path
    web_dist: Path
    max_upload_bytes: int
    max_image_dimension: int
    cors_origins: tuple[str, ...]

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
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "source-temp").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "rectified").mkdir(parents=True, exist_ok=True)
