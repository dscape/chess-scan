from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from chess_scan.config import PROJECT_ROOT, Settings
from chess_scan.main import create_app


def test_scan_confirm_and_learning_status(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=PROJECT_ROOT / "models",
        web_dist=tmp_path / "missing-web",
        max_upload_bytes=2 * 1024 * 1024,
        max_image_dimension=1200,
        cors_origins=("http://localhost:5173",),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        detection_response = client.post(
            "/api/detect-board",
            files={"image": ("board.png", _board_png(), "image/png")},
        )
        assert detection_response.status_code == 200, detection_response.text
        detection = detection_response.json()
        assert detection["found"] is True
        assert len(detection["corners"]) == 4
        assert len(detection["grid_points"]) == 81

        scan_response = client.post(
            "/api/scans",
            files={"image": ("board.png", _board_png(), "image/png")},
        )
        assert scan_response.status_code == 200, scan_response.text
        scan = scan_response.json()
        assert len(scan["labels"]) == 64
        assert len(scan["probabilities"]) == 64

        labels = scan["labels"]
        confirm_response = client.post(
            f"/api/scans/{scan['scan_id']}/confirm",
            json={
                "labels": labels,
                "orientation": "white",
                "side_to_move": "w",
                "consent_training": True,
                "client_session_id": "test-client",
            },
        )
        assert confirm_response.status_code == 200, confirm_response.text
        assert confirm_response.json()["changed_squares"] == 0
        assert not (settings.data_dir / "source-temp" / f"{scan['scan_id']}.jpg").exists()
        assert (settings.data_dir / "rectified" / f"{scan['scan_id']}.jpg").exists()

        status = client.get("/api/learning/status").json()
        assert status["confirmed_boards"] == 1
        assert status["training_boards"] == 1
        assert status["active_model"] == "chess-steps-v2"


def _board_png() -> bytes:
    board = np.zeros((640, 640, 3), dtype=np.uint8)
    square = 80
    for row in range(8):
        for col in range(8):
            value = 235 if (row + col) % 2 == 0 else 110
            board[row * square : (row + 1) * square, col * square : (col + 1) * square] = value
    image = Image.fromarray(board, "RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
