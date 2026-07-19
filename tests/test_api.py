from __future__ import annotations

import asyncio
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from chess_scan.bootstrap import initialize_database
from chess_scan.config import PROJECT_ROOT, Settings
from chess_scan.errors import ArtifactHashMismatchError
from chess_scan.main import RequestBodyLimitMiddleware, create_app
from chess_scan.model_artifact import sha256_file


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
        assert len(scan["prediction_revision"]) == 64

        restored_response = client.get(f"/api/scans/{scan['scan_id']}")
        assert restored_response.status_code == 200, restored_response.text
        restored = restored_response.json()
        assert restored["labels"] == scan["labels"]
        assert restored["corners"] == scan["corners"]
        assert restored["prediction_revision"] == scan["prediction_revision"]
        assert client.get(restored["source_image_url"]).status_code == 200
        assert client.get(restored["rectified_image_url"]).status_code == 200

        labels = scan["labels"]
        invalid_confirm = client.post(
            f"/api/scans/{scan['scan_id']}/confirm",
            json={
                "labels": labels,
                "orientation": "white",
                "side_to_move": "w",
                "castling": "KK",
                "en_passant": "z9",
            },
        )
        assert invalid_confirm.status_code == 422

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
        source_path = settings.data_dir / "source-temp" / f"{scan['scan_id']}.jpg"
        assert not source_path.exists()
        assert (settings.data_dir / "rectified" / f"{scan['scan_id']}.jpg").exists()
        assert client.get(f"/api/scans/{scan['scan_id']}").status_code == 410
        source_path.write_bytes(b"deletion failed")
        assert client.get(restored["source_image_url"]).status_code == 404

        status = client.get("/api/learning/status").json()
        assert status["confirmed_boards"] == 1
        assert status["training_boards"] == 1
        assert status["active_model"] == "chess-steps-v2"
        assert status["learning_state"] == "collecting"
        assert status["learning_progress"] == 1
        assert status["learning_target"] == 100
        assert status["candidate_model"] is None


def test_request_limit_and_api_fallback(tmp_path: Path) -> None:
    web_dist = tmp_path / "web"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<html>app</html>")
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=PROJECT_ROOT / "models",
        web_dist=web_dist,
        max_upload_bytes=1024,
        max_image_dimension=1200,
        cors_origins=(),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        oversized = client.post("/api/scans", content=b"x" * (300 * 1024))
        assert oversized.status_code == 413
        assert client.get("/api/not-a-route").status_code == 404
        assert client.get("/some/client/route").text == "<html>app</html>"


def test_unknown_length_body_is_bounded_and_busy_upload_is_rejected_before_reading() -> None:
    async def exercise() -> None:
        downstream_called = False

        async def downstream(scope, receive, send) -> None:
            nonlocal downstream_called
            downstream_called = True

        messages = iter(
            [
                {"type": "http.request", "body": b"123456", "more_body": True},
                {"type": "http.request", "body": b"789012", "more_body": False},
            ]
        )

        async def receive():
            return next(messages)

        sent: list[dict] = []

        async def send(message) -> None:
            sent.append(message)

        middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/other",
                "headers": [],
            },
            receive,
            send,
        )
        assert downstream_called is False
        assert sent[0]["status"] == 413

        receive_called = False

        async def busy_receive():
            nonlocal receive_called
            receive_called = True
            return {"type": "http.request", "body": b"x", "more_body": False}

        slots: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        app = SimpleNamespace(state=SimpleNamespace(processing_slots=slots))
        sent.clear()
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/scans",
                "headers": [(b"content-length", b"1")],
                "app": app,
                "state": {},
            },
            busy_receive,
            send,
        )
        assert receive_called is False
        assert sent[0]["status"] == 503

    asyncio.run(exercise())


def test_startup_rejects_corrupted_active_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    base_path = model_dir / "chess-steps-v2.onnx"
    base_metadata_path = model_dir / "chess-steps-v2.json"
    shutil.copyfile(PROJECT_ROOT / "models/chess-steps-v2.onnx", base_path)
    shutil.copyfile(PROJECT_ROOT / "models/chess-steps-v2.json", base_metadata_path)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=model_dir,
        web_dist=tmp_path / "web",
        max_upload_bytes=1024,
        max_image_dimension=1200,
        cors_origins=(),
    )
    database = initialize_database(settings)
    candidate_path = model_dir / "candidate.onnx"
    shutil.copyfile(base_path, candidate_path)
    database.register_candidate(
        version="candidate",
        artifact_path=candidate_path,
        metadata={
            "artifact_sha256": sha256_file(candidate_path),
            "eligible_for_promotion": True,
        },
    )
    database.promote_model("candidate")
    candidate_path.write_bytes(b"corrupted")

    with pytest.raises(ArtifactHashMismatchError):
        with TestClient(create_app(settings)):
            pass

    assert json.loads(base_metadata_path.read_text())["artifact_sha256"] == sha256_file(base_path)


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
