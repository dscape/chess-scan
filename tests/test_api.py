from __future__ import annotations

import asyncio
import io
import json
import shutil
import sqlite3
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from chess_scan.bootstrap import initialize_database
from chess_scan.commentary_planner import (
    CommentaryCoach,
    PlannerProviderError,
    ProviderResult,
)
from chess_scan.config import PROJECT_ROOT, Settings
from chess_scan.errors import ArtifactHashMismatchError
from chess_scan.main import RequestBodyLimitMiddleware, _run_in_slot, create_app
from chess_scan.model_artifact import sha256_file


class _DefectivePlannerProvider:
    provider_name = "test-provider"
    model = "test-model"

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        raise RuntimeError("planner implementation defect")


class _BusyPlannerProvider:
    provider_name = "test-provider"
    model = "test-model"

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        raise PlannerProviderError("provider_busy", provider_called=False)


class _DeferredPlannerProvider:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self) -> None:
        self.completion: Future[bytes] = Future()

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        raise PlannerProviderError(
            "timeout",
            request={"provider": "api-test"},
            provider_called=True,
            completion=self.completion,
        )


class _RejectingExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        raise RuntimeError("executor is closed")


class _ApiPlannerProvider:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        self.calls += 1
        return ProviderResult(
            raw_output='{"claim_ids":["claim-1"],"focus":"concept"}',
            request={"provider": "api-test"},
            input_tokens=100,
            output_tokens=12,
        )


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

        predicted_labels = scan["labels"]
        invalid_confirm = client.post(
            f"/api/scans/{scan['scan_id']}/confirm",
            json={
                "labels": predicted_labels,
                "orientation": "white",
                "side_to_move": "w",
                "castling": "KK",
                "en_passant": "z9",
            },
        )
        assert invalid_confirm.status_code == 422

        invalid_position = client.post(
            f"/api/scans/{scan['scan_id']}/confirm",
            json={
                "labels": [0] * 64,
                "orientation": "white",
                "side_to_move": "w",
            },
        )
        assert invalid_position.status_code == 409
        assert "Expected exactly one white king" in invalid_position.json()["detail"]

        labels = [0] * 64
        labels[15] = 12  # black king h7
        labels[18] = 10  # black rook c6
        labels[52] = 5  # white queen e2
        labels[60] = 6  # white king e1
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
        confirmation = confirm_response.json()
        assert confirmation["warnings"] == []
        assert confirmation["coaching_available"] is False
        assert confirmation["changed_squares"] == sum(
            predicted != corrected
            for predicted, corrected in zip(predicted_labels, labels, strict=True)
        )

        review_position = client.get(f"/api/reviews/{confirmation['feedback_id']}")
        assert review_position.status_code == 200, review_position.text
        assert review_position.json()["full_fen"] == confirmation["full_fen"]
        assert review_position.json()["orientation"] == "white"
        assert review_position.json()["coaching_available"] is False
        assert client.get("/api/reviews/missing").status_code == 404

        review_request = {
            "fen": "8/7k/2r5/8/8/8/4Q3/4K3 w - - 0 1",
            "feedback_id": confirmation["feedback_id"],
            "analysis": {
                "score_pov": "side_to_move",
                "lines": [
                    {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": 520},
                        "wdl": [930, 69, 1],
                        "pv": ["e2e4", "h7g8", "e4c6"],
                        "stable": True,
                    }
                ],
            },
        }
        review_response = client.post("/api/position-reviews", json=review_request)
        assert review_response.status_code == 200, review_response.text
        position_review = review_response.json()
        assert position_review["best_move"] == {"uci": "e2e4", "san": "Qe4+"}
        assert position_review["topic"] == {"id": "double-attack", "name": "Double attack"}
        assert position_review["engine"] == "Stockfish 18 lite"
        assert position_review["score"]["value"] == 520
        assert position_review["hint"]["markers"] == [
            {"square": "c6", "role": "focus"},
            {"square": "h7", "role": "focus"},
        ]
        assert position_review["schema_version"] == "position-analysis-5"
        assert position_review["explanation"][0]["arrows"][0] == {
            "from_square": "e2",
            "to_square": "e4",
            "role": "engine",
        }
        assert position_review["explanation"][0]["badge"] == {
            "kind": "fork",
            "square": "e4",
            "role": "engine",
            "arrow_index": 1,
        }
        review_id = position_review["review_id"]
        assert len(review_id) == 32
        stored_review = client.get(f"/api/position-reviews/{review_id}")
        assert stored_review.status_code == 200
        assert stored_review.json() == position_review
        coaching = client.post(f"/api/position-reviews/{review_id}/coaching")
        assert coaching.status_code == 200
        assert coaching.json()["status"] == "disabled"
        assert coaching.json()["lesson_ids"] == []
        assert client.post("/api/position-reviews/missing/coaching").status_code == 404

        preflight_slots = client.app.state.commentary_preflight_slots
        held_preflight_slots = [
            preflight_slots.get_nowait() for _ in range(preflight_slots.qsize())
        ]
        try:
            busy_preflight = client.post(
                f"/api/position-reviews/{review_id}/coaching",
                headers={"Origin": "http://localhost:5173"},
            )
            assert busy_preflight.status_code == 503
            assert busy_preflight.headers["retry-after"] == "1"
            assert busy_preflight.headers["access-control-expose-headers"] == "Retry-After"
            assert busy_preflight.json()["detail"] == "Coaching is busy; try again shortly"
        finally:
            for slot in held_preflight_slots:
                preflight_slots.put_nowait(slot)

        planner_provider = _ApiPlannerProvider()
        client.app.state.service.commentary_coach = CommentaryCoach(planner_provider)

        commentary_slots = client.app.state.commentary_slots
        held_slots = [commentary_slots.get_nowait() for _ in range(commentary_slots.qsize())]
        try:
            busy_coaching = client.post(f"/api/position-reviews/{review_id}/coaching")
            assert busy_coaching.status_code == 503
            assert busy_coaching.headers["retry-after"] == "1"
            assert busy_coaching.json()["detail"] == "Coaching is busy; try again shortly"
        finally:
            for slot in held_slots:
                commentary_slots.put_nowait(slot)

        service = client.app.state.service
        with patch.object(
            service,
            "get_position_review",
            wraps=service.get_position_review,
        ) as get_position_review:
            accepted_coaching = client.post(f"/api/position-reviews/{review_id}/coaching")
        assert get_position_review.call_count == 1
        assert accepted_coaching.status_code == 200
        assert accepted_coaching.json()["status"] == "accepted"
        assert len(accepted_coaching.json()["run_id"]) == 32
        assert accepted_coaching.json()["lesson_ids"] == ["explanation-1"]
        assert (
            client.post(f"/api/position-reviews/{review_id}/coaching").json()
            == accepted_coaching.json()
        )
        assert planner_provider.calls == 1

        deferred_review = client.post("/api/position-reviews", json=review_request).json()
        deferred_provider = _DeferredPlannerProvider()
        client.app.state.service.commentary_coach = CommentaryCoach(deferred_provider)
        deferred_preflight = service.preflight_position_coaching(deferred_review["review_id"])
        with pytest.raises(ValueError, match="different review"):
            service.create_position_coaching(
                review_id,
                preflight=deferred_preflight,
            )
        deferred_coaching = client.post(
            f"/api/position-reviews/{deferred_review['review_id']}/coaching"
        )
        assert deferred_coaching.status_code == 200
        assert deferred_coaching.json()["status"] == "fallback"
        with sqlite3.connect(settings.data_dir / "chess-scan.sqlite3") as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM commentary_planner_reservations WHERE review_id = ?",
                    (deferred_review["review_id"],),
                ).fetchone()[0]
                == 1
            )
        deferred_provider.completion.set_result(b"")
        with sqlite3.connect(settings.data_dir / "chess-scan.sqlite3") as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM commentary_planner_reservations WHERE review_id = ?",
                    (deferred_review["review_id"],),
                ).fetchone()[0]
                == 0
            )

        busy_review = client.post("/api/position-reviews", json=review_request).json()
        service.commentary_coach = CommentaryCoach(_DefectivePlannerProvider())
        defective_preflight = service.preflight_position_coaching(busy_review["review_id"])
        with pytest.raises(RuntimeError, match="planner implementation defect"):
            service.create_position_coaching(
                busy_review["review_id"],
                preflight=defective_preflight,
            )
        with sqlite3.connect(settings.data_dir / "chess-scan.sqlite3") as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM commentary_planner_reservations WHERE review_id = ?",
                    (busy_review["review_id"],),
                ).fetchone()[0]
                == 0
            )
        service.commentary_coach = CommentaryCoach(_BusyPlannerProvider())
        busy_fallback = client.post(f"/api/position-reviews/{busy_review['review_id']}/coaching")
        assert busy_fallback.status_code == 200
        assert busy_fallback.json()["status"] == "fallback"
        with sqlite3.connect(settings.data_dir / "chess-scan.sqlite3") as connection:
            assert (
                connection.execute(
                    "SELECT provider_called FROM commentary_planner_runs WHERE review_id = ?",
                    (busy_review["review_id"],),
                ).fetchone()[0]
                == 0
            )

        rating = client.post(
            f"/api/position-reviews/{review_id}/feedback",
            json={
                "rating": "unhelpful",
                "reason": "irrelevant_topic",
                "coaching_status": "accepted",
                "commentary_run_id": accepted_coaching.json()["run_id"],
            },
        )
        assert rating.status_code == 200
        assert len(rating.json()["feedback_id"]) == 32
        invalid_rating = client.post(
            f"/api/position-reviews/{review_id}/feedback",
            json={"rating": "helpful", "reason": "incorrect_chess"},
        )
        assert invalid_rating.status_code == 422
        missing_snapshot = client.post(
            f"/api/position-reviews/{review_id}/feedback",
            json={"rating": "helpful", "reason": "correct"},
        )
        assert missing_snapshot.status_code == 422

        with sqlite3.connect(settings.data_dir / "chess-scan.sqlite3") as connection:
            stored_coaching = json.loads(
                connection.execute(
                    "SELECT response_json FROM commentary_planner_runs WHERE review_id = ?",
                    (review_id,),
                ).fetchone()[0]
            )
            stored_coaching["run_id"] = "0" * 32
            connection.execute(
                "UPDATE commentary_planner_runs SET response_json = ? WHERE review_id = ?",
                (json.dumps(stored_coaching), review_id),
            )
            connection.commit()
        corrupt_coaching = client.post(f"/api/position-reviews/{review_id}/coaching")
        assert corrupt_coaching.status_code == 500
        assert corrupt_coaching.json()["detail"] == "Stored position review is invalid"

        with sqlite3.connect(settings.data_dir / "chess-scan.sqlite3") as connection:
            connection.execute(
                "UPDATE feedback_events SET final_fen = ? WHERE id = ?",
                ("8/8/8/8/8/8/8/8 w - - 0 1", confirmation["feedback_id"]),
            )
            connection.commit()
        invalid_stored_review = client.get(f"/api/reviews/{confirmation['feedback_id']}")
        assert invalid_stored_review.status_code == 500
        assert invalid_stored_review.json()["detail"] == "Stored review data is invalid"

        source_path = settings.data_dir / "source-temp" / f"{scan['scan_id']}.jpg"
        assert not source_path.exists()
        assert (settings.data_dir / "rectified" / f"{scan['scan_id']}.jpg").exists()
        assert client.get(f"/api/scans/{scan['scan_id']}").status_code == 410
        source_path.write_bytes(b"deletion failed")
        assert client.get(restored["source_image_url"]).status_code == 404

        status = client.get("/api/learning/status").json()
        assert status["confirmed_boards"] == 1
        assert status["training_boards"] == 1
        assert status["active_model"] == "chess-steps-v5"
        assert status["learning_state"] == "collecting"
        assert status["learning_progress"] == 1
        assert status["learning_target"] == 100
        assert status["candidate_model"] is None


def test_scan_reuses_live_corners_and_preserves_manual_recovery(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=PROJECT_ROOT / "models",
        web_dist=tmp_path / "missing-web",
        max_upload_bytes=2 * 1024 * 1024,
        max_image_dimension=1200,
        cors_origins=(),
    )
    app = create_app(settings)
    board_corners = [[0, 0], [639, 0], [639, 639], [0, 639]]
    with TestClient(app) as client:
        live_response = client.post(
            "/api/scans",
            data={
                "corners": json.dumps(board_corners),
                "detection_method": "checkerboard",
            },
            files={"image": ("board.png", _board_png(), "image/png")},
        )
        assert live_response.status_code == 200, live_response.text
        live_scan = live_response.json()
        assert live_scan["detection_method"] == "checkerboard"
        assert live_scan["corners"] == board_corners

        invalid_live_response = client.post(
            "/api/scans",
            data={
                "corners": json.dumps(board_corners),
                "detection_method": "manual_adjustment_needed",
            },
            files={"image": ("board.png", _board_png(), "image/png")},
        )
        assert invalid_live_response.status_code == 422

        outside_live_response = client.post(
            "/api/scans",
            data={
                "corners": json.dumps([[-1, 0], [639, 0], [639, 639], [0, 639]]),
                "detection_method": "checkerboard",
            },
            files={"image": ("board.png", _board_png(), "image/png")},
        )
        assert outside_live_response.status_code == 422

        uncertain_live_response = client.post(
            "/api/scans",
            data={
                "corners": json.dumps([[20, 0], [639, 0], [639, 639], [20, 639]]),
                "detection_method": "checkerboard",
            },
            files={"image": ("board.png", _board_png(), "image/png")},
        )
        assert uncertain_live_response.status_code == 200, uncertain_live_response.text
        assert uncertain_live_response.json()["detection_method"] == "manual_adjustment_needed"

        manual_response = client.post(
            "/api/scans",
            files={"image": ("blank.png", _solid_png(), "image/png")},
        )
        assert manual_response.status_code == 200, manual_response.text
        manual_scan = manual_response.json()
        assert manual_scan["detection_method"] == "manual_adjustment_needed"
        assert len(manual_scan["corners"]) == 4

        reprocessed = client.post(
            f"/api/scans/{manual_scan['scan_id']}/reprocess",
            json={"corners": board_corners},
        )
        assert reprocessed.status_code == 200, reprocessed.text
        assert reprocessed.json()["detection_method"] == "manual"

        outside_reprocess = client.post(
            f"/api/scans/{manual_scan['scan_id']}/reprocess",
            json={"corners": [[-1, 0], [639, 0], [639, 639], [0, 639]]},
        )
        assert outside_reprocess.status_code == 422


def test_bounded_slot_is_restored_when_executor_rejects_submission() -> None:
    async def exercise() -> None:
        slots: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        slots.put_nowait(None)
        with pytest.raises(RuntimeError, match="executor is closed"):
            await _run_in_slot(
                slots,
                lambda: None,
                busy_message="busy",
                executor=_RejectingExecutor(),
            )
        assert slots.qsize() == 1

    asyncio.run(exercise())


def test_bounded_slot_is_restored_as_soon_as_work_finishes() -> None:
    async def exercise(executor: ThreadPoolExecutor) -> None:
        slots: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        slots.put_nowait(None)
        result = await _run_in_slot(
            slots,
            lambda: "done",
            busy_message="busy",
            executor=executor,
        )
        assert result == "done"
        assert slots.qsize() == 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(exercise(executor))


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
        assert client.get("/reviews/0123456789abcdef0123456789abcdef").text == "<html>app</html>"


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
    base_path = model_dir / "chess-steps-v5.onnx"
    base_metadata_path = model_dir / "chess-steps-v5.json"
    shutil.copyfile(PROJECT_ROOT / "models/chess-steps-v5.onnx", base_path)
    shutil.copyfile(PROJECT_ROOT / "models/chess-steps-v5.json", base_metadata_path)
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
    return _png_bytes(board)


def _solid_png() -> bytes:
    return _png_bytes(np.full((640, 640, 3), 230, dtype=np.uint8))


def _png_bytes(pixels: np.ndarray) -> bytes:
    image = Image.fromarray(pixels, "RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
