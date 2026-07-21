"""FastAPI entry point for Chess Scan."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, TypeVar

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from chess_scan.bootstrap import initialize_database
from chess_scan.classifier import ModelManager
from chess_scan.config import Settings
from chess_scan.errors import ScanStateError, StoredDataIntegrityError
from chess_scan.review import build_position_review
from chess_scan.review_topics import REVIEW_TOPICS, TOPIC_REGISTRY_VERSION
from chess_scan.schemas import (
    BoardDetectionResponse,
    ConfirmRequest,
    ConfirmResponse,
    LearningStatusResponse,
    PositionReviewRequest,
    PositionReviewResponse,
    ReprocessRequest,
    ReviewPositionResponse,
    ReviewTopicRegistryResponse,
    ReviewTopicResponse,
    ScanResponse,
)
from chess_scan.service import ScannerService

logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database = initialize_database(resolved_settings)
        models = ModelManager(database)
        models.active()
        service = ScannerService(
            settings=resolved_settings,
            database=database,
            models=models,
        )
        application.state.settings = resolved_settings
        application.state.database = database
        application.state.service = service
        application.state.processing_slots = _processing_slots(2)
        cleanup_task = asyncio.create_task(_cleanup_sources_periodically(service))
        try:
            yield
        finally:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task

    application = FastAPI(
        title="Chess Scan",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=5,
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=resolved_settings.max_upload_bytes + 256 * 1024,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    _register_api_routes(application)
    _register_web_routes(application, resolved_settings.web_dist)
    return application


async def _cleanup_sources_periodically(service: ScannerService) -> None:
    while True:
        delay = 60 * 60
        try:
            result = await asyncio.to_thread(service.remove_stale_sources)
            if result.backlog:
                delay = 1
        except Exception:
            logger.exception("Unexpected failure while removing stale scan files")
        await asyncio.sleep(delay)


def _register_api_routes(application: FastAPI) -> None:
    @application.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        version = request.app.state.service.models.active().version
        return {"status": "ok", "model": version}

    @application.post("/api/detect-board", response_model=BoardDetectionResponse)
    async def detect_board(
        request: Request,
        image: Annotated[UploadFile, File(description="Transient camera preview frame")],
    ) -> BoardDetectionResponse:
        settings: Settings = request.app.state.settings
        payload = await image.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(413, "Camera frame is larger than the configured upload limit")
        if not payload:
            raise HTTPException(400, "Camera frame is empty")
        try:
            return await _run_processing(request, lambda: request.app.state.service.detect(payload))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/scans", response_model=ScanResponse)
    async def create_scan(
        request: Request,
        image: Annotated[UploadFile, File(description="One photographed chess diagram")],
    ) -> ScanResponse:
        settings: Settings = request.app.state.settings
        payload = await image.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(413, "Image is larger than the configured upload limit")
        if not payload:
            raise HTTPException(400, "Image is empty")
        try:
            return await _run_processing(request, lambda: request.app.state.service.scan(payload))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/scans/{scan_id}", response_model=ScanResponse)
    def get_scan(scan_id: str, request: Request) -> ScanResponse:
        try:
            return request.app.state.service.get_scan(scan_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ScanStateError as exc:
            raise HTTPException(410, str(exc)) from exc

    @application.post("/api/scans/{scan_id}/reprocess", response_model=ScanResponse)
    async def reprocess_scan(
        scan_id: str,
        body: ReprocessRequest,
        request: Request,
    ) -> ScanResponse:
        try:
            return await _run_processing(
                request,
                lambda: request.app.state.service.reprocess(scan_id, body.corners),
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/scans/{scan_id}/source", response_class=FileResponse)
    def source_image(scan_id: str, request: Request) -> FileResponse:
        try:
            path = request.app.state.service.source_path(scan_id)
        except (KeyError, ScanStateError) as exc:
            raise HTTPException(404, "Source image not found") from exc
        return _scan_file(path, "Source image not found")

    @application.get("/api/scans/{scan_id}/rectified", response_class=FileResponse)
    def rectified_image(scan_id: str, request: Request) -> FileResponse:
        try:
            path = request.app.state.service.rectified_path(scan_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _scan_file(path, "Rectified image not found")

    @application.post("/api/scans/{scan_id}/confirm", response_model=ConfirmResponse)
    def confirm_scan(
        scan_id: str,
        body: ConfirmRequest,
        request: Request,
    ) -> ConfirmResponse:
        try:
            return request.app.state.service.confirm(scan_id, body)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get("/api/reviews/{feedback_id}", response_model=ReviewPositionResponse)
    def review_position(feedback_id: str, request: Request) -> ReviewPositionResponse:
        try:
            return request.app.state.service.review_position(feedback_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except StoredDataIntegrityError as exc:
            logger.exception("Stored review data is invalid for %s", feedback_id)
            raise HTTPException(500, "Stored review data is invalid") from exc

    @application.post("/api/position-reviews", response_model=PositionReviewResponse)
    def position_review(body: PositionReviewRequest) -> PositionReviewResponse:
        try:
            return build_position_review(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/review-topics", response_model=ReviewTopicRegistryResponse)
    def review_topics() -> ReviewTopicRegistryResponse:
        return ReviewTopicRegistryResponse(
            version=TOPIC_REGISTRY_VERSION,
            topics=[
                ReviewTopicResponse(
                    id=topic.id,
                    name=topic.name,
                    level=topic.level,
                    course=topic.course,
                    capability=topic.capability,
                )
                for topic in REVIEW_TOPICS
            ],
        )

    @application.get("/api/learning/status", response_model=LearningStatusResponse)
    def learning_status(request: Request) -> LearningStatusResponse:
        return request.app.state.service.learning_status()


def _scan_file(path: Path, missing_message: str) -> FileResponse:
    try:
        stat_result = path.stat()
    except FileNotFoundError as exc:
        raise HTTPException(404, missing_message) from exc
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "Content-Encoding": "identity"},
        stat_result=stat_result,
    )


def _register_web_routes(application: FastAPI, web_dist: Path) -> None:
    assets = web_dist / "assets"
    if assets.exists():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = web_dist / "index.html"
    if not index.exists():
        return

    @application.get("/{path:path}", include_in_schema=False)
    def web_app(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "API route not found")
        candidate = (web_dist / path).resolve()
        if path and candidate.is_relative_to(web_dist) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def _processing_slots(capacity: int) -> asyncio.Queue[None]:
    slots: asyncio.Queue[None] = asyncio.Queue(maxsize=capacity)
    for _ in range(capacity):
        slots.put_nowait(None)
    return slots


async def _run_processing(request: Request, operation: Callable[[], _Result]) -> _Result:
    if request.scope.get("state", {}).get("processing_admitted"):
        return await asyncio.to_thread(operation)

    slots: asyncio.Queue[None] = request.app.state.processing_slots
    try:
        slots.get_nowait()
    except asyncio.QueueEmpty as exc:
        raise HTTPException(503, "Scanner is busy; try again shortly") from exc
    try:
        return await asyncio.to_thread(operation)
    finally:
        slots.put_nowait(None)


class RequestBodyLimitMiddleware:
    """Reject oversized bodies and admit uploads before multipart parsing."""

    _UPLOAD_PATHS = {"/api/detect-board", "/api/scans"}

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await _request_too_large(scope, receive, send)
            return

        slots: asyncio.Queue[None] | None = None
        if scope.get("path") in self._UPLOAD_PATHS:
            slots = scope["app"].state.processing_slots
            try:
                slots.get_nowait()
            except asyncio.QueueEmpty:
                await _request_busy(scope, receive, send)
                return
            scope.setdefault("state", {})["processing_admitted"] = True

        try:
            if content_length is not None:
                await self.app(scope, receive, send)
            else:
                await self._spool_unknown_length_body(scope, receive, send)
        finally:
            if slots is not None:
                slots.put_nowait(None)

    async def _spool_unknown_length_body(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        with tempfile.SpooledTemporaryFile(max_size=256 * 1024) as body:
            received = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                received += len(chunk)
                if received > self.max_body_bytes:
                    await _request_too_large(scope, receive, send)
                    return
                body.write(chunk)
                if not message.get("more_body", False):
                    break
            body.seek(0)

            async def replay() -> Message:
                chunk = body.read(64 * 1024)
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": bool(chunk),
                }

            await self.app(scope, replay, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                length = int(value)
            except ValueError:
                return None
            return length if length >= 0 else None
    return None


async def _request_too_large(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse({"detail": "Request body is larger than the configured limit"}, 413)
    await response(scope, receive, send)


async def _request_busy(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse({"detail": "Scanner is busy; try again shortly"}, 503)
    await response(scope, receive, send)


app = create_app()
