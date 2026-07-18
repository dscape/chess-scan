"""FastAPI entry point for Chess Scan."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chess_scan.bootstrap import initialize_database
from chess_scan.classifier import ModelManager
from chess_scan.config import Settings
from chess_scan.schemas import (
    BoardDetectionResponse,
    ConfirmRequest,
    ConfirmResponse,
    LearningStatusResponse,
    ReprocessRequest,
    ScanResponse,
)
from chess_scan.service import ScannerService


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database = initialize_database(resolved_settings)
        service = ScannerService(
            settings=resolved_settings,
            database=database,
            models=ModelManager(database),
        )
        service.remove_stale_sources()
        cleanup_task = asyncio.create_task(_cleanup_sources_periodically(service))
        application.state.settings = resolved_settings
        application.state.database = database
        application.state.service = service
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
        await asyncio.sleep(60 * 60)
        service.remove_stale_sources()


def _register_api_routes(application: FastAPI) -> None:
    @application.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        status = request.app.state.database.learning_status()
        return {"status": "ok", "model": str(status["active_model"])}

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
            return request.app.state.service.detect(payload)
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
            return request.app.state.service.scan(payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/scans/{scan_id}/reprocess", response_model=ScanResponse)
    def reprocess_scan(
        scan_id: str,
        body: ReprocessRequest,
        request: Request,
    ) -> ScanResponse:
        try:
            return request.app.state.service.reprocess(scan_id, body.corners)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/api/scans/{scan_id}/rectified", response_class=FileResponse)
    def rectified_image(scan_id: str, request: Request) -> FileResponse:
        try:
            path = request.app.state.service.rectified_path(scan_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if not path.exists():
            raise HTTPException(404, "Rectified image not found")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

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

    @application.get("/api/learning/status", response_model=LearningStatusResponse)
    def learning_status(request: Request) -> LearningStatusResponse:
        return request.app.state.service.learning_status()


def _register_web_routes(application: FastAPI, web_dist: Path) -> None:
    assets = web_dist / "assets"
    if assets.exists():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = web_dist / "index.html"
    if not index.exists():
        return

    @application.get("/{path:path}", include_in_schema=False)
    def web_app(path: str) -> FileResponse:
        candidate = (web_dist / path).resolve()
        if path and candidate.is_relative_to(web_dist) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
