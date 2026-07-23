"""Bounded, evidence-constrained planning for optional deeper coaching."""

from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import multiprocessing
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Connection
from queue import Empty, LifoQueue
from threading import BoundedSemaphore, Event, Lock
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from chess_scan.commentary_contract import (
    COMMENTARY_MAX_LESSONS,
    COMMENTARY_MODEL_MAX_LENGTH,
    COMMENTARY_PROVIDER_MAX_LENGTH,
    CommentaryClaimRecord,
    CommentaryRunRecord,
    commentary_claim_candidates,
    normalize_commentary_identity,
)
from chess_scan.config import Settings
from chess_scan.schemas import (
    COMMENTARY_FALLBACK_MESSAGE,
    COMMENTARY_FOCUS_HEADLINES,
    COMMENTARY_NO_CLAIM_MESSAGE,
    COMMENTARY_REVIEW_HEADLINE,
    ENGINE_ONLY_EVIDENCE_KINDS,
    PositionCoachingResponse,
    PositionReviewResponse,
    ReviewAnnotation,
)

PLANNER_VERSION = "commentary-planner-1"
PROMPT_VERSION = "commentary-selection-1"
EVIDENCE_PACKET_VERSION = "commentary-evidence-1"
_MAX_PROVIDER_RESPONSE_BYTES = 64 * 1024
_MAX_PACKET_BYTES = 48 * 1024
_MAX_TOKEN_COUNT = 10_000_000
logger = logging.getLogger(__name__)
_SYSTEM_PROMPT = """You are a chess lesson selector, not a chess analyst.
Use only the supplied claim IDs. Do not add chess facts, moves, history, intent, or prose.
Return one JSON object with exactly two keys:
- claim_ids: one or two unique IDs from allowed_claim_ids, most useful first
- focus: one of cause, concept, comparison
Return JSON only."""


class PlannerProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        raw_output: str | None = None,
        request: dict[str, Any] | None = None,
        provider_called: bool = False,
        completion: Future[bytes] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.raw_output = raw_output
        self.request = request or {}
        self.provider_called = provider_called
        self.completion = completion


class _ProviderHttpError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"provider HTTP {status}")
        self.status = status


class PlannerOutputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProviderResult:
    raw_output: str
    request: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None


class PlannerProvider(Protocol):
    provider_name: str
    model: str

    def complete(self, evidence_packet: dict[str, Any]) -> ProviderResult: ...


@dataclass(frozen=True, slots=True)
class CommentaryPlannerRun:
    record: CommentaryRunRecord
    provider_completion: Future[bytes] | None

    @property
    def response(self) -> PositionCoachingResponse:
        return self.record.response


@dataclass(frozen=True, slots=True)
class DisabledCommentaryRun:
    response: PositionCoachingResponse


class CommentaryCoach:
    def __init__(self, provider: PlannerProvider | None = None) -> None:
        self.provider = provider

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name if self.provider is not None else "disabled"

    @property
    def model(self) -> str:
        return self.provider.model if self.provider is not None else "disabled"

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()

    @classmethod
    def from_settings(cls, settings: Settings) -> CommentaryCoach:
        if not settings.commentary_planner_enabled:
            return cls()
        if settings.commentary_planner_endpoint is None:
            raise ValueError("Commentary planner endpoint is required")
        if settings.commentary_planner_model is None:
            raise ValueError("Commentary planner model is required")
        return cls(
            OpenAICompatiblePlannerProvider(
                endpoint=settings.commentary_planner_endpoint,
                api_key=settings.commentary_planner_api_key,
                provider_name=settings.commentary_planner_provider,
                model=settings.commentary_planner_model,
                timeout_seconds=settings.commentary_planner_timeout_seconds,
                max_output_tokens=settings.commentary_planner_max_output_tokens,
                max_concurrent=settings.commentary_planner_max_concurrent,
            )
        )

    def plan(
        self,
        review: PositionReviewResponse,
        *,
        run_id: str | None = None,
    ) -> CommentaryPlannerRun | DisabledCommentaryRun:
        if review.review_id is None:
            raise ValueError("Stored review ID is required for coaching")
        if self.provider is None:
            return _disabled_run(review.review_id)

        run_id = run_id or uuid.uuid4().hex
        candidates = commentary_claim_candidates(review.evidence, review.explanation)
        request: dict[str, Any] = {}
        started = time.monotonic()
        provider_result: ProviderResult | None = None
        provider_error_output: str | None = None
        provider_called = False
        provider_completion: Future[bytes] | None = None
        try:
            if not candidates:
                raise PlannerOutputError("no_claim_candidates")
            packet = _evidence_packet(review, candidates)
            provider_result = self.provider.complete(packet)
            provider_called = True
            request = provider_result.request
            claim_ids, focus = _verified_selection(provider_result.raw_output, candidates)
            response = _accepted_response(
                review.review_id,
                run_id,
                candidates,
                claim_ids,
                focus,
            )
            error_code = None
        except PlannerProviderError as error:
            response = _fallback_response(review.review_id, run_id, candidates)
            claim_ids = ()
            error_code = error.code
            request = error.request
            provider_error_output = error.raw_output
            provider_called = error.provider_called
            provider_completion = error.completion
        except PlannerOutputError as error:
            response = _fallback_response(review.review_id, run_id, candidates)
            claim_ids = ()
            error_code = error.code
        except Exception:
            logger.exception("Unexpected commentary planner defect")
            raise
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        return CommentaryPlannerRun(
            record=CommentaryRunRecord(
                response=response,
                provider=self.provider_name,
                model=self.model,
                prompt_version=PROMPT_VERSION,
                request=request,
                raw_output=_persistable_text(
                    provider_result.raw_output
                    if provider_result is not None
                    else provider_error_output
                ),
                accepted_claim_ids=claim_ids,
                claim_candidates=candidates,
                latency_ms=latency_ms,
                input_tokens=provider_result.input_tokens if provider_result else None,
                output_tokens=provider_result.output_tokens if provider_result else None,
                error_code=error_code,
                provider_called=provider_called,
            ),
            provider_completion=provider_completion,
        )


def validate_position_coaching(
    review: PositionReviewResponse,
    payload: dict[str, Any],
) -> PositionCoachingResponse:
    response = PositionCoachingResponse.model_validate(payload)
    if response.review_id != review.review_id:
        raise ValueError("Stored coaching references a different review")
    allowed_ids = {lesson.id for lesson in eligible_commentary_lessons(review)}
    if not set(response.lesson_ids) <= allowed_ids:
        raise ValueError("Stored coaching contains an unsupported lesson")
    return response


class _ProviderMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    content: str


class _ProviderChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: _ProviderMessage


class _ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    choices: list[_ProviderChoice] = Field(default_factory=list)
    usage: _ProviderUsage | None = None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def _provider_transport_worker(connection: Connection) -> None:
    try:
        while True:
            command = connection.recv()
            if command is None:
                return
            endpoint, data, headers, timeout_seconds = command
            connection.send(
                _provider_request_outcome(
                    endpoint,
                    data,
                    headers,
                    timeout_seconds,
                )
            )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _provider_request_outcome(
    endpoint: str,
    data: bytes,
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[str, int | None, bytes]:
    try:
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirectHandler())
        with opener.open(request, timeout=timeout_seconds) as response:
            body = response.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
        return "ok", None, body
    except urllib.error.HTTPError as error:
        try:
            return "http", int(error.code), b""
        finally:
            error.close()
    except ValueError:
        return "invalid", None, b""
    except TimeoutError:
        return "timeout", None, b""
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        return "network", None, b""
    except Exception:
        return "internal", None, b""


@dataclass(slots=True)
class _TransportWorker:
    connection: Connection
    process: multiprocessing.Process
    retired: bool = False


def _stop_transport_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)
    process.close()


class OpenAICompatiblePlannerProvider:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str | None,
        provider_name: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        max_concurrent: int,
    ) -> None:
        if any(ord(character) < 32 or ord(character) == 127 for character in endpoint):
            raise ValueError("Commentary planner endpoint contains control characters")
        parsed = urlsplit(endpoint)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("Commentary planner endpoint has an invalid port") from error
        try:
            hostname = parsed.hostname
        except ValueError as error:
            raise ValueError("Commentary planner endpoint has an invalid host") from error
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Commentary planner endpoint must be an HTTP(S) URL")
        if not hostname or not _is_valid_hostname(hostname):
            raise ValueError("Commentary planner endpoint must have a valid host")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not _is_valid_request_path(parsed.path)
        ):
            raise ValueError("Commentary planner endpoint cannot contain credentials or a query")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("Commentary planner endpoint has an invalid port")
        if parsed.scheme == "http" and not _is_loopback_host(hostname):
            raise ValueError("Plaintext commentary endpoints must use a loopback host")
        if api_key is not None and (
            len(api_key) > 4096 or any(not 33 <= ord(character) <= 126 for character in api_key)
        ):
            raise ValueError("Commentary planner API key contains invalid header characters")
        self.endpoint = endpoint
        self.api_key = api_key
        self.provider_name = normalize_commentary_identity(
            provider_name,
            label="provider name",
            max_length=COMMENTARY_PROVIDER_MAX_LENGTH,
        )
        self.model = normalize_commentary_identity(
            model,
            label="model name",
            max_length=COMMENTARY_MODEL_MAX_LENGTH,
        )
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self._transport_slots = BoundedSemaphore(max_concurrent)
        self._closed = Event()
        self._max_transport_workers = max_concurrent
        self._available_transport_workers: LifoQueue[_TransportWorker] = LifoQueue()
        self._transport_workers: list[_TransportWorker] = []
        self._transport_worker_lock = Lock()
        self._transport = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="commentary-http",
        )

    def close(self) -> None:
        self._closed.set()
        with self._transport_worker_lock:
            workers = list(self._transport_workers)
        for worker in workers:
            self._retire_transport_worker(worker)
        self._transport.shutdown(wait=False, cancel_futures=True)

    def complete(self, evidence_packet: dict[str, Any]) -> ProviderResult:
        packet_json = json.dumps(evidence_packet, separators=(",", ":"), ensure_ascii=True)
        if len(packet_json.encode()) > _MAX_PACKET_BYTES:
            raise PlannerProviderError("packet_too_large")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": packet_json},
            ],
        }
        headers = {"Content-Type": "application/json", "User-Agent": "Chess-Scan/1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_snapshot = {
            "prompt_version": PROMPT_VERSION,
            "endpoint": self.endpoint,
            "method": "POST",
            "headers": {
                "Content-Type": headers["Content-Type"],
                "User-Agent": headers["User-Agent"],
                "Authorization": "Bearer [configured]" if self.api_key else None,
            },
            "json": payload,
        }
        try:
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, separators=(",", ":")).encode(),
                headers=headers,
                method="POST",
            )
        except ValueError as error:
            raise PlannerProviderError(
                "invalid_request",
                request=request_snapshot,
                provider_called=False,
            ) from error
        if not self._transport_slots.acquire(blocking=False):
            raise PlannerProviderError("provider_busy", request=request_snapshot)
        try:
            future = self._transport.submit(self._send, request)
        except RuntimeError as error:
            self._transport_slots.release()
            raise PlannerProviderError("provider_unavailable", request=request_snapshot) from error
        future.add_done_callback(self._release_transport_slot)
        try:
            body = future.result(timeout=self.timeout_seconds)
        except ValueError as error:
            raise PlannerProviderError(
                "invalid_request",
                request=request_snapshot,
                provider_called=True,
            ) from error
        except TimeoutError as error:
            raise PlannerProviderError(
                "timeout",
                request=request_snapshot,
                provider_called=True,
                completion=future,
            ) from error
        except (_ProviderHttpError, urllib.error.HTTPError) as error:
            status_code = error.status if isinstance(error, _ProviderHttpError) else error.code
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            raise PlannerProviderError(
                f"http_{status_code}",
                request=request_snapshot,
                provider_called=True,
            ) from error
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
        ) as error:
            raise PlannerProviderError(
                "network_error",
                request=request_snapshot,
                provider_called=True,
            ) from error
        if len(body) > _MAX_PROVIDER_RESPONSE_BYTES:
            raise PlannerProviderError(
                "response_too_large",
                request=request_snapshot,
                provider_called=True,
            )
        raw_body = body.decode("utf-8", errors="replace")
        try:
            decoded = json.loads(body)
            provider_response = _ProviderResponse.model_validate(decoded)
            choice = provider_response.choices[0]
            choice.message.content.encode("utf-8")
        except (
            UnicodeDecodeError,
            UnicodeEncodeError,
            ValueError,
            RecursionError,
            IndexError,
        ) as error:
            raise PlannerProviderError(
                "invalid_provider_response",
                raw_output=raw_body,
                request=request_snapshot,
                provider_called=True,
            ) from error
        usage = provider_response.usage
        return ProviderResult(
            raw_output=choice.message.content,
            request=request_snapshot,
            input_tokens=_optional_nonnegative_int(usage.prompt_tokens if usage else None),
            output_tokens=_optional_nonnegative_int(usage.completion_tokens if usage else None),
        )

    def _send(self, request: urllib.request.Request) -> bytes:
        deadline = time.monotonic() + self.timeout_seconds
        worker: _TransportWorker | None = None
        reusable = False
        try:
            worker = self._take_transport_worker(deadline)
            worker.connection.send(
                (
                    request.full_url,
                    request.data or b"",
                    dict(request.header_items()),
                    self.timeout_seconds,
                )
            )
            while True:
                if self._closed.is_set():
                    raise urllib.error.URLError("provider transport closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("provider deadline exceeded")
                if worker.connection.poll(min(remaining, 0.05)):
                    try:
                        kind, status, body = worker.connection.recv()
                    except EOFError as error:
                        raise RuntimeError("Provider transport exited unexpectedly") from error
                    reusable = kind not in {"internal", "timeout"}
                    break
        finally:
            if worker is not None:
                if reusable:
                    self._release_transport_worker(worker)
                else:
                    self._retire_transport_worker(worker)
        if kind == "ok":
            return body
        if kind == "http" and status is not None:
            raise _ProviderHttpError(status)
        if kind == "invalid":
            raise ValueError("invalid provider request")
        if kind == "timeout":
            raise TimeoutError("provider deadline exceeded")
        if kind == "network":
            raise urllib.error.URLError("provider transport failed")
        raise RuntimeError("Provider transport failed unexpectedly")

    def _take_transport_worker(self, deadline: float) -> _TransportWorker:
        while True:
            try:
                worker = self._available_transport_workers.get_nowait()
            except Empty:
                worker = None
            if worker is not None:
                if not worker.retired and worker.process.is_alive():
                    return worker
                self._retire_transport_worker(worker)
                continue

            with self._transport_worker_lock:
                if self._closed.is_set():
                    raise urllib.error.URLError("provider transport closed")
                if len(self._transport_workers) < self._max_transport_workers:
                    worker = self._start_transport_worker()
                    self._transport_workers.append(worker)
                    return worker

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("provider deadline exceeded")
            try:
                worker = self._available_transport_workers.get(timeout=min(remaining, 0.05))
            except Empty:
                continue
            if not worker.retired and worker.process.is_alive():
                return worker
            self._retire_transport_worker(worker)

    def _start_transport_worker(self) -> _TransportWorker:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_provider_transport_worker,
            args=(child,),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            process.close()
            raise
        child.close()
        return _TransportWorker(connection=parent, process=process)

    def _release_transport_worker(self, worker: _TransportWorker) -> None:
        if self._closed.is_set() or worker.retired or not worker.process.is_alive():
            self._retire_transport_worker(worker)
            return
        self._available_transport_workers.put(worker)

    def _retire_transport_worker(self, worker: _TransportWorker) -> None:
        with self._transport_worker_lock:
            if worker.retired:
                return
            worker.retired = True
            if worker in self._transport_workers:
                self._transport_workers.remove(worker)
        worker.connection.close()
        _stop_transport_process(worker.process)

    def _release_transport_slot(self, _future: Future[bytes]) -> None:
        self._transport_slots.release()


def eligible_commentary_lessons(
    review: PositionReviewResponse,
) -> tuple[ReviewAnnotation, ...]:
    return tuple(
        candidate.lesson
        for candidate in commentary_claim_candidates(review.evidence, review.explanation)
    )


def _evidence_packet(
    review: PositionReviewResponse,
    candidates: tuple[CommentaryClaimRecord, ...],
) -> dict[str, Any]:
    referenced_evidence_ids = {
        evidence_id for candidate in candidates for evidence_id in candidate.lesson.evidence_ids
    }
    evidence = [
        item
        for item in review.evidence
        if item.id in referenced_evidence_ids and item.kind not in ENGINE_ONLY_EVIDENCE_KINDS
    ]
    included_evidence_ids = {item.id for item in evidence}
    packet = {
        "schema_version": EVIDENCE_PACKET_VERSION,
        "fen": review.fen,
        "engine": review.engine,
        "evaluation": review.evaluation,
        "score": (
            review.score.model_dump(mode="json", exclude_none=True) if review.score else None
        ),
        "best_move": (
            review.best_move.model_dump(mode="json", exclude_none=True)
            if review.best_move
            else None
        ),
        "attempt": (
            {
                "move": review.attempt.move.model_dump(mode="json", exclude_none=True),
                "verdict": review.attempt.verdict,
                "equivalent": review.attempt.equivalent,
                "expected_score_loss": review.attempt.expected_score_loss,
            }
            if review.attempt
            else None
        ),
        "lines": [
            {
                "role": line.role,
                "score": line.score.model_dump(mode="json", exclude_none=True),
                "moves": [move.uci for move in line.moves],
            }
            for line in review.lines
        ],
        "evidence": [item.model_dump(mode="json", exclude_none=True) for item in evidence],
        "claims": [
            {
                "id": candidate.id,
                "label": candidate.lesson.label,
                "text": candidate.lesson.text,
                "scope": candidate.lesson.scope,
                "ply": candidate.lesson.ply,
                "evidence_ids": [
                    evidence_id
                    for evidence_id in candidate.lesson.evidence_ids
                    if evidence_id in included_evidence_ids
                ],
            }
            for candidate in candidates
        ],
        "allowed_claim_ids": [candidate.id for candidate in candidates],
    }
    encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=True).encode()
    if len(encoded) > _MAX_PACKET_BYTES:
        raise PlannerProviderError("packet_too_large")
    return packet


def _verified_selection(
    raw_output: str,
    candidates: tuple[CommentaryClaimRecord, ...],
) -> tuple[tuple[str, ...], str]:
    try:
        encoded = raw_output.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PlannerOutputError("invalid_unicode") from error
    if len(encoded) > 10 * 1024:
        raise PlannerOutputError("output_too_large")
    try:
        decoded = json.loads(raw_output, object_pairs_hook=_unique_json_object)
    except PlannerOutputError:
        raise
    except (ValueError, RecursionError) as error:
        raise PlannerOutputError("invalid_json") from error
    if not isinstance(decoded, dict) or set(decoded) != {"claim_ids", "focus"}:
        raise PlannerOutputError("invalid_shape")
    claim_ids = decoded["claim_ids"]
    focus = decoded["focus"]
    if (
        not isinstance(claim_ids, list)
        or not 1 <= len(claim_ids) <= COMMENTARY_MAX_LESSONS
        or any(not isinstance(claim_id, str) for claim_id in claim_ids)
        or len(set(claim_ids)) != len(claim_ids)
    ):
        raise PlannerOutputError("invalid_claim_ids")
    allowed = {candidate.id for candidate in candidates}
    if not set(claim_ids) <= allowed:
        raise PlannerOutputError("unsupported_claim")
    if not isinstance(focus, str) or focus not in COMMENTARY_FOCUS_HEADLINES:
        raise PlannerOutputError("invalid_focus")
    return tuple(claim_ids), focus


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise PlannerOutputError("invalid_shape")
    return dict(pairs)


def _accepted_response(
    review_id: str,
    run_id: str,
    candidates: tuple[CommentaryClaimRecord, ...],
    claim_ids: tuple[str, ...],
    focus: str,
) -> PositionCoachingResponse:
    by_id = {candidate.id: candidate for candidate in candidates}
    lessons = [by_id[claim_id].lesson for claim_id in claim_ids]
    headline = COMMENTARY_FOCUS_HEADLINES[focus]
    return PositionCoachingResponse(
        review_id=review_id,
        run_id=run_id,
        status="accepted",
        planner_version=PLANNER_VERSION,
        headline=headline,
        lesson_ids=[lesson.id for lesson in lessons],
    )


def _fallback_response(
    review_id: str,
    run_id: str,
    candidates: tuple[CommentaryClaimRecord, ...],
) -> PositionCoachingResponse:
    return PositionCoachingResponse(
        review_id=review_id,
        run_id=run_id,
        status="fallback",
        planner_version=PLANNER_VERSION,
        headline=COMMENTARY_REVIEW_HEADLINE,
        lesson_ids=[candidate.lesson.id for candidate in candidates[:1]],
        message=(COMMENTARY_FALLBACK_MESSAGE if candidates else COMMENTARY_NO_CLAIM_MESSAGE),
    )


def _disabled_run(review_id: str) -> DisabledCommentaryRun:
    response = PositionCoachingResponse(
        review_id=review_id,
        status="disabled",
        planner_version=PLANNER_VERSION,
        headline=COMMENTARY_REVIEW_HEADLINE,
        lesson_ids=[],
    )
    return DisabledCommentaryRun(response=response)


def _is_valid_request_path(path: str) -> bool:
    allowed = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$&'()*+,;=:@/%"
    )
    if any(character not in allowed for character in path):
        return False
    index = 0
    while index < len(path):
        if path[index] != "%":
            index += 1
            continue
        if index + 2 >= len(path) or any(
            character not in "0123456789abcdefABCDEF" for character in path[index + 1 : index + 3]
        ):
            return False
        index += 3
    return True


def _is_valid_hostname(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    try:
        ascii_hostname = hostname.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if not ascii_hostname or len(ascii_hostname) > 253:
        return False
    labels = ascii_hostname.split(".")
    return all(
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.rstrip(".").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _persistable_text(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(value, ensure_ascii=True)
    return value


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and 0 <= value <= _MAX_TOKEN_COUNT else None
