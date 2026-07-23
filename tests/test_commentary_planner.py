from __future__ import annotations

import http.client
import io
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chess_scan.commentary_planner import (
    CommentaryCoach,
    OpenAICompatiblePlannerProvider,
    PlannerProviderError,
    ProviderResult,
    validate_position_coaching,
)
from chess_scan.config import Settings
from chess_scan.review import build_position_review
from chess_scan.schemas import PositionCoachingResponse, PositionReviewRequest


class _RespondingProviderHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {"choices": [{"message": {"content": '{"claim_ids":["claim-1"],"focus":"cause"}'}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _StalledProviderHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        time.sleep(2)

    def log_message(self, format: str, *args: object) -> None:
        pass


class FakeProvider:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output
        self.packets: list[dict[str, object]] = []

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        self.packets.append(evidence_packet)
        return ProviderResult(
            raw_output=self.raw_output,
            request={"provider": "fake"},
            input_tokens=120,
            output_tokens=18,
        )


class UnexpectedProvider:
    provider_name = "test-provider"
    model = "unexpected"

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        raise RuntimeError("sensitive provider detail")


class FailingProvider:
    provider_name = "test-provider"
    model = "test-model"

    def complete(self, evidence_packet: dict[str, object]) -> ProviderResult:
        raise PlannerProviderError("timeout")


def test_planner_accepts_only_selected_evidence_backed_claims() -> None:
    provider = FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    coach = CommentaryCoach(provider)

    run = coach.plan(_review())

    assert run.response.status == "accepted"
    assert run.response.headline == "Follow the cause and effect."
    assert run.response.lesson_ids == ["explanation-1"]
    assert run.record.accepted_claim_ids == ("claim-1",)
    assert run.record.raw_output == provider.raw_output
    assert run.record.input_tokens == 120
    assert run.record.output_tokens == 18
    assert run.record.error_code is None
    assert run.record.provider_called is True
    assert run.provider_completion is None
    assert run.response.run_id is not None
    assert run.record.request == {"provider": "fake"}

    packet = provider.packets[0]
    serialized = json.dumps(packet)
    assert packet["schema_version"] == "commentary-evidence-1"
    assert packet["allowed_claim_ids"] == ["claim-1"]
    assert [evidence["id"] for evidence in packet["evidence"]] == ["f1-e1"]
    assert packet["claims"][0]["evidence_ids"] == ["f1-e1"]
    assert "engine-best" not in serialized
    assert "review_id" not in serialized
    assert "feedback_id" not in serialized
    assert "image" not in serialized
    assert "article" not in serialized


def test_stored_coaching_rejects_tampered_lesson_ids() -> None:
    review = _review()
    provider = FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "concept"}))
    run = CommentaryCoach(provider).plan(review)
    payload = run.response.model_dump(mode="json")
    payload["lesson_ids"] = ["invented-lesson"]
    with pytest.raises(ValueError, match="unsupported lesson"):
        validate_position_coaching(review, payload)

    payload = run.response.model_dump(mode="json")
    payload["run_id"] = None
    with pytest.raises(ValueError, match="requires a planner run ID"):
        validate_position_coaching(review, payload)

    engine_only = review.explanation[0].model_copy(
        update={
            "id": "engine-only",
            "label": "Engine only",
            "evidence_ids": ["engine-best"],
        }
    )
    review_with_engine_copy = review.model_copy(
        update={"explanation": [*review.explanation, engine_only]}
    )
    payload = run.response.model_dump(mode="json")
    payload["lesson_ids"] = [engine_only.id]
    with pytest.raises(ValueError, match="unsupported lesson"):
        validate_position_coaching(review_with_engine_copy, payload)


def test_coaching_contract_rejects_invented_display_copy() -> None:
    with pytest.raises(ValueError, match="fixed focus headline"):
        PositionCoachingResponse(
            review_id="1" * 32,
            run_id="2" * 32,
            status="accepted",
            planner_version="commentary-planner-1",
            headline="The queen wins by force.",
            lesson_ids=["explanation-1"],
        )
    with pytest.raises(ValueError, match="fixed verified copy"):
        PositionCoachingResponse(
            review_id="1" * 32,
            run_id="2" * 32,
            status="fallback",
            planner_version="commentary-planner-1",
            headline="Verified review",
            lesson_ids=["explanation-1"],
            message="The queen wins by force.",
        )


def test_planner_rejects_unsupported_claims_and_falls_back_safely() -> None:
    raw_output = json.dumps({"claim_ids": ["invented"], "focus": "concept"})
    coach = CommentaryCoach(FakeProvider(raw_output))

    run = coach.plan(_review())

    assert run.response.status == "fallback"
    assert run.response.message is not None
    assert "unavailable" in run.response.message
    assert run.response.lesson_ids == ["explanation-1"]
    assert run.record.accepted_claim_ids == ()
    assert run.record.raw_output == raw_output
    assert run.record.error_code == "unsupported_claim"


def test_planner_rejects_duplicate_json_keys() -> None:
    raw_output = '{"claim_ids":["claim-1"],"claim_ids":["claim-1"],"focus":"cause"}'

    run = CommentaryCoach(FakeProvider(raw_output)).plan(_review())

    assert run.response.status == "fallback"
    assert run.record.error_code == "invalid_shape"


def test_planner_treats_bounded_json_integer_limit_as_invalid_output() -> None:
    raw_output = '{"claim_ids":["claim-1"],"focus":' + "9" * 5000 + "}"

    run = CommentaryCoach(FakeProvider(raw_output)).plan(_review())

    assert run.response.status == "fallback"
    assert run.record.error_code == "invalid_json"


def test_planner_provider_failure_never_exposes_raw_error_text() -> None:
    coach = CommentaryCoach(FailingProvider())

    run = coach.plan(_review())

    assert run.response.status == "fallback"
    assert run.response.message is not None
    assert "timeout" not in run.response.message
    assert run.record.raw_output is None
    assert run.record.error_code == "timeout"


def test_unpersistable_provider_content_is_escaped_in_fallback() -> None:
    run = CommentaryCoach(FakeProvider("\ud800")).plan(_review())

    assert run.response.status == "fallback"
    assert run.record.error_code == "invalid_unicode"
    assert run.record.raw_output == '"\\ud800"'
    assert run.record.raw_output.encode("utf-8")


def test_unexpected_provider_defect_is_not_cached_as_a_fallback() -> None:
    with pytest.raises(RuntimeError, match="sensitive provider detail"):
        CommentaryCoach(UnexpectedProvider()).plan(_review())


@pytest.mark.parametrize(
    "body",
    [
        b'{"choices":[{"message":{"content":"{}"}}],"usage":[]}',
        b'{"choices":[{"message":{"content":"\\ud800"}}]}',
        b"\xff",
    ],
)
def test_provider_rejects_malformed_envelopes_safely(
    body: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatiblePlannerProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=64,
        max_concurrent=1,
    )
    monkeypatch.setattr(provider, "_send", lambda request: body)

    with pytest.raises(PlannerProviderError) as error:
        provider.complete({"allowed_claim_ids": ["claim-1"]})

    assert error.value.code == "invalid_provider_response"
    assert error.value.raw_output is not None
    provider.close()


def test_provider_closes_http_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatiblePlannerProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=64,
        max_concurrent=1,
    )
    response = io.BytesIO(b"error")
    http_error = urllib.error.HTTPError(
        provider.endpoint,
        429,
        "rate limited",
        {},
        response,
    )

    def failed_response(request: object) -> bytes:
        raise http_error

    monkeypatch.setattr(provider, "_send", failed_response)
    with pytest.raises(PlannerProviderError) as error:
        provider.complete({"allowed_claim_ids": ["claim-1"]})

    assert error.value.code == "http_429"
    assert response.closed
    provider.close()


def test_provider_normalizes_malformed_http_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatiblePlannerProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=64,
        max_concurrent=1,
    )

    def malformed_response(request: object) -> bytes:
        raise http.client.BadStatusLine("broken")

    monkeypatch.setattr(provider, "_send", malformed_response)
    with pytest.raises(PlannerProviderError) as error:
        provider.complete({"allowed_claim_ids": ["claim-1"]})

    assert error.value.code == "network_error"
    assert error.value.provider_called is True
    provider.close()


def test_provider_records_exact_sanitized_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        b'{"choices":[{"message":{"content":"'
        b'{\\"claim_ids\\":[\\"claim-1\\"],\\"focus\\":\\"cause\\"}'
        b'"}}]}'
    )
    provider = OpenAICompatiblePlannerProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key="secret",
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=64,
        max_concurrent=1,
    )
    monkeypatch.setattr(provider, "_send", lambda request: body)

    result = provider.complete({"allowed_claim_ids": ["claim-1"]})

    assert result.request["prompt_version"] == "commentary-selection-1"
    assert result.request["endpoint"] == "https://provider.invalid/v1/chat/completions"
    assert result.request["json"]["temperature"] == 0
    assert result.request["json"]["max_tokens"] == 64
    assert result.request["json"]["response_format"] == {"type": "json_object"}
    assert result.request["headers"]["Authorization"] == "Bearer [configured]"
    assert "secret" not in json.dumps(result.request)
    provider.close()


def test_provider_discards_unpersistable_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = {
        "choices": [{"message": {"content": '{"claim_ids":["claim-1"],"focus":"cause"}'}}],
        "usage": {"prompt_tokens": 2**63, "completion_tokens": 12},
    }
    provider = OpenAICompatiblePlannerProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=1,
        max_output_tokens=64,
        max_concurrent=1,
    )
    monkeypatch.setattr(provider, "_send", lambda request: json.dumps(envelope).encode())

    result = provider.complete({"allowed_claim_ids": ["claim-1"]})

    assert result.input_tokens is None
    assert result.output_tokens == 12
    provider.close()


def test_provider_call_has_an_end_to_end_caller_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatiblePlannerProvider(
        endpoint="https://provider.invalid/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=0.05,
        max_output_tokens=64,
        max_concurrent=1,
    )

    def slow_response(request: object) -> bytes:
        time.sleep(0.2)
        return b"{}"

    monkeypatch.setattr(provider, "_send", slow_response)
    started = time.monotonic()
    with pytest.raises(PlannerProviderError) as error:
        provider.complete({"allowed_claim_ids": ["claim-1"]})

    assert error.value.code == "timeout"
    assert error.value.provider_called is True
    assert error.value.completion is not None
    assert time.monotonic() - started < 0.15
    provider.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@provider.example/v1/chat/completions",
        "https://provider.example/v1/chat/completions?api_key=secret",
    ],
)
def test_provider_rejects_credentials_in_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="credentials or a query"):
        OpenAICompatiblePlannerProvider(
            endpoint=endpoint,
            api_key=None,
            provider_name="test-provider",
            model="test-model",
            timeout_seconds=1,
            max_output_tokens=64,
            max_concurrent=1,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://provider.example:notaport/v1/chat/completions",
        "https://provider.example/v1/chat/completions\n",
        "https://provider.example/v1/chat completions",
        "https://provider.example/v1/café",
        "https://provider.example/v1/%not-hex",
    ],
)
def test_provider_rejects_malformed_endpoint(endpoint: str) -> None:
    with pytest.raises(
        ValueError,
        match="invalid port|control characters|credentials or a query",
    ):
        OpenAICompatiblePlannerProvider(
            endpoint=endpoint,
            api_key=None,
            provider_name="test-provider",
            model="test-model",
            timeout_seconds=1,
            max_output_tokens=64,
            max_concurrent=1,
        )


def test_provider_reuses_a_bounded_transport_worker() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RespondingProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = OpenAICompatiblePlannerProvider(
        endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=2,
        max_output_tokens=64,
        max_concurrent=2,
    )
    try:
        first = provider.complete({"allowed_claim_ids": ["claim-1"]})
        first_pid = provider._transport_workers[0].process.pid
        second = provider.complete({"allowed_claim_ids": ["claim-1"]})
        assert first.raw_output == second.raw_output
        assert len(provider._transport_workers) == 1
        assert provider._transport_workers[0].process.pid == first_pid
    finally:
        provider.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_provider_deadline_terminates_stalled_transport() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StalledProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = OpenAICompatiblePlannerProvider(
        endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        api_key=None,
        provider_name="test-provider",
        model="test-model",
        timeout_seconds=0.2,
        max_output_tokens=64,
        max_concurrent=1,
    )
    started = time.monotonic()
    try:
        with pytest.raises(PlannerProviderError, match="timeout") as raised:
            provider.complete({"allowed_claim_ids": ["claim-1"]})
        if raised.value.completion is not None:
            with pytest.raises(TimeoutError):
                raised.value.completion.result(timeout=1)
        assert time.monotonic() - started < 1.5
    finally:
        provider.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_provider_rejects_hostless_endpoint() -> None:
    with pytest.raises(ValueError, match="valid host"):
        OpenAICompatiblePlannerProvider(
            endpoint="https://:443/v1/chat/completions",
            api_key=None,
            provider_name="test-provider",
            model="test-model",
            timeout_seconds=1,
            max_output_tokens=64,
            max_concurrent=1,
        )


def test_provider_rejects_unsafe_api_key_without_echoing_it() -> None:
    secret = "secret\r\nInjected: value"
    with pytest.raises(ValueError) as error:
        OpenAICompatiblePlannerProvider(
            endpoint="https://provider.example/v1/chat/completions",
            api_key=secret,
            provider_name="test-provider",
            model="test-model",
            timeout_seconds=1,
            max_output_tokens=64,
            max_concurrent=1,
        )

    assert secret not in str(error.value)


def test_provider_rejects_plaintext_non_loopback_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OpenAICompatiblePlannerProvider(
            endpoint="http://provider.example/v1/chat/completions",
            api_key=None,
            provider_name="test-provider",
            model="test-model",
            timeout_seconds=1,
            max_output_tokens=64,
            max_concurrent=1,
        )


@pytest.mark.parametrize(
    ("provider_name", "model"),
    [(" ", "model"), ("provider", "m" * 161)],
)
def test_settings_validate_planner_identity_at_startup(
    provider_name: str,
    model: str,
) -> None:
    with pytest.raises(ValueError, match="provider name|model name"):
        Settings(
            data_dir=Path("data"),
            model_dir=Path("models"),
            web_dist=Path("web/dist"),
            max_upload_bytes=1024,
            max_image_dimension=1024,
            cors_origins=(),
            commentary_planner_provider=provider_name,
            commentary_planner_model=model,
        )


def test_candidate_free_review_skips_packet_and_provider_call() -> None:
    provider = FakeProvider('{"claim_ids":["claim-1"],"focus":"cause"}')
    review = _review().model_copy(update={"explanation": []})

    run = CommentaryCoach(provider).plan(review)

    assert run.response.status == "fallback"
    assert run.response.message == (
        "No deeper evidence-backed lesson is available for this position."
    )
    assert run.record.error_code == "no_claim_candidates"
    assert run.record.provider_called is False
    assert run.record.request == {}
    assert provider.packets == []


def test_disabled_planner_makes_no_generated_claims() -> None:
    coach = CommentaryCoach()

    run = coach.plan(_review())

    assert run.response.status == "disabled"
    assert run.response.lesson_ids == []
    assert not hasattr(run, "record")


def _review():
    request = PositionReviewRequest.model_validate(
        {
            "fen": "8/7k/2r5/8/8/8/4Q3/4K3 w - - 0 1",
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
    )
    return build_position_review(
        request,
        review_id="0123456789abcdef0123456789abcdef",
    )
