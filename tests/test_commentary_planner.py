from __future__ import annotations

import http.client
import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chess_scan.commentary_planner import (
    CommentaryCoach,
    OpenAICompatiblePlannerProvider,
    PlannerProviderError,
    ProviderResult,
    _provider_request_outcome,
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
    assert run.response.headline == "The point of the double attack"
    assert run.response.lesson_ids == ["explanation-1"]
    assert [section.kind for section in run.response.sections] == [
        "diagnosis",
        "continuation",
        "practice",
    ]
    assert [
        segment.move.san
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "move"
    ] == ["Qe4+", "Qe4+", "Kg8", "Qxc6", "Qe4+"]
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
    assert packet["schema_version"] == "commentary-evidence-2"
    assert packet["allowed_claim_ids"] == ["claim-1"]
    assert packet["required_primary_claim_id"] is None
    assert [evidence["id"] for evidence in packet["evidence"]] == ["f1-e1"]
    assert packet["claims"][0]["evidence_ids"] == ["f1-e1"]
    assert "engine-best" not in serialized
    assert "review_id" not in serialized
    assert "feedback_id" not in serialized
    assert "image" not in serialized
    assert "article" not in serialized


def test_focus_changes_deterministic_section_emphasis() -> None:
    cause = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(_review())
    concept = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "concept"}))
    ).plan(_review())

    assert [section.kind for section in cause.response.sections] == [
        "diagnosis",
        "continuation",
        "practice",
    ]
    assert [section.kind for section in concept.response.sections] == [
        "diagnosis",
        "practice",
        "continuation",
    ]


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


def test_coaching_narrative_explains_cause_line_option_and_habit() -> None:
    review = _attempt_review()
    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(review)

    sections = run.response.sections
    assert [section.title for section in sections] == [
        "Why the move loses material",
        "The line to calculate",
        "A better first move",
        "A captures-first check",
    ]
    diagnosis = sections[0]
    assert [segment.move.san for segment in diagnosis.segments if segment.type == "move"] == [
        "Nd4",
        "Nxd4",
        "Kd1",
        "Nd4",
    ]
    diagnosis_text = "".join(
        segment.text for segment in diagnosis.segments if segment.type == "text"
    )
    assert "take the knight on d4" in diagnosis_text
    assert "−3.5" in diagnosis_text
    assert {"engine-best", "engine-attempt"} <= set(diagnosis.evidence_ids)
    assert sections[-1].kind == "practice"

    payload = run.response.model_dump(mode="json")
    payload["sections"][0]["segments"][0]["move"] = {
        "uci": "e1d1",
        "san": "Kd1",
    }
    with pytest.raises(ValueError, match="contradicts its checked line"):
        validate_position_coaching(review, payload)

    payload = run.response.model_dump(mode="json")
    payload["sections"][0]["segments"][1]["text"] = "The player failed to calculate."
    with pytest.raises(ValueError, match="narrative contradicts"):
        validate_position_coaching(review, payload)

    payload = run.response.model_dump(mode="json")
    payload["sections"][0]["segments"][1]["text"] = "Try Qh5 instead."
    with pytest.raises(ValueError, match="unstructured SAN"):
        validate_position_coaching(review, payload)


def test_lesson_copy_turns_checked_san_into_move_segments() -> None:
    review = _review()
    lesson = review.explanation[0].model_copy(
        update={"text": "Qe4+ creates the double attack before Qxc6 wins the rook."}
    )
    review = review.model_copy(update={"explanation": [lesson]})

    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "concept"}))
    ).plan(review)

    move_sans = [
        segment.move.san
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "move"
    ]
    text = " ".join(
        segment.text
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "text"
    )
    assert "Qxc6" in move_sans
    assert "Qxc6" not in text
    validate_position_coaching(review, run.response.model_dump(mode="json"))

    evidence_id = lesson.evidence_ids[0]
    counterfactual_evidence = next(
        evidence for evidence in review.evidence if evidence.id == evidence_id
    ).model_copy(update={"proof": "counterfactual"})
    counterfactual_review = review.model_copy(
        update={
            "evidence": [
                counterfactual_evidence if evidence.id == evidence_id else evidence
                for evidence in review.evidence
            ]
        }
    )
    counterfactual = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "concept"}))
    ).plan(counterfactual_review)
    diagnosis = counterfactual.response.sections[0]
    diagnosis_moves = [segment.move.san for segment in diagnosis.segments if segment.type == "move"]
    diagnosis_text = "".join(
        segment.text for segment in diagnosis.segments if segment.type == "text"
    )
    assert diagnosis_moves == ["Qe4+"]
    assert "counterfactual queen capture on c6" in diagnosis_text


def test_lesson_copy_does_not_repeat_an_internal_primary_move() -> None:
    review = _review()
    lesson = review.explanation[0].model_copy(
        update={"text": "The checked line beginning with Qe4+ creates a double attack."}
    )
    review = review.model_copy(update={"explanation": [lesson]})

    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(review)

    diagnosis = run.response.sections[0]
    assert [segment.move.san for segment in diagnosis.segments if segment.type == "move"] == [
        "Qe4+"
    ]


def test_internal_pawn_moves_are_structured_without_turning_squares_into_moves() -> None:
    review = _pawn_review()
    lesson = review.explanation[0].model_copy(
        update={"text": "The checked line begins with e4 before e5."}
    )
    review = review.model_copy(update={"explanation": [lesson]})

    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(review)
    diagnosis = run.response.sections[0]
    assert [segment.move.san for segment in diagnosis.segments if segment.type == "move"] == [
        "e4",
        "e5",
    ]

    evidence_id = lesson.evidence_ids[0]
    counterfactual_review = review.model_copy(
        update={
            "evidence": [
                evidence.model_copy(update={"proof": "counterfactual"})
                if evidence.id == evidence_id
                else evidence
                for evidence in review.evidence
            ]
        }
    )
    counterfactual = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(counterfactual_review)
    counterfactual_diagnosis = counterfactual.response.sections[0]
    assert [
        segment.move.san for segment in counterfactual_diagnosis.segments if segment.type == "move"
    ] == ["e4"]
    assert "counterfactual pawn move to e5" in "".join(
        segment.text for segment in counterfactual_diagnosis.segments if segment.type == "text"
    )


def test_repeated_san_references_advance_through_the_checked_line() -> None:
    review = _repeated_san_review()
    lesson = review.explanation[0].model_copy(
        update={"text": "Nc3 improves the knight before Nc3 repeats the position."}
    )
    review = review.model_copy(update={"explanation": [lesson]})

    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(review)
    diagnosis = run.response.sections[0]
    assert [segment.ply for segment in diagnosis.segments if segment.type == "move"] == [0, 4]


def test_line_consequence_must_match_its_checked_line() -> None:
    review = _pawn_review()
    evidence_id = review.explanation[0].evidence_ids[0]
    review = review.model_copy(
        update={
            "evidence": [
                evidence.model_copy(update={"proof": "line_consequence", "moves": ["a2a3"]})
                if evidence.id == evidence_id
                else evidence
                for evidence in review.evidence
            ]
        }
    )

    with pytest.raises(
        ValueError, match="Line-consequence evidence is absent from its checked line"
    ):
        CommentaryCoach(
            FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
        ).plan(review)


def test_leading_san_must_match_the_checked_token_exactly() -> None:
    review = _pawn_review()
    lesson = review.explanation[0].model_copy(update={"text": "e4+ gives check."})
    review = review.model_copy(update={"explanation": [lesson]})

    with pytest.raises(ValueError, match=r"e4\+ is absent from its checked line"):
        CommentaryCoach(
            FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
        ).plan(review)


def test_noncounterfactual_lesson_rejects_a_move_outside_its_line() -> None:
    review = _review()
    lesson = review.explanation[0].model_copy(
        update={"text": "Qe4+ creates the double attack before Qh5."}
    )
    review = review.model_copy(update={"explanation": [lesson]})

    with pytest.raises(ValueError, match="Qh5 is absent from its checked line"):
        CommentaryCoach(
            FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
        ).plan(review)


def test_secondary_line_claim_shows_its_ordered_proof_sequence() -> None:
    review = _review()
    extra_evidence = review.evidence[0].model_copy(
        update={
            "id": "extra-evidence",
            "proof": "line_consequence",
            "moves": [review.lines[0].moves[0].uci, review.lines[0].moves[2].uci],
        }
    )
    extra_lesson = review.explanation[0].model_copy(
        update={
            "id": "extra-lesson",
            "label": "Best · Material sequence",
            "text": "Qe4+ starts the checked sequence.",
            "evidence_ids": [extra_evidence.id],
        }
    )
    review = review.model_copy(
        update={
            "evidence": [*review.evidence, extra_evidence],
            "explanation": [*review.explanation, extra_lesson],
        }
    )

    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1", "claim-2"], "focus": "cause"}))
    ).plan(review)

    idea = next(section for section in run.response.sections if section.kind == "idea")
    assert [segment.move.san for segment in idea.segments if segment.type == "move"][-3:] == [
        "Qe4+",
        "Kg8",
        "Qxc6",
    ]


def test_root_annotation_uses_its_verified_evidence_line() -> None:
    review = _review()
    root_lesson = review.explanation[0].model_copy(update={"scope": "root"})
    review = review.model_copy(update={"explanation": [root_lesson]})

    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(review)

    moves = [
        segment
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "move"
    ]
    assert moves
    assert all(move.scope == "best_line" for move in moves)
    assert moves[0].move.san == "Qe4+"
    validate_position_coaching(review, run.response.model_dump(mode="json"))


def test_equivalent_first_choice_avoids_a_fake_better_move() -> None:
    base = _review()
    request = PositionReviewRequest.model_validate(
        {
            "fen": base.fen,
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
                "attempt": {
                    "move": "e2e4",
                    "line": {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": 500},
                        "wdl": [920, 78, 2],
                        "pv": ["e2e4", "h7g8", "e4c6"],
                        "stable": True,
                    },
                },
            },
        }
    )
    review = build_position_review(request, review_id="2" * 32)
    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "comparison"}))
    ).plan(review)

    assert all(section.kind != "alternative" for section in run.response.sections)
    copy = " ".join(
        segment.text
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "text"
    )
    assert "evaluation shifts" not in copy
    assert "as its first choice" in copy
    assert "most forcing reply" not in copy
    assert "test the checked reply" in copy
    assert all(
        segment.role != "better"
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "move"
    )


def test_coaching_contract_rejects_missing_or_conflicting_narrative() -> None:
    run = CommentaryCoach(
        FakeProvider(json.dumps({"claim_ids": ["claim-1"], "focus": "cause"}))
    ).plan(_review())
    payload = run.response.model_dump(mode="json")
    payload["headline"] = "The queen wins by force."
    with pytest.raises(ValueError, match="headline must match"):
        PositionCoachingResponse.model_validate(payload)

    payload = run.response.model_dump(mode="json")
    payload["sections"] = []
    with pytest.raises(ValueError, match="verified narrative"):
        PositionCoachingResponse.model_validate(payload)


def test_planner_rejects_unsupported_claims_and_falls_back_safely() -> None:
    raw_output = json.dumps({"claim_ids": ["invented"], "focus": "concept"})
    coach = CommentaryCoach(FakeProvider(raw_output))

    run = coach.plan(_review())

    assert run.response.status == "fallback"
    assert run.response.message == (
        "The coach could not prioritize an extra lesson. The checked analysis is unchanged."
    )
    assert run.response.lesson_ids == ["explanation-1"]
    assert run.response.sections
    assert run.record.accepted_claim_ids == ()
    assert run.record.raw_output == raw_output
    assert run.record.error_code == "unsupported_claim"


def test_planner_requires_the_checked_refutation_as_primary() -> None:
    review = _attempt_review()
    extra_evidence = review.evidence[0].model_copy(update={"id": "extra-evidence"})
    extra_lesson = review.explanation[1].model_copy(
        update={
            "id": "extra-lesson",
            "label": "Best · Secondary idea",
            "scope": "best_line",
            "evidence_ids": [extra_evidence.id],
        }
    )
    review = review.model_copy(
        update={
            "evidence": [*review.evidence, extra_evidence],
            "explanation": [*review.explanation, extra_lesson],
        }
    )

    provider = FakeProvider(json.dumps({"claim_ids": ["claim-2"], "focus": "concept"}))
    run = CommentaryCoach(provider).plan(review)

    assert run.response.status == "fallback"
    assert run.response.lesson_ids == ["explanation-2"]
    assert run.record.error_code == "causal_claim_required"
    assert provider.packets[0]["required_primary_claim_id"] == "claim-1"


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

    assert result.request["prompt_version"] == "commentary-selection-2"
    assert result.request["endpoint"] == "https://provider.invalid/v1/chat/completions"
    assert "temperature" not in result.request["json"]
    assert result.request["json"]["max_tokens"] == 64
    response_format = result.request["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["required"] == [
        "claim_ids",
        "focus",
    ]
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


def test_provider_transport_disables_ambient_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_handlers: list[object] = []

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    class Opener:
        def open(self, request: object, *, timeout: float) -> Response:
            return Response(b"{}")

    def build_opener(*handlers: object) -> Opener:
        captured_handlers.extend(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    outcome = _provider_request_outcome(
        "http://127.0.0.1:8765/v1/chat/completions",
        b"{}",
        {"Authorization": "Bearer secret"},
        1,
    )

    assert outcome == ("ok", None, b"{}")
    proxy_handler = next(
        handler for handler in captured_handlers if isinstance(handler, urllib.request.ProxyHandler)
    )
    assert proxy_handler.proxies == {}


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


def test_fixed_single_claim_skips_provider_selection() -> None:
    provider = FakeProvider('{"claim_ids":["claim-1"],"focus":"cause"}')
    coach = CommentaryCoach(provider)
    review = build_position_review(
        PositionReviewRequest(fen="5Q1k/8/6K1/8/8/8/8/8 b - - 0 1"),
        review_id="3" * 32,
    )

    assert coach.selection_required(review) is False
    run = coach.plan(review, provider_selection=False)

    assert run.response.status == "fallback"
    assert run.response.sections
    assert all(section.kind != "practice" for section in run.response.sections)
    assert run.record.error_code == "selection_not_needed"
    assert run.record.provider_called is False
    assert provider.packets == []
    assert "forced reply" not in " ".join(
        segment.text
        for section in run.response.sections
        for segment in section.segments
        if segment.type == "text"
    )
    assert CommentaryCoach(provider).selection_required(_attempt_review()) is True


def test_candidate_free_review_skips_packet_and_provider_call() -> None:
    provider = FakeProvider('{"claim_ids":["claim-1"],"focus":"cause"}')
    review = _review().model_copy(update={"explanation": []})

    run = CommentaryCoach(provider).plan(review)

    assert run.response.status == "fallback"
    assert run.response.message == (
        "The checked analysis contains no additional evidence-backed coaching note."
    )
    assert run.response.sections == []
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


def _attempt_review():
    request = PositionReviewRequest.model_validate(
        {
            "fen": "7k/8/2n5/8/8/5N2/8/4K3 w - - 0 1",
            "analysis": {
                "score_pov": "side_to_move",
                "lines": [
                    {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": 0},
                        "wdl": [100, 800, 100],
                        "pv": ["e1d1", "h8g8", "d1e1"],
                        "stable": True,
                    }
                ],
                "attempt": {
                    "move": "f3d4",
                    "line": {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": -350},
                        "wdl": [0, 100, 900],
                        "pv": ["f3d4", "c6d4", "e1d1"],
                        "stable": True,
                    },
                },
            },
        }
    )
    return build_position_review(request, review_id="1" * 32)


def _repeated_san_review():
    request = PositionReviewRequest.model_validate(
        {
            "fen": "7k/p7/8/8/8/8/P7/1N2K3 w - - 0 1",
            "analysis": {
                "score_pov": "side_to_move",
                "lines": [
                    {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": 0},
                        "wdl": [100, 800, 100],
                        "pv": ["b1c3", "h8h7", "c3b1", "h7h8", "b1c3"],
                        "stable": True,
                    }
                ],
            },
        }
    )
    return build_position_review(request, review_id="5" * 32)


def _pawn_review():
    request = PositionReviewRequest.model_validate(
        {
            "fen": "7k/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "analysis": {
                "score_pov": "side_to_move",
                "lines": [
                    {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": 50},
                        "wdl": [300, 600, 100],
                        "pv": ["e2e4", "h8g8", "e4e5"],
                        "stable": True,
                    }
                ],
            },
        }
    )
    return build_position_review(request, review_id="4" * 32)


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
