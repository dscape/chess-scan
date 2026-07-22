from __future__ import annotations

import chess
import pytest
from pydantic import ValidationError

from chess_scan.review import _evidence_arrows, _evidence_badge, build_position_review
from chess_scan.review_detectors import (
    Evidence,
    ReviewContext,
    build_analyzed_line,
    teaching_subjects,
)
from chess_scan.schemas import PositionReviewRequest, PositionReviewResponse

DOUBLE_ATTACK_FEN = "8/7k/2r5/8/8/8/4Q3/4K3 w - - 0 1"
DOUBLE_ATTACK_LINE = ["e2e4", "h7g8", "e4c6"]


def _request(
    *,
    fen: str = DOUBLE_ATTACK_FEN,
    moves: list[str] = DOUBLE_ATTACK_LINE,
) -> PositionReviewRequest:
    return PositionReviewRequest.model_validate(
        {
            "fen": fen,
            "analysis": {
                "score_pov": "side_to_move",
                "lines": [
                    {
                        "rank": 1,
                        "depth": 18,
                        "score": {"kind": "cp", "value": 520},
                        "wdl": [930, 69, 1],
                        "pv": moves,
                        "stable": True,
                    }
                ],
            },
        }
    )


def test_review_returns_one_topic_and_spoiler_free_human_hint() -> None:
    review = build_position_review(_request())

    assert review.best_move is not None
    assert review.best_move.uci == "e2e4"
    assert review.best_move.san == "Qe4+"
    assert review.topic.id == "double-attack"
    assert review.topic.name == "Double attack"
    assert review.hint.label == "Double attack"
    assert [marker.model_dump() for marker in review.hint.markers] == [
        {"square": "c6", "role": "focus"},
        {"square": "h7", "role": "focus"},
    ]
    assert "Qe4" not in review.hint.text
    assert "e2e4" not in review.hint.text
    assert review.explanation[0].label == "Double attack"
    assert "two important targets" in review.explanation[0].text
    assert [arrow.model_dump() for arrow in review.explanation[0].arrows] == [
        {"from_square": "e2", "to_square": "e4", "role": "engine"},
        {"from_square": "e4", "to_square": "c6", "role": "attack"},
        {"from_square": "e4", "to_square": "h7", "role": "attack"},
    ]
    assert review.explanation[0].badge is not None
    assert review.explanation[0].badge.model_dump() == {
        "kind": "fork",
        "square": "e4",
        "role": "engine",
        "arrow_index": 1,
    }
    assert "attacks the rook on c6 and the king on h7 at once" in review.explanation[0].text
    assert review.explanation[0].scope == "best_line"
    assert review.explanation[0].ply == 0
    assert review.evaluation == "The side to move has a winning advantage"
    assert review.score is not None and review.score.value == 520


def test_check_topic_teaches_giving_check_to_the_opponent() -> None:
    review = build_position_review(
        _request(
            fen="7k/8/8/8/8/8/8/R5K1 w - - 0 1",
            moves=["a1a8", "h8g7"],
        )
    )

    assert review.topic.name == "Giving check"
    assert "opponent must answer" in review.explanation[0].text


def test_topic_selection_uses_the_piece_and_line_shape() -> None:
    review = build_position_review(
        _request(
            fen="8/7k/8/7r/4N3/8/8/4K3 w - - 0 1",
            moves=["e4f6", "h7g7", "f6h5"],
        )
    )

    assert review.topic.id == "knight-fork"
    assert review.topic.name == "Knight fork"


def test_development_requires_the_best_move_to_develop_a_minor_piece() -> None:
    review = build_position_review(
        _request(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            moves=["g1f3", "g8f6"],
        )
    )

    assert review.topic.name == "Development"
    assert [marker.square for marker in review.hint.markers] == ["g1", "f3"]
    assert [arrow.role for arrow in review.explanation[0].arrows] == ["engine"]


def test_review_returns_the_compact_annotation_contract() -> None:
    payload = build_position_review(_request()).model_dump()

    assert set(payload) == {
        "schema_version",
        "review_id",
        "fen",
        "engine",
        "evaluation",
        "score",
        "score_pov",
        "best_move",
        "lines",
        "attempt",
        "topic",
        "findings",
        "evidence",
        "hint",
        "explanation",
    }
    assert payload["schema_version"] == "position-analysis-4"
    assert payload["lines"][0]["role"] == "best_candidate"
    assert payload["findings"][0]["evidence_ids"] == ["f1-e1"]
    evidence_ids = {item["id"] for item in payload["evidence"]}
    assert all(note["evidence_ids"] for note in [payload["hint"], *payload["explanation"]])
    assert all(
        set(note["evidence_ids"]) <= evidence_ids
        for note in [payload["hint"], *payload["explanation"]]
    )
    assert set(payload["topic"]) == {"id", "name"}


def test_later_evidence_replays_to_its_proven_ply_without_leaking_hint_squares() -> None:
    review = build_position_review(
        _request(
            fen="4r1k1/ppp2pp1/3b1r2/1Qp4p/3P3q/P1P1B2P/2P2PB1/R3R1K1 b - - 3 21",
            moves=["e8e3", "e1e3", "h4f2", "g1h1", "f2e3"],
        )
    )

    assert review.explanation[0].scope == "best_line"
    assert review.explanation[0].ply == 2
    assert review.hint.markers == []

    stored = review.model_dump()
    stored["hint"]["markers"] = [{"square": stored["evidence"][0]["squares"][0], "role": "focus"}]
    with pytest.raises(ValidationError, match="ply contradicts its evidence"):
        PositionReviewResponse.model_validate(stored)


def test_quiet_context_features_abstain_from_move_specific_coaching() -> None:
    review = build_position_review(
        _request(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            moves=["a2a3", "a7a6"],
        )
    )

    assert review.topic.name == "Find the best move"
    assert review.findings == []
    assert review.explanation[-1].label == "Engine choice"
    assert review.explanation[-1].evidence_ids == ["engine-best"]


def test_stored_review_contract_rejects_unknown_evidence_and_mismatched_lines() -> None:
    payload = build_position_review(_request()).model_dump()
    payload["hint"]["evidence_ids"] = ["missing"]
    with pytest.raises(ValidationError, match="unknown evidence"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["best_move"] = {"uci": "e2d3", "san": "Qd3+"}
    with pytest.raises(ValidationError, match="does not match"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["scope"] = "attempt_line"
    with pytest.raises(ValidationError, match="scope contradicts its evidence"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["arrows"][0]["badge"] = "fork"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["arrows"] = []
    with pytest.raises(ValidationError) as missing_arrow:
        PositionReviewResponse.model_validate(payload)
    assert {error["type"] for error in missing_arrow.value.errors()} == {
        "review_badge_arrow_missing"
    }

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["badge"].update({"square": "c6", "arrow_index": 0})
    with pytest.raises(ValidationError) as mismatched_arrow:
        PositionReviewResponse.model_validate(payload)
    assert {error["type"] for error in mismatched_arrow.value.errors()} == {
        "review_badge_arrow_mismatch"
    }

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["badge"]["square"] = "d5"
    with pytest.raises(ValidationError, match="badge is not supported"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["ply"] = 1
    with pytest.raises(ValidationError, match="ply contradicts its evidence"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["arrows"][0]["to_square"] = "e3"
    with pytest.raises(ValidationError, match="move arrow contradicts"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["markers"][0]["square"] = "c5"
    with pytest.raises(ValidationError, match="marker is not supported"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["explanation"][0]["ply"] = 99
    with pytest.raises(ValidationError, match="exceeds its checked line"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["score"]["value"] = 519
    with pytest.raises(ValidationError, match="score does not match"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    payload["lines"][0]["moves"][0]["san"] = "Qe4"
    with pytest.raises(ValidationError, match="contradictory SAN"):
        PositionReviewResponse.model_validate(payload)

    payload = build_position_review(_request()).model_dump()
    best_evidence = next(item for item in payload["evidence"] if item["kind"] == "engine_candidate")
    best_evidence["wdl"] = [929, 70, 1]
    with pytest.raises(ValidationError, match="contradicts its canonical line"):
        PositionReviewResponse.model_validate(payload)

    request_payload = _request().model_dump()
    request_payload["analysis"]["attempt"] = {
        "move": "e2d3",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "cp", "value": 80},
            "wdl": [330, 400, 270],
            "pv": ["e2d3", "h7g7", "d3c4"],
            "stable": True,
        },
    }
    payload = build_position_review(
        PositionReviewRequest.model_validate(request_payload)
    ).model_dump()
    attempt_evidence = next(
        item for item in payload["evidence"] if item["kind"] == "engine_comparison"
    )
    attempt_evidence["verdict"] = "good"
    with pytest.raises(ValidationError, match="contradicts the checked attempt"):
        PositionReviewResponse.model_validate(payload)


def test_detector_evidence_rejects_malformed_proofs_moves_and_squares() -> None:
    with pytest.raises(ValueError, match="Unknown evidence proof"):
        Evidence("bad", "bad", proof="invented")
    with pytest.raises(ValueError, match="invalid move"):
        Evidence("bad", "bad", moves=("not-a-move",))
    with pytest.raises(ValueError, match="invalid move"):
        Evidence("bad", "bad", moves=("0000",))
    with pytest.raises(ValueError, match="invalid square"):
        Evidence("bad", "bad", squares=("z9",))


def test_review_rejects_missing_or_illegal_engine_analysis() -> None:
    missing = PositionReviewRequest.model_validate({"fen": DOUBLE_ATTACK_FEN})
    with pytest.raises(ValueError, match="analysis is required"):
        build_position_review(missing)

    illegal_position = _request(fen="8/8/8/8/8/8/8/8 w - - 0 1", moves=["a1a2"])
    with pytest.raises(ValueError, match="Invalid FEN"):
        build_position_review(illegal_position)

    illegal_line = _request(moves=["e2f4"])
    with pytest.raises(ValueError, match="illegal move"):
        build_position_review(illegal_line)


def test_finished_positions_receive_a_rules_based_result_without_analysis() -> None:
    request = PositionReviewRequest.model_validate({"fen": "5Q1k/8/6K1/8/8/8/8/8 b - - 0 1"})

    review = build_position_review(request)

    assert review.evaluation == "Checkmate"
    assert review.best_move is None
    assert review.score is None
    assert review.topic.name == "Checkmate"
    assert [marker.square for marker in review.hint.markers] == ["g6", "h8"]
    assert review.engine == "Deterministic rules"

    request.analysis = _request().analysis
    with pytest.raises(ValueError, match="must not include engine analysis"):
        build_position_review(request)


def test_claimable_fifty_move_draw_is_handled_as_terminal() -> None:
    review = build_position_review(
        PositionReviewRequest.model_validate({"fen": "7k/7r/8/8/8/8/8/R6K w - - 100 51"})
    )

    assert review.evaluation == "Drawn position"
    assert review.best_move is None
    assert "fifty moves" in review.hint.text


def test_review_does_not_treat_an_unanswered_horizon_capture_as_material_gain() -> None:
    request = _request(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        moves=["e2e4", "e7e5", "f1b5", "b8c6", "b5c6"],
    )

    review = build_position_review(request)

    assert review.topic.name == "Development"
    assert "central space" in review.explanation[0].text


def test_review_compares_an_attempt_by_expected_score_not_exact_move_only() -> None:
    payload = _request().model_dump()
    payload["analysis"]["attempt"] = {
        "move": "e2d3",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "cp", "value": 80},
            "wdl": [330, 400, 270],
            "pv": ["e2d3", "h7g7", "d3c4"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.attempt is not None
    assert review.attempt.move.san == "Qd3+"
    assert review.attempt.verdict == "blunder"
    assert review.attempt.equivalent is False
    assert review.attempt.expected_score_loss == pytest.approx(0.4345)
    assert review.attempt.line.role == "attempt_refutation"
    assert review.explanation[0].label == "Your move"
    assert "percentage points" in review.explanation[0].text
    assert review.explanation[0].scope == "attempt_refutation"
    assert review.explanation[0].ply == 0
    assert review.explanation[0].arrows[0].role == "played"
    assert review.explanation[0].markers[0].role == "danger"


def test_bad_attempt_gets_a_hypothetical_reply_diagram_without_a_named_tactic() -> None:
    payload = _request(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        moves=["e2e4", "e7e5"],
    ).model_dump()
    payload["analysis"]["attempt"] = {
        "move": "a2a3",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "cp", "value": -100},
            "wdl": [100, 300, 600],
            "pv": ["a2a3", "a7a6", "b2b3"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.explanation[0].label == "Your move"
    assert review.explanation[1].label == "Strongest reply"
    assert review.explanation[1].scope == "attempt_refutation"
    assert review.explanation[1].ply == 1
    assert review.explanation[1].arrows[0].role == "reply"
    assert "hypothetical" in review.explanation[1].text

    stored = review.model_dump()
    stored["explanation"][1]["arrows"][0]["to_square"] = "g7"
    with pytest.raises(ValidationError, match="move arrow contradicts"):
        PositionReviewResponse.model_validate(stored)

    stored = review.model_dump()
    stored["explanation"][1]["markers"][0]["square"] = "b3"
    with pytest.raises(ValidationError, match="marker is not supported"):
        PositionReviewResponse.model_validate(stored)


def test_empty_root_finding_keeps_the_checked_better_move_diagram() -> None:
    payload = _request(
        fen="6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
        moves=["a1a7", "g8f8"],
    ).model_dump()
    payload["analysis"]["attempt"] = {
        "move": "g1f2",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "cp", "value": 0},
            "wdl": [200, 400, 400],
            "pv": ["g1f2", "g8f8"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    better_move = next(note for note in review.explanation if note.label == "Better move")
    assert better_move.arrows[0].role == "engine"
    assert better_move.arrows[0].from_square == "a1"
    assert better_move.arrows[0].to_square == "a7"


def test_review_accepts_a_different_engine_candidate_when_effectively_equivalent() -> None:
    payload = _request().model_dump()
    alternative = {
        "rank": 2,
        "depth": 18,
        "score": {"kind": "cp", "value": 500},
        "wdl": [920, 78, 2],
        "pv": ["e2d3", "h7g7", "d3c4"],
        "stable": True,
    }
    payload["analysis"]["lines"].append(alternative)
    payload["analysis"]["attempt"] = {
        "move": "e2d3",
        "line": {**alternative, "rank": 1},
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.attempt is not None
    assert review.attempt.verdict == "excellent"
    assert review.attempt.equivalent is True
    assert [line.rank for line in review.lines] == [1, 2]


def test_saturated_losing_wdl_uses_mate_distance_to_grade_the_attempt() -> None:
    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["score"] = {"kind": "mate", "value": -10}
    payload["analysis"]["lines"][0]["wdl"] = [0, 0, 1000]
    payload["analysis"]["attempt"] = {
        "move": "e2d3",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "mate", "value": -1},
            "wdl": [0, 0, 1000],
            "pv": ["e2d3", "h7g7", "d3c4"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.attempt is not None
    assert review.attempt.expected_score_loss == 0
    assert review.attempt.equivalent is False
    assert review.attempt.verdict == "blunder"
    assert review.attempt.line.role == "attempt_refutation"
    assert "0 percentage points" not in review.explanation[0].text
    refutation_evidence = next(
        item for item in review.evidence if item.scope == "attempt_refutation"
    )
    assert refutation_evidence.ply >= 1


def test_saturated_wdl_uses_centipawns_as_an_equivalence_tiebreak() -> None:
    payload = _request().model_dump()
    payload["analysis"]["attempt"] = {
        "move": "e2d3",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "cp", "value": 400},
            "wdl": [930, 69, 1],
            "pv": ["e2d3", "h7g7", "d3c4"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.attempt is not None
    assert review.attempt.equivalent is False
    assert review.attempt.centipawn_loss == 120
    assert review.attempt.verdict == "good"


def test_positive_forced_mate_outranks_unrelated_position_features() -> None:
    payload = _request(fen="3r2k1/5ppp/8/8/1b6/8/1PP2PPP/R2Q2K1 w - - 0 1").model_dump()
    payload["analysis"]["lines"][0] = {
        "rank": 1,
        "depth": 18,
        "score": {"kind": "mate", "value": 7},
        "wdl": [1000, 0, 0],
        "pv": ["d1d8"],
        "stable": True,
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.topic.name == "Mating technique"
    assert review.findings == []
    assert review.hint.evidence_ids == ["engine-best"]


def test_saturated_wdl_does_not_treat_losing_a_forced_mate_as_equivalent() -> None:
    payload = _request(fen="3r2k1/5ppp/8/8/1b6/8/1PP2PPP/R2Q2K1 w - - 0 1").model_dump()
    payload["analysis"]["lines"][0] = {
        "rank": 1,
        "depth": 21,
        "score": {"kind": "mate", "value": 7},
        "wdl": [1000, 0, 0],
        "pv": ["d1d8"],
        "stable": True,
    }
    payload["analysis"]["attempt"] = {
        "move": "a1b1",
        "line": {
            "rank": 1,
            "depth": 21,
            "score": {"kind": "cp", "value": 578},
            "wdl": [1000, 0, 0],
            "pv": ["a1b1", "d8d1", "b1d1"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.attempt is not None
    assert review.attempt.equivalent is False
    assert review.attempt.lost_forced_mate is True
    assert review.attempt.verdict == "inaccuracy"
    assert review.attempt.line.role == "attempt_line"
    assert "gives up the forced mate" in review.explanation[0].text


def test_losing_attempt_does_not_claim_it_remains_favorable() -> None:
    payload = _request(fen="8/7k/8/8/2r5/8/4Q3/4K3 w - - 0 1").model_dump()
    payload["analysis"]["lines"][0] = {
        "rank": 1,
        "depth": 21,
        "score": {"kind": "mate", "value": 9},
        "wdl": [1000, 0, 0],
        "pv": ["e2c4"],
        "stable": True,
    }
    payload["analysis"]["attempt"] = {
        "move": "e2e4",
        "line": {
            "rank": 1,
            "depth": 18,
            "score": {"kind": "cp", "value": -823},
            "wdl": [0, 0, 1000],
            "pv": ["e2e4", "c4e4", "e1d2"],
            "stable": True,
        },
    }

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.attempt is not None and review.attempt.verdict == "blunder"
    assert "loses about 100" in review.explanation[0].text
    assert "remains favorable" not in review.explanation[0].text
    assert review.explanation[1].scope == "attempt_refutation"
    assert review.explanation[1].ply == 1


def test_created_threat_diagram_separates_setup_from_future_capture() -> None:
    board = chess.Board("7k/8/8/8/2q5/8/8/R5K1 w - - 0 1")
    context = ReviewContext(board, build_analyzed_line(board, ["a1a4", "h8g8", "a4c4"]))
    finding = next(item for item in teaching_subjects(context) if item.handler == "threat")

    arrows = _evidence_arrows(finding)
    badge = _evidence_badge(finding, tuple(arrows), scope="best_line")

    assert [(arrow.role, arrow.from_square, arrow.to_square) for arrow in arrows] == [
        ("engine", "a1", "a4"),
        ("threat", "a4", "c4"),
    ]
    assert badge is not None
    assert badge.model_dump() == {
        "kind": "capture",
        "square": "c4",
        "role": "threat",
        "arrow_index": 1,
    }


def test_review_uses_the_causal_move_for_later_theme_arrows() -> None:
    fen = "4Rr1k/pp4pp/5r2/6q1/2P4n/8/PP3QB1/4R1K1 w - - 8 41"
    moves = ["e8f8", "f6f8", "f2f8"]
    board = chess.Board(fen)
    context = ReviewContext(board, build_analyzed_line(board, moves))
    finding = next(item for item in teaching_subjects(context) if item.handler == "xray")

    arrows = _evidence_arrows(finding)

    assert arrows[0].from_square == "f2"
    assert arrows[0].to_square == "f8"
    review = build_position_review(_request(fen=fen, moves=moves))
    evidence = next(item for item in review.evidence if item.kind == "xRayAttack")
    assert evidence.ply == 2
    assert evidence.from_square == "f2"
    assert evidence.to_square == "f8"


def test_review_preserves_uncapturable_en_passant_target() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    review = build_position_review(_request(fen=fen, moves=["e7e5", "g1f3"]))

    assert review.fen == fen


def test_review_preserves_underpromotion_in_uci_and_san() -> None:
    review = build_position_review(
        _request(
            fen="7k/P7/8/8/8/8/8/7K w - - 0 1",
            moves=["a7a8n"],
        )
    )

    assert review.best_move is not None
    assert review.best_move.uci == "a7a8n"
    assert review.best_move.san == "a8=N"
    assert review.lines[0].moves[0].san == "a8=N"


def test_review_request_rejects_unknown_fields() -> None:
    payload = _request().model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PositionReviewRequest.model_validate(payload)


def test_review_request_bounds_untrusted_engine_payload() -> None:
    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["pv"] = ["e2e4"] * 17
    with pytest.raises(ValidationError, match="at most 16 items"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["pv"] = ["e2e2"]
    with pytest.raises(ValidationError, match="invalid UCI move"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["wdl"] = [1, 2, 3]
    with pytest.raises(ValidationError, match="total 1000"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["stable"] = False
    with pytest.raises(ValidationError, match="must be stable"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["attempt"] = {
        "move": "e2d3",
        "line": {**payload["analysis"]["lines"][0], "pv": ["e2e4"]},
    }
    with pytest.raises(ValidationError, match="must begin with the attempted move"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["depth"] = 7
    with pytest.raises(ValidationError, match="greater than or equal to 8"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["score"] = {"kind": "mate", "value": 0}
    with pytest.raises(ValidationError, match="Mate scores must be non-zero"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["score"] = {"kind": "mate", "value": 3}
    with pytest.raises(ValidationError, match="decisive WDL"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    duplicate = {**payload["analysis"]["lines"][0], "rank": 2}
    payload["analysis"]["lines"].append(duplicate)
    with pytest.raises(ValidationError, match="distinct moves"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["score_pov"] = "white"
    with pytest.raises(ValidationError, match="side_to_move"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["analysis"]["lines"][0]["score"]["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PositionReviewRequest.model_validate(payload)
