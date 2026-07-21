from __future__ import annotations

import pytest
from pydantic import ValidationError

from chess_scan.review import build_position_review
from chess_scan.schemas import PositionReviewRequest

DOUBLE_ATTACK_FEN = "8/7k/8/8/2r5/8/4Q3/4K3 w - - 0 1"
DOUBLE_ATTACK_LINE = ["e2e4", "h7g8", "e4c4"]


def _request(
    *,
    fen: str = DOUBLE_ATTACK_FEN,
    moves: list[str] = DOUBLE_ATTACK_LINE,
    study_level: int = 2,
    mode: str = "general",
) -> PositionReviewRequest:
    return PositionReviewRequest.model_validate(
        {
            "fen": fen,
            "study_level": study_level,
            "mode": mode,
            "lines": [
                {
                    "multipv": 1,
                    "depth": 18,
                    "score": {"kind": "cp", "value": 520},
                    "wdl": [930, 69, 1],
                    "pv": moves,
                }
            ],
        }
    )


def test_review_uses_level_specific_term_and_grounded_mock_explanation() -> None:
    review = build_position_review(_request())

    assert review.best_move.uci == "e2e4"
    assert review.best_move.san == "Qe4+"
    assert review.primary_finding is not None
    assert review.primary_finding.topic == "Double attack: queen"
    assert review.primary_finding.topic_id == "level-2.basic.double-attack-one"
    assert review.primary_finding.evidence[0].squares == ["h7", "c4"]
    assert review.verbalizer == "mock"
    assert review.explanation.startswith("Stockfish prefers Qe4+.")
    assert "attacks the king and the rook at once" in review.explanation
    assert review.lines[0].moves[1].san == "Kg8"
    assert review.evaluation == "The side to move has a winning advantage"

    level_one = build_position_review(_request(study_level=1))
    assert level_one.primary_finding is not None
    assert level_one.primary_finding.topic == "The twofold attack"


def test_topic_selection_uses_the_piece_and_line_shape() -> None:
    knight_fork = _request(
        fen="8/7k/8/7r/4N3/8/8/4K3 w - - 0 1",
        moves=["e4f6", "h7g7", "f6h5"],
    )

    review = build_position_review(knight_fork)

    assert review.primary_finding is not None
    assert review.primary_finding.topic == "Double attack: knight"
    assert review.primary_finding.topic_id == "level-2.basic.double-attack-two"


def test_practice_modes_do_not_change_chess_facts() -> None:
    mixed = build_position_review(_request(mode="mix"))
    thinking = build_position_review(_request(mode="thinking_ahead"))

    assert mixed.best_move == thinking.best_move
    assert mixed.primary_finding == thinking.primary_finding


def test_review_keeps_legal_alternative_engine_candidates_separate() -> None:
    request = _request()
    alternative = request.lines[0].model_copy(deep=True)
    alternative.multipv = 2
    alternative.score.value = 180
    alternative.pv = ["e2d3", "h7g7", "d3c4"]
    request.lines.append(alternative)

    review = build_position_review(request)

    assert review.best_move is not None
    assert review.best_move.uci == "e2e4"
    assert [line.multipv for line in review.lines] == [1, 2]
    assert review.lines[1].moves[0].san == "Qd3+"


def test_study_level_and_mode_bound_the_visible_calculation() -> None:
    payload = _request().model_dump()
    payload["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    payload["lines"][0]["pv"] = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]

    payload["study_level"] = 1
    basic = build_position_review(PositionReviewRequest.model_validate(payload))
    assert len(basic.lines[0].moves) == 3

    payload["study_level"] = 2
    level_two = build_position_review(PositionReviewRequest.model_validate(payload))
    assert len(level_two.lines[0].moves) == 5

    payload["mode"] = "thinking_ahead"
    thinking_ahead = build_position_review(PositionReviewRequest.model_validate(payload))
    assert len(thinking_ahead.lines[0].moves) == 6


def test_review_rejects_illegal_position_line_and_duplicate_candidates() -> None:
    illegal_position = _request(fen="8/8/8/8/8/8/8/8 w - - 0 1", moves=["a1a2"])
    with pytest.raises(ValueError, match="Invalid FEN"):
        build_position_review(illegal_position)

    illegal_line = _request(moves=["e2f4"])
    with pytest.raises(ValueError, match="illegal move"):
        build_position_review(illegal_line)

    duplicate = _request()
    duplicate.lines.append(duplicate.lines[0].model_copy())
    with pytest.raises(ValueError, match="duplicate MultiPV"):
        build_position_review(duplicate)


def test_finished_positions_receive_a_deterministic_result_without_engine_lines() -> None:
    payload = _request().model_dump()
    payload["fen"] = "5Q1k/8/6K1/8/8/8/8/8 b - - 0 1"
    payload["lines"] = []

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.evaluation == "Checkmate"
    assert review.best_move is None
    assert review.lines == []
    assert review.primary_finding is not None
    assert review.primary_finding.topic == "Mate"
    assert review.primary_finding.evidence[0].kind == "terminal_position"
    assert review.explanation.startswith("The position is already checkmate.")
    assert review.engine == "Deterministic rules"


def test_claimable_fifty_move_draw_is_handled_as_terminal() -> None:
    payload = _request().model_dump()
    payload["fen"] = "7k/7r/8/8/8/8/8/R6K w - - 100 51"
    payload["lines"] = []

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.evaluation == "Drawn position"
    assert review.best_move is None
    assert "fifty moves" in review.explanation


def test_detectors_only_use_the_calculation_shown_to_the_student() -> None:
    payload = _request().model_dump()
    payload["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    payload["lines"][0]["pv"] = ["e2e4", "e7e5", "f1b5", "b8c6", "b5c6"]

    payload["study_level"] = 1
    basic = build_position_review(PositionReviewRequest.model_validate(payload))
    assert all(
        evidence.kind != "material_gain_line"
        for finding in basic.findings
        for evidence in finding.evidence
    )

    payload["study_level"] = 2
    level_two = build_position_review(PositionReviewRequest.model_validate(payload))
    assert any(
        evidence.kind == "material_gain_line"
        for finding in level_two.findings
        for evidence in finding.evidence
    )


def test_review_uses_implementation_owned_engine_attribution() -> None:
    payload = _request().model_dump()
    payload["engine"] = "Not Stockfish"

    review = build_position_review(PositionReviewRequest.model_validate(payload))

    assert review.engine == "Stockfish 18 lite"
    assert review.explanation.startswith("Stockfish prefers")


def test_review_request_bounds_untrusted_engine_payload() -> None:
    payload = _request().model_dump()
    payload["lines"][0]["pv"] = ["e2e4"] * 25
    with pytest.raises(ValidationError, match="at most 24 items"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["lines"][0]["wdl"] = [1, 2, 3]
    with pytest.raises(ValidationError, match="total 1000"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["lines"][0]["pv"] = ["e2e2"]
    with pytest.raises(ValidationError, match="invalid UCI move"):
        PositionReviewRequest.model_validate(payload)
