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
) -> PositionReviewRequest:
    return PositionReviewRequest.model_validate(
        {
            "fen": fen,
            "line": {
                "multipv": 1,
                "depth": 18,
                "score": {"kind": "cp", "value": 520},
                "pv": moves,
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
    assert review.hint.squares == ["h7", "c4"]
    assert "Qe4" not in review.hint.text
    assert "e2e4" not in review.hint.text
    assert review.explanation[0].label == "Double attack"
    assert "two important targets" in review.explanation[0].text
    assert [arrow.model_dump() for arrow in review.explanation[0].arrows] == [
        {"from_square": "e2", "to_square": "e4", "kind": "move"},
        {"from_square": "e4", "to_square": "h7", "kind": "idea"},
        {"from_square": "e4", "to_square": "c4", "kind": "idea"},
    ]
    assert "attacks the king and the rook at once" in review.explanation[1].text
    assert review.evaluation == "The side to move has a winning advantage"
    assert review.score is not None and review.score.value == 520


def test_topic_selection_uses_the_piece_and_line_shape() -> None:
    review = build_position_review(
        _request(
            fen="8/7k/8/7r/4N3/8/8/4K3 w - - 0 1",
            moves=["e4f6", "h7g7", "f6h5"],
        )
    )

    assert review.topic.id == "knight-fork"
    assert review.topic.name == "Knight fork"


def test_position_subjects_highlight_own_pieces_without_drawing_attack_arrows() -> None:
    review = build_position_review(
        _request(
            fen="2r1k1nr/pp3ppp/q1n1p3/2bpP3/P7/1PP2N2/2Q2PPP/RNB1K2R w - - 0 1",
            moves=["c2d1", "g8e7", "b1a3", "c5a3", "c1a3"],
        )
    )

    assert review.topic.name == "Development"
    assert review.hint.squares == ["b1", "c1", "g8"]
    assert [arrow.kind for arrow in review.explanation[0].arrows] == ["move"]


def test_review_has_no_course_or_candidate_line_setup() -> None:
    payload = build_position_review(_request()).model_dump()

    assert "lines" not in payload
    assert "findings" not in payload
    assert "primary_finding" not in payload
    assert "verbalizer" not in payload
    assert "level" not in payload["topic"]


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
    assert review.hint.squares == ["g6", "h8"]
    assert review.engine == "Deterministic rules"

    request.line = _request().line
    with pytest.raises(ValueError, match="must not include engine analysis"):
        build_position_review(request)


def test_claimable_fifty_move_draw_is_handled_as_terminal() -> None:
    review = build_position_review(
        PositionReviewRequest.model_validate({"fen": "7k/7r/8/8/8/8/8/R6K w - - 100 51"})
    )

    assert review.evaluation == "Drawn position"
    assert review.best_move is None
    assert "fifty moves" in review.hint.text


def test_review_checks_a_short_fixed_prefix_for_grounded_topics() -> None:
    request = _request(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        moves=["e2e4", "e7e5", "f1b5", "b8c6", "b5c6", "d7c6"],
    )

    review = build_position_review(request)

    assert review.topic.name == "Winning material"
    assert "gains about" in review.explanation[1].text


def test_removed_study_controls_are_rejected() -> None:
    payload = _request().model_dump()
    payload["study_level"] = 2
    payload["mode"] = "general"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PositionReviewRequest.model_validate(payload)


def test_review_request_bounds_untrusted_engine_payload() -> None:
    payload = _request().model_dump()
    payload["line"]["pv"] = ["e2e4"] * 25
    with pytest.raises(ValidationError, match="at most 24 items"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["line"]["pv"] = ["e2e2"]
    with pytest.raises(ValidationError, match="invalid UCI move"):
        PositionReviewRequest.model_validate(payload)

    payload = _request().model_dump()
    payload["line"]["multipv"] = 2
    with pytest.raises(ValidationError, match="Input should be 1"):
        PositionReviewRequest.model_validate(payload)
