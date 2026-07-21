"""Grounded, single-move position review with human-authored coaching copy."""

from __future__ import annotations

import chess

from chess_scan.board import validate_full_fen
from chess_scan.review_detectors import (
    AnalyzedLine,
    DetectedSubject,
    ReviewContext,
    build_analyzed_line,
    detect_subjects,
)
from chess_scan.review_topics import ReviewTopic, topic_for
from chess_scan.schemas import (
    PositionReviewRequest,
    PositionReviewResponse,
    PositionTopicResponse,
    ReviewAnnotation,
    ReviewArrow,
    ReviewMove,
)

_STOCKFISH_ENGINE = "Stockfish 18 lite"
_RULES_ENGINE = "Deterministic rules"
_REVIEW_PLIES = 5


def build_position_review(request: PositionReviewRequest) -> PositionReviewResponse:
    board = validate_full_fen(request.fen)
    if board.is_game_over(claim_draw=True):
        if request.line is not None:
            raise ValueError("A finished position must not include engine analysis")
        return _terminal_review(board)
    if request.line is None:
        raise ValueError("Engine analysis is required for a playable position")

    analyzed = build_analyzed_line(board, request.line.pv)
    checked_line = _line_prefix(analyzed, _REVIEW_PLIES)
    findings = detect_subjects(ReviewContext(board=board, line=checked_line))
    finding = findings[0] if findings else None
    evidence_kind = finding.evidence[0].kind if finding else ""
    topic = topic_for(finding.handler if finding else "", evidence_kind)
    best_move = _review_move(board, checked_line.moves[0])
    squares = _evidence_squares(finding)
    arrows = _review_arrows(checked_line, finding)

    return PositionReviewResponse(
        fen=board.fen(),
        engine=_STOCKFISH_ENGINE,
        evaluation=_evaluation_label(request.line.score.kind, request.line.score.value),
        score=request.line.score,
        best_move=best_move,
        topic=_topic_response(topic),
        hint=ReviewAnnotation(
            label=topic.name,
            text=topic.hint,
            squares=squares,
        ),
        explanation=_explanation(topic, finding, best_move, squares, arrows),
    )


def _explanation(
    topic: ReviewTopic,
    finding: DetectedSubject | None,
    best_move: ReviewMove,
    squares: list[str],
    arrows: list[ReviewArrow],
) -> list[ReviewAnnotation]:
    notes = [
        ReviewAnnotation(
            label=topic.name,
            text=topic.idea,
            squares=squares,
            arrows=arrows,
        )
    ]
    if finding is not None:
        evidence = finding.evidence[0]
        notes.append(
            ReviewAnnotation(
                label="Why it works",
                text=evidence.summary,
                squares=list(evidence.squares),
                arrows=arrows,
            )
        )
    else:
        notes.append(
            ReviewAnnotation(
                label="Engine choice",
                text=(
                    f"The clearest first move is {best_move.san}. "
                    "The checked line supports the move, but not a more specific tactical label."
                ),
                arrows=arrows,
            )
        )
    return notes


def _review_arrows(
    line: AnalyzedLine,
    finding: DetectedSubject | None,
) -> list[ReviewArrow]:
    move = line.moves[0]
    arrows = [
        ReviewArrow(
            from_square=chess.square_name(move.from_square),
            to_square=chess.square_name(move.to_square),
            kind="move",
        )
    ]
    if finding is None:
        return arrows

    after = line.boards[0]
    mover = not after.turn
    seen = {(move.from_square, move.to_square)}
    for square_name in _evidence_squares(finding):
        try:
            target = chess.parse_square(square_name)
        except ValueError:
            continue
        target_piece = after.piece_at(target)
        if target_piece is None or target_piece.color == mover:
            continue
        attackers = sorted(
            after.attackers(mover, target),
            key=lambda square: square != move.to_square,
        )
        for attacker in attackers:
            edge = (attacker, target)
            if attacker == target or edge in seen:
                continue
            arrows.append(
                ReviewArrow(
                    from_square=chess.square_name(attacker),
                    to_square=square_name,
                    kind="idea",
                )
            )
            seen.add(edge)
            break
        if len(arrows) == 4:
            break
    return arrows


def _evidence_squares(finding: DetectedSubject | None) -> list[str]:
    if finding is None:
        return []
    return list(
        dict.fromkeys(square for evidence in finding.evidence for square in evidence.squares)
    )


def _line_prefix(line: AnalyzedLine, max_plies: int) -> AnalyzedLine:
    return AnalyzedLine(
        moves=line.moves[:max_plies],
        boards=line.boards[:max_plies],
    )


def _review_move(board: chess.Board, move: chess.Move) -> ReviewMove:
    return ReviewMove(uci=move.uci(), san=board.san(move))


def _topic_response(topic: ReviewTopic) -> PositionTopicResponse:
    return PositionTopicResponse(id=topic.id, name=topic.name)


def _terminal_review(board: chess.Board) -> PositionReviewResponse:
    is_mate = board.is_checkmate()
    topic = topic_for("mate" if is_mate else "draw")
    king_squares = [
        chess.square_name(square)
        for color in (chess.WHITE, chess.BLACK)
        if (square := board.king(color)) is not None
    ]
    if is_mate:
        evaluation = "Checkmate"
        hint = (
            "The position is already over: the checked king has no legal move, capture, or block."
        )
        explanation = (
            "Every escape square is covered, and the checking piece cannot be captured or blocked."
        )
    else:
        evaluation = "Drawn position"
        outcome = board.outcome(claim_draw=True)
        reason = (
            outcome.termination.name.lower().replace("_", " ")
            if outcome
            else "no legal continuation"
        )
        hint = f"The position is already drawn by {reason}."
        explanation = (
            "There is no winning continuation from this position under the rules of chess."
        )

    annotation = ReviewAnnotation(
        label="Final position",
        text=explanation,
        squares=king_squares,
    )
    return PositionReviewResponse(
        fen=board.fen(),
        engine=_RULES_ENGINE,
        evaluation=evaluation,
        score=None,
        best_move=None,
        topic=_topic_response(topic),
        hint=ReviewAnnotation(
            label=topic.name,
            text=hint,
            squares=king_squares,
        ),
        explanation=[annotation],
    )


def _evaluation_label(kind: str, value: int) -> str:
    if kind == "mate":
        if value > 0:
            return f"Forced mate in {value}"
        return f"Facing forced mate in {abs(value)}"
    absolute = abs(value)
    side = "The side to move" if value >= 0 else "The opponent"
    if absolute < 30:
        return "The position is balanced"
    if absolute < 100:
        return f"{side} has a small advantage"
    if absolute < 250:
        return f"{side} has a clear advantage"
    return f"{side} has a winning advantage"
