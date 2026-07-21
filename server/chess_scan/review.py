"""Single-position review orchestration and deterministic mock verbalization."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chess_scan.board import validate_full_fen
from chess_scan.review_detectors import (
    AnalyzedLine,
    DetectedSubject,
    ReviewContext,
    build_analyzed_line,
    detect_subjects,
)
from chess_scan.review_topics import (
    Course,
    ReviewTopic,
    TopicCapability,
    topics_for_level,
)
from chess_scan.schemas import (
    EngineLineInput,
    PositionReviewRequest,
    PositionReviewResponse,
    ReviewEvidence,
    ReviewLine,
    ReviewMove,
    TopicFindingResponse,
)


@dataclass(frozen=True, slots=True)
class PlannedReview:
    primary: TopicFindingResponse | None
    findings: tuple[TopicFindingResponse, ...]


_STOCKFISH_ENGINE = "Stockfish 18 lite"
_RULES_ENGINE = "Deterministic rules"
_AUTOMATIC_CAPABILITIES = {TopicCapability.DETECTOR, TopicCapability.EVALUATOR}


def build_position_review(request: PositionReviewRequest) -> PositionReviewResponse:
    board = validate_full_fen(request.fen)
    if board.is_game_over(claim_draw=True):
        if request.lines:
            raise ValueError("A finished position must not include engine variations")
        return _terminal_review(board, request)

    ordered_inputs = _ordered_lines(request.lines)
    analyzed_lines = tuple(build_analyzed_line(board, line.pv) for line in ordered_inputs)
    teaching_plies = _calculation_plies(request.study_level, request.mode)
    lines = tuple(
        _review_line(board, line, analyzed, max_plies=teaching_plies)
        for line, analyzed in zip(ordered_inputs, analyzed_lines, strict=True)
    )
    main_input = ordered_inputs[0]
    teaching_line = _line_prefix(analyzed_lines[0], teaching_plies)
    plan = _plan_review(
        ReviewContext(board=board, line=teaching_line),
        study_level=request.study_level,
    )
    explanation = MockVerbalizer().verbalize(
        board=board,
        line=lines[0],
        plan=plan,
    )
    return PositionReviewResponse(
        fen=board.fen(),
        engine=_STOCKFISH_ENGINE,
        evaluation=_evaluation_label(main_input.score.kind, main_input.score.value),
        best_move=lines[0].moves[0],
        lines=list(lines),
        primary_finding=plan.primary,
        findings=list(plan.findings),
        explanation=explanation,
    )


class MockVerbalizer:
    """Stable stand-in for the future s46 LLM adapter."""

    def verbalize(
        self,
        *,
        board: chess.Board,
        line: ReviewLine,
        plan: PlannedReview,
    ) -> str:
        best = line.moves[0]
        sentences = [f"Stockfish prefers {best.san}."]
        if plan.primary is None:
            sentences.append(
                "No supported study topic can be proven reliably from the checked line, "
                "so the review stops at the concrete moves."
            )
        else:
            sentences.extend(evidence.summary for evidence in plan.primary.evidence)
            sentences.append(f"The review classifies this as {plan.primary.topic.lower()}.")
        if len(line.moves) > 1:
            reply = line.moves[1]
            continuation = f"After {reply.san}"
            if len(line.moves) > 2:
                continuation += f", the line continues with {line.moves[2].san}"
            sentences.append(f"{continuation}.")
        if board.is_check():
            sentences.append("The starting side is in check, so answering the check comes first.")
        return " ".join(sentences)


def _plan_review(context: ReviewContext, *, study_level: int) -> PlannedReview:
    findings: list[TopicFindingResponse] = []
    for detected in detect_subjects(context):
        topic = _topic_for_handler(detected, study_level=study_level)
        if topic is None:
            continue
        findings.append(_finding_response(topic, detected))
    primary = findings[0] if findings else None
    return PlannedReview(primary=primary, findings=tuple(findings))


def _topic_for_handler(
    finding: DetectedSubject,
    *,
    study_level: int,
) -> ReviewTopic | None:
    candidates = list(_eligible_topics(finding.handler, study_level))
    if not candidates:
        return None
    evidence_kind = finding.evidence[0].kind
    preferred_slug = _preferred_topic_slug(finding.handler, evidence_kind, study_level)
    if preferred_slug:
        preferred = [topic for topic in candidates if topic.id.endswith(f".{preferred_slug}")]
        if preferred:
            candidates = preferred
    return _best_topic(candidates)


def _preferred_topic_slug(handler: str, evidence_kind: str, study_level: int) -> str | None:
    if handler == "mate":
        if evidence_kind == "mate":
            return "mate-one"
        return "mate-in-two" if study_level >= 2 else "mate-two"
    if handler == "double_attack" and study_level >= 2:
        return {
            "two_targets_queen": "double-attack-one",
            "two_targets_knight": "double-attack-two",
        }.get(evidence_kind, "double-attack-pieces")
    if handler == "mate_technique":
        return "mating-with-rook" if evidence_kind == "rook_mate" else "mating-with-queen"
    return None


def _finding_response(topic: ReviewTopic, finding: DetectedSubject) -> TopicFindingResponse:
    return TopicFindingResponse(
        topic_id=topic.id,
        topic=topic.name,
        level=topic.level,
        confidence=finding.confidence,
        evidence=[
            ReviewEvidence(
                kind=evidence.kind,
                summary=evidence.summary,
                squares=list(evidence.squares),
                moves=list(evidence.moves),
            )
            for evidence in finding.evidence
        ],
    )


def _review_line(
    board: chess.Board,
    line: EngineLineInput,
    analyzed: AnalyzedLine,
    *,
    max_plies: int,
) -> ReviewLine:
    moves = [
        ReviewMove(
            uci=move.uci(),
            san=(board if index == 0 else analyzed.boards[index - 1]).san(move),
        )
        for index, move in enumerate(analyzed.moves[:max_plies])
    ]
    return ReviewLine(
        multipv=line.multipv,
        depth=line.depth,
        score=line.score,
        wdl=line.wdl,
        moves=moves,
    )


def _line_prefix(line: AnalyzedLine, max_plies: int) -> AnalyzedLine:
    return AnalyzedLine(
        moves=line.moves[:max_plies],
        boards=line.boards[:max_plies],
    )


def _calculation_plies(study_level: int, mode: str) -> int:
    base = {1: 3, 2: 5, 3: 7, 4: 9, 5: 12, 6: 16}[study_level]
    return min(24, base + (2 if mode == "thinking_ahead" else 0))


def _ordered_lines(lines: list[EngineLineInput]) -> list[EngineLineInput]:
    ordered = sorted(lines, key=lambda line: line.multipv)
    indexes = [line.multipv for line in ordered]
    if not indexes or indexes[0] != 1:
        raise ValueError("Engine analysis must include MultiPV line 1")
    if len(indexes) != len(set(indexes)):
        raise ValueError("Engine analysis contains duplicate MultiPV lines")
    return ordered


def _terminal_review(
    board: chess.Board,
    request: PositionReviewRequest,
) -> PositionReviewResponse:
    is_mate = board.is_checkmate()
    handler = "mate" if is_mate else "draw"
    topic = _terminal_topic(handler, request.study_level)
    if is_mate:
        evaluation = "Checkmate"
        summary = "The side to move is checkmated and has no legal reply."
        explanation = (
            "The position is already checkmate. The checked king has no legal move, "
            "capture, or block."
        )
    else:
        evaluation = "Drawn position"
        outcome = board.outcome(claim_draw=True)
        reason = (
            outcome.termination.name.lower().replace("_", " ")
            if outcome
            else "no legal continuation"
        )
        summary = f"The position is drawn by {reason}."
        explanation = f"The game is already over: {summary.lower()}"
    primary = None
    if topic is not None:
        primary = TopicFindingResponse(
            topic_id=topic.id,
            topic=topic.name,
            level=topic.level,
            confidence=1.0,
            evidence=[
                ReviewEvidence(
                    kind="terminal_position",
                    summary=summary,
                    squares=[
                        chess.square_name(square)
                        for color in (chess.WHITE, chess.BLACK)
                        if (square := board.king(color)) is not None
                    ],
                )
            ],
        )
    return PositionReviewResponse(
        fen=board.fen(),
        engine=_RULES_ENGINE,
        evaluation=evaluation,
        best_move=None,
        lines=[],
        primary_finding=primary,
        findings=[primary] if primary else [],
        explanation=explanation,
    )


def _terminal_topic(handler: str, study_level: int) -> ReviewTopic | None:
    candidates = list(_eligible_topics(handler, study_level))
    if handler == "mate":
        generic = [topic for topic in candidates if topic.name == "Mate"]
        if generic:
            candidates = generic
    return _best_topic(candidates)


def _eligible_topics(handler: str, study_level: int) -> tuple[ReviewTopic, ...]:
    return tuple(
        topic
        for topic in topics_for_level(study_level)
        if topic.handler == handler and topic.capability in _AUTOMATIC_CAPABILITIES
    )


def _best_topic(candidates: list[ReviewTopic]) -> ReviewTopic | None:
    return (
        min(candidates, key=lambda topic: (topic.level, topic.course is Course.PLUS, topic.id))
        if candidates
        else None
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
