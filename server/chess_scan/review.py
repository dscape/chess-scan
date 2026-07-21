"""Grounded single-position review orchestration and deterministic commentary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import chess

from chess_scan.board import validate_full_fen
from chess_scan.review_detectors import (
    AnalyzedLine,
    DetectedSubject,
    Evidence,
    ReviewContext,
    build_analyzed_line,
    teaching_subjects,
)
from chess_scan.review_topics import DEFAULT_TOPIC, ReviewTopic, topic_for
from chess_scan.schemas import (
    EngineLineInput,
    EngineScore,
    PositionReviewRequest,
    PositionReviewResponse,
    PositionTopicResponse,
    ReviewAnnotation,
    ReviewArrow,
    ReviewAttemptResponse,
    ReviewEvidenceResponse,
    ReviewFindingResponse,
    ReviewLineResponse,
    ReviewMove,
    ReviewPieceRef,
)

_STOCKFISH_ENGINE = "Stockfish 18 lite"
_RULES_ENGINE = "Deterministic rules"
_FindingScope = Literal["best_line", "attempt_refutation"]


@dataclass(frozen=True, slots=True)
class _SelectedFinding:
    finding: DetectedSubject
    scope: _FindingScope


@dataclass(frozen=True, slots=True)
class _ExplanationPlan:
    topic: ReviewTopic
    finding: DetectedSubject | None
    evidence_ids: tuple[str, ...]
    scope: _FindingScope
    ply: int
    squares: tuple[str, ...]
    arrows: tuple[ReviewArrow, ...]


def build_position_review(
    request: PositionReviewRequest,
    *,
    review_id: str | None = None,
) -> PositionReviewResponse:
    board = validate_full_fen(request.fen)
    if board.is_game_over(claim_draw=True):
        if request.analysis is not None:
            raise ValueError("A finished position must not include engine analysis")
        return _terminal_review(board, fen=request.fen, review_id=review_id)
    if request.analysis is None:
        raise ValueError("Engine analysis is required for a playable position")

    analyzed_candidates = [build_analyzed_line(board, line.pv) for line in request.analysis.lines]
    candidate_lines = [
        _review_line(
            board,
            line,
            analyzed,
            role="best_candidate" if line.rank == 1 else "alternative_candidate",
        )
        for line, analyzed in zip(
            request.analysis.lines,
            analyzed_candidates,
            strict=True,
        )
    ]
    best_input = request.analysis.lines[0]
    best_analyzed = analyzed_candidates[0]
    best_findings = teaching_subjects(ReviewContext(board=board, line=best_analyzed))
    attempt, refutation_findings = _review_attempt(board, request, best_input)
    selected = _select_findings(best_findings, refutation_findings, attempt)
    forced_mate = best_input.score.kind == "mate" and best_input.score.value > 0
    teach_forced_mate = forced_mate and (
        attempt is None or attempt.verdict not in {"mistake", "blunder"}
    )
    if teach_forced_mate:
        selected = tuple(
            subject for subject in selected if subject.finding.handler in {"mate", "mate_technique"}
        )[:1]
    findings, evidence = _finding_contract(selected)
    best_evidence_id = "engine-best"
    evidence.append(
        ReviewEvidenceResponse(
            id=best_evidence_id,
            kind="engine_candidate",
            scope="best_line",
            proof="line_consequence",
            ply=0,
            actor=None,
            targets=[],
            squares=[],
            moves=list(best_input.pv),
            score=best_input.score,
            wdl=best_input.wdl,
        )
    )
    attempt_evidence_id = None
    if attempt is not None:
        attempt_evidence_id = "engine-attempt"
        evidence.append(
            ReviewEvidenceResponse(
                id=attempt_evidence_id,
                kind="engine_comparison",
                scope=(
                    "attempt_refutation"
                    if attempt.verdict in {"mistake", "blunder"}
                    else "attempt_line"
                ),
                proof="line_consequence",
                ply=0,
                actor=None,
                targets=[],
                squares=[],
                moves=[move.uci for move in attempt.line.moves],
                score=attempt.line.score,
                wdl=attempt.line.wdl,
                expected_score_loss=attempt.expected_score_loss,
                centipawn_loss=attempt.centipawn_loss,
                lost_forced_mate=attempt.lost_forced_mate,
                mate_delay=attempt.mate_delay,
                verdict=attempt.verdict,
            )
        )
    primary_item = selected[0] if selected else None
    primary = primary_item.finding if primary_item else None
    topic = topic_for("mate_technique") if teach_forced_mate else _topic_for_finding(primary)
    primary_evidence_ids = findings[0].evidence_ids if findings else [best_evidence_id]
    primary_scope = primary_item.scope if primary_item else "best_line"
    primary_ply = primary.evidence[0].ply if primary is not None else 0
    if primary_scope == "attempt_refutation":
        primary_ply += 1
    best_move = _review_move(board, best_analyzed.moves[0])
    explanation_squares = _hint_squares(primary)
    hint_squares = explanation_squares if primary_ply == 0 else []
    arrows = _evidence_arrows(primary)

    explanation = _explanation(
        plan=_ExplanationPlan(
            topic=topic,
            finding=primary,
            evidence_ids=tuple(primary_evidence_ids),
            scope=primary_scope,
            ply=primary_ply,
            squares=tuple(explanation_squares),
            arrows=tuple(arrows),
        ),
        best_move=best_move,
        attempt=attempt,
        attempt_evidence_id=attempt_evidence_id,
        best_evidence_id=best_evidence_id,
    )
    return PositionReviewResponse(
        review_id=review_id,
        fen=request.fen,
        engine=_STOCKFISH_ENGINE,
        evaluation=_evaluation_label(best_input.score.kind, best_input.score.value),
        score=best_input.score,
        score_pov="side_to_move",
        best_move=best_move,
        lines=candidate_lines,
        attempt=attempt,
        topic=_topic_response(topic),
        findings=findings,
        evidence=evidence,
        hint=ReviewAnnotation(
            label=topic.name,
            text=topic.hint,
            squares=hint_squares,
            evidence_ids=primary_evidence_ids,
        ),
        explanation=explanation,
    )


def _review_attempt(
    board: chess.Board,
    request: PositionReviewRequest,
    best: EngineLineInput,
) -> tuple[ReviewAttemptResponse | None, tuple[DetectedSubject, ...]]:
    attempt_input = request.analysis.attempt if request.analysis is not None else None
    if attempt_input is None:
        return None, ()

    analyzed = build_analyzed_line(board, attempt_input.line.pv)
    move = _review_move(board, analyzed.moves[0])
    expected_loss = max(
        0.0,
        _expected_score(best.wdl) - _expected_score(attempt_input.line.wdl),
    )
    attempt_score = attempt_input.line.score
    equivalent = _scores_are_equivalent(best.score, attempt_score, expected_loss)
    lost_forced_mate = (
        best.score.kind == "mate"
        and best.score.value > 0
        and not (attempt_score.kind == "mate" and attempt_score.value > 0)
    )
    centipawn_loss = (
        max(0, best.score.value - attempt_score.value)
        if best.score.kind == attempt_score.kind == "cp"
        else None
    )
    mate_delay = (
        max(0, attempt_score.value - best.score.value)
        if best.score.kind == attempt_score.kind == "mate"
        and best.score.value > 0
        and attempt_score.value > 0
        else None
    )
    verdict = _attempt_verdict(
        attempted=attempt_input.move,
        best=best.pv[0],
        expected_loss=expected_loss,
        equivalent=equivalent,
        lost_forced_mate=lost_forced_mate,
        score_deterioration=_score_deterioration_verdict(best.score, attempt_score),
    )
    response = ReviewAttemptResponse(
        move=move,
        verdict=verdict,
        equivalent=equivalent or attempt_input.move == best.pv[0],
        expected_score_loss=round(expected_loss, 4),
        centipawn_loss=centipawn_loss,
        lost_forced_mate=lost_forced_mate,
        mate_delay=mate_delay,
        line=_review_line(
            board,
            attempt_input.line,
            analyzed,
            role=("attempt_refutation" if verdict in {"mistake", "blunder"} else "attempt_line"),
        ),
    )
    if response.verdict not in {"mistake", "blunder"} or len(analyzed.moves) < 2:
        return response, ()

    opponent_board = analyzed.boards[0]
    opponent_line = AnalyzedLine(analyzed.moves[1:], analyzed.boards[1:])
    if not opponent_line.moves:
        return response, ()
    return response, teaching_subjects(ReviewContext(opponent_board, opponent_line))


def _select_findings(
    best_findings: tuple[DetectedSubject, ...],
    refutation_findings: tuple[DetectedSubject, ...],
    attempt: ReviewAttemptResponse | None,
) -> tuple[_SelectedFinding, ...]:
    selected: list[_SelectedFinding] = []
    if attempt is not None and attempt.verdict in {"mistake", "blunder"} and refutation_findings:
        selected.append(_SelectedFinding(refutation_findings[0], "attempt_refutation"))
    selected.extend(_SelectedFinding(finding, "best_line") for finding in best_findings)

    deduplicated: list[_SelectedFinding] = []
    seen: set[tuple[str, str]] = set()
    for item in selected:
        key = (item.finding.handler, item.scope)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
        if len(deduplicated) == 3:
            break
    return tuple(deduplicated)


def _finding_contract(
    selected: tuple[_SelectedFinding, ...],
) -> tuple[list[ReviewFindingResponse], list[ReviewEvidenceResponse]]:
    findings: list[ReviewFindingResponse] = []
    evidence_items: list[ReviewEvidenceResponse] = []
    for finding_index, item in enumerate(selected, start=1):
        topic = _topic_for_finding(item.finding)
        evidence_ids: list[str] = []
        for evidence_index, evidence in enumerate(item.finding.evidence, start=1):
            evidence_id = f"f{finding_index}-e{evidence_index}"
            evidence_ids.append(evidence_id)
            evidence_items.append(
                _evidence_response(
                    evidence_id,
                    evidence,
                    scope=item.scope,
                )
            )
        findings.append(
            ReviewFindingResponse(
                topic=_topic_response(topic),
                evidence_ids=evidence_ids,
            )
        )
    return findings, evidence_items


def _evidence_response(
    evidence_id: str,
    evidence: Evidence,
    *,
    scope: _FindingScope,
) -> ReviewEvidenceResponse:
    return ReviewEvidenceResponse(
        id=evidence_id,
        kind=evidence.kind,
        scope=scope,
        proof=evidence.proof,
        ply=evidence.ply + (1 if scope == "attempt_refutation" else 0),
        actor=_piece_response(evidence.actor) if evidence.actor else None,
        targets=[_piece_response(target) for target in evidence.targets],
        from_square=evidence.from_square,
        to_square=evidence.to_square,
        squares=list(evidence.squares),
        moves=list(evidence.moves),
    )


def _piece_response(piece: object) -> ReviewPieceRef:
    return ReviewPieceRef(
        color=getattr(piece, "color"),
        piece=getattr(piece, "piece"),
        square=getattr(piece, "square"),
    )


def _review_line(
    board: chess.Board,
    line: EngineLineInput,
    analyzed: AnalyzedLine,
    *,
    role: str,
) -> ReviewLineResponse:
    moves = [
        _review_move(board if index == 0 else analyzed.boards[index - 1], move)
        for index, move in enumerate(analyzed.moves)
    ]
    return ReviewLineResponse(
        role=role,
        rank=line.rank,
        depth=line.depth,
        score=line.score,
        wdl=line.wdl,
        moves=moves,
    )


def _explanation(
    *,
    plan: _ExplanationPlan,
    best_move: ReviewMove,
    attempt: ReviewAttemptResponse | None,
    attempt_evidence_id: str | None,
    best_evidence_id: str,
) -> list[ReviewAnnotation]:
    notes: list[ReviewAnnotation] = []
    if attempt is not None:
        notes.append(_attempt_annotation(attempt, evidence_id=attempt_evidence_id))
    topic_text = plan.topic.idea
    if plan.finding is None:
        topic_text = (
            "Stockfish verifies a forced mate from this position."
            if plan.topic.id == "mate-technique"
            else "The checked line supports the engine choice, but no specific tactical mechanism."
        )
    notes.append(
        ReviewAnnotation(
            label=plan.topic.name,
            text=topic_text,
            scope=plan.scope,
            ply=plan.ply,
            squares=list(plan.squares),
            arrows=list(plan.arrows),
            evidence_ids=list(plan.evidence_ids),
        )
    )
    if plan.finding is not None:
        evidence = plan.finding.evidence[0]
        notes.append(
            ReviewAnnotation(
                label="Why it works",
                text=evidence.summary,
                scope=plan.scope,
                ply=plan.ply,
                squares=list(evidence.squares),
                arrows=list(plan.arrows),
                evidence_ids=list(plan.evidence_ids),
            )
        )
    else:
        engine_text = (
            f"{best_move.san} starts Stockfish's shortest checked mating line."
            if plan.topic.id == "mate-technique"
            else (
                f"The clearest first move is {best_move.san}. The checked line supports "
                "the move, but not a more specific tactical label."
            )
        )
        notes.append(
            ReviewAnnotation(
                label="Engine choice",
                text=engine_text,
                scope="best_line",
                ply=0,
                arrows=list(plan.arrows),
                evidence_ids=[best_evidence_id],
            )
        )
    return notes


def _attempt_annotation(
    attempt: ReviewAttemptResponse,
    *,
    evidence_id: str | None,
) -> ReviewAnnotation:
    percent = round(attempt.expected_score_loss * 100)
    if attempt.equivalent:
        text = f"{attempt.move.san} is effectively equivalent to the engine's first choice."
    elif attempt.verdict == "blunder" and percent > 0:
        text = (
            f"{attempt.move.san} loses about {percent} percentage points of expected score. "
            "The hypothetical refutation below shows the concrete consequence."
        )
    elif attempt.verdict in {"mistake", "blunder"}:
        text = (
            f"{attempt.move.san} is a {attempt.verdict}. "
            "The hypothetical refutation below shows the concrete consequence."
        )
    elif attempt.lost_forced_mate:
        text = (
            f"{attempt.move.san} remains favorable, but it gives up the forced mate "
            "Stockfish found in the best line."
        )
    elif percent > 0:
        text = (
            f"{attempt.move.san} is a {attempt.verdict}. It gives up about {percent} "
            "percentage points of expected score."
        )
    elif attempt.centipawn_loss is not None:
        text = (
            f"{attempt.move.san} is not equivalent to the first choice: its checked "
            f"evaluation is {attempt.centipawn_loss} centipawns lower."
        )
    else:
        text = f"{attempt.move.san} is not equivalent to the engine's first choice."
    return ReviewAnnotation(
        label="Your move",
        text=text,
        evidence_ids=[evidence_id] if evidence_id else [],
    )


def _evidence_arrows(finding: DetectedSubject | None) -> list[ReviewArrow]:
    if finding is None or not finding.evidence:
        return []
    evidence = finding.evidence[0]
    arrows: list[ReviewArrow] = []
    if evidence.from_square is not None and evidence.to_square is not None:
        arrows.append(
            ReviewArrow(
                from_square=evidence.from_square,
                to_square=evidence.to_square,
                kind="move",
            )
        )
    elif evidence.moves:
        move = chess.Move.from_uci(evidence.moves[0])
        arrows.append(
            ReviewArrow(
                from_square=chess.square_name(move.from_square),
                to_square=chess.square_name(move.to_square),
                kind="move",
            )
        )
    if evidence.actor is not None:
        for target in evidence.targets[:3]:
            if evidence.actor.square == target.square:
                continue
            arrows.append(
                ReviewArrow(
                    from_square=evidence.actor.square,
                    to_square=target.square,
                    kind="idea",
                )
            )
    return arrows


def _hint_squares(finding: DetectedSubject | None) -> list[str]:
    if finding is None:
        return []
    evidence = finding.evidence[0]
    if evidence.targets:
        return list(dict.fromkeys(target.square for target in evidence.targets))
    return list(dict.fromkeys(evidence.squares))


def _review_move(board: chess.Board, move: chess.Move) -> ReviewMove:
    return ReviewMove(uci=move.uci(), san=board.san(move))


def _topic_for_finding(finding: DetectedSubject | None) -> ReviewTopic:
    if finding is None:
        return DEFAULT_TOPIC
    return topic_for(finding.handler, finding.evidence[0].kind)


def _topic_response(topic: ReviewTopic) -> PositionTopicResponse:
    return PositionTopicResponse(id=topic.id, name=topic.name)


def _expected_score(wdl: list[int]) -> float:
    return (wdl[0] + wdl[1] / 2) / 1000


def _scores_are_equivalent(
    best: EngineScore,
    attempted: EngineScore,
    expected_loss: float,
) -> bool:
    if expected_loss > 0.02:
        return False
    best_kind = best.kind
    best_value = best.value
    attempted_kind = attempted.kind
    attempted_value = attempted.value
    if best_kind == "mate":
        if best_value > 0:
            return attempted_kind == "mate" and 0 < attempted_value <= best_value + 1
        if attempted_kind == "cp":
            return attempted_value >= 0
        return attempted_value > 0 or abs(attempted_value) >= abs(best_value) - 1
    if attempted_kind == "mate":
        return attempted_value > 0
    return attempted_value >= best_value - 50


def _attempt_verdict(
    *,
    attempted: str,
    best: str,
    expected_loss: float,
    equivalent: bool,
    lost_forced_mate: bool,
    score_deterioration: str | None,
) -> str:
    if attempted == best:
        return "best"
    if equivalent:
        return "excellent"
    if score_deterioration is not None:
        return score_deterioration
    if lost_forced_mate and expected_loss <= 0.02:
        return "inaccuracy"
    if expected_loss <= 0.05:
        return "good"
    if expected_loss <= 0.10:
        return "inaccuracy"
    if expected_loss <= 0.20:
        return "mistake"
    return "blunder"


def _score_deterioration_verdict(
    best: EngineScore,
    attempted: EngineScore,
) -> str | None:
    if best.kind == attempted.kind == "mate":
        if best.value < 0 and attempted.value < 0:
            lost_distance = abs(best.value) - abs(attempted.value)
            if lost_distance >= 4:
                return "blunder"
            if lost_distance >= 2:
                return "mistake"
        if best.value > 0 and attempted.value > best.value + 1:
            return "inaccuracy"
        return None
    if best.kind == "cp" and attempted.kind == "mate" and attempted.value < 0:
        return "blunder"
    if best.kind == attempted.kind == "cp":
        centipawn_loss = best.value - attempted.value
        if centipawn_loss >= 400:
            return "blunder"
        if centipawn_loss >= 200:
            return "mistake"
    return None


def _terminal_review(
    board: chess.Board,
    *,
    fen: str,
    review_id: str | None,
) -> PositionReviewResponse:
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

    evidence_id = "terminal-e1"
    evidence = ReviewEvidenceResponse(
        id=evidence_id,
        kind="checkmate" if is_mate else "draw",
        scope="terminal",
        proof="direct_rule",
        ply=0,
        actor=None,
        targets=[],
        squares=king_squares,
        moves=[],
    )
    finding = ReviewFindingResponse(
        topic=_topic_response(topic),
        evidence_ids=[evidence_id],
    )
    return PositionReviewResponse(
        review_id=review_id,
        fen=fen,
        engine=_RULES_ENGINE,
        evaluation=evaluation,
        score=None,
        score_pov=None,
        best_move=None,
        lines=[],
        attempt=None,
        topic=_topic_response(topic),
        findings=[finding],
        evidence=[evidence],
        hint=ReviewAnnotation(
            label=topic.name,
            text=hint,
            scope="terminal",
            squares=king_squares,
            evidence_ids=[evidence_id],
        ),
        explanation=[
            ReviewAnnotation(
                label="Final position",
                text=explanation,
                scope="terminal",
                squares=king_squares,
                evidence_ids=[evidence_id],
            )
        ],
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
