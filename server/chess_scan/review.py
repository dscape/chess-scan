"""Grounded single-position review orchestration and deterministic commentary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import chess

from chess_scan.board import validate_full_fen
from chess_scan.review_detectors import (
    MAX_TEACHING_FINDINGS,
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
    PositionAttemptRequest,
    PositionReviewRequest,
    PositionReviewResponse,
    PositionTopicResponse,
    ReviewAnnotation,
    ReviewArrow,
    ReviewArrowRole,
    ReviewAttemptInput,
    ReviewAttemptResponse,
    ReviewBadge,
    ReviewDiagramBadge,
    ReviewEvidenceResponse,
    ReviewFindingResponse,
    ReviewLineResponse,
    ReviewMarkerRole,
    ReviewMove,
    ReviewPieceRef,
    ReviewSquareMarker,
    review_attempt_headline,
)

_STOCKFISH_ENGINE = "Stockfish 18 lite"
_RULES_ENGINE = "Deterministic rules"
_EQUIVALENT_EXPECTED_LOSS = 0.02
_NEAR_EQUAL_CENTIPAWNS = 20
_NEAR_EQUAL_EXPECTED_LOSS = 0.05
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
    markers: tuple[ReviewSquareMarker, ...]
    arrows: tuple[ReviewArrow, ...]
    badge: ReviewDiagramBadge | None


def build_position_attempt(request: PositionAttemptRequest) -> ReviewAttemptResponse:
    board = validate_full_fen(request.fen)
    if board.is_game_over(claim_draw=False):
        raise ValueError("A finished position cannot compare an attempted move")
    build_analyzed_line(board, request.analysis.best_line.pv)
    attempt, _ = _review_attempt(
        board,
        request.analysis.attempt,
        request.analysis.best_line,
        include_refutation_findings=False,
        same_move_is_best=not request.path_dependent,
    )
    if attempt is None:
        raise ValueError("Position-attempt analysis requires an attempted move")
    return attempt


def build_position_review(
    request: PositionReviewRequest,
    *,
    review_id: str | None = None,
) -> PositionReviewResponse:
    board = validate_full_fen(request.fen)
    rules_terminal = board.is_game_over(claim_draw=False)
    claimed_terminal = request.analysis is None and board.is_game_over(claim_draw=True)
    if rules_terminal or claimed_terminal:
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
    attempt, refutation_findings = _review_attempt(
        board,
        request.analysis.attempt,
        best_input,
    )
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
    plans = _explanation_plans(selected, findings)
    primary_evidence_ids = findings[0].evidence_ids if findings else [best_evidence_id]
    best_move = _review_move(board, best_analyzed.moves[0])
    if not plans:
        plans = (
            _ExplanationPlan(
                topic=topic,
                finding=None,
                evidence_ids=tuple(primary_evidence_ids),
                scope="best_line",
                ply=0,
                markers=(),
                arrows=(_move_arrow(best_move.uci, role="engine"),),
                badge=ReviewDiagramBadge(
                    kind="engine",
                    square=chess.square_name(best_analyzed.moves[0].to_square),
                    role="engine",
                    arrow_index=0,
                ),
            ),
        )
    hint_markers = _hint_markers(primary) if plans[0].ply == 0 else []

    explanation = _explanation(
        plans=plans,
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
            id="hint",
            label=topic.name,
            text=topic.hint,
            markers=hint_markers,
            evidence_ids=primary_evidence_ids,
        ),
        explanation=explanation,
    )


def _review_attempt(
    board: chess.Board,
    attempt_input: ReviewAttemptInput | None,
    best: EngineLineInput,
    *,
    include_refutation_findings: bool = True,
    same_move_is_best: bool = True,
) -> tuple[ReviewAttemptResponse | None, tuple[DetectedSubject, ...]]:
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
        same_move_is_best=same_move_is_best,
    )
    presented_equivalent = equivalent or (same_move_is_best and attempt_input.move == best.pv[0])
    response = ReviewAttemptResponse(
        move=move,
        headline=review_attempt_headline(
            move.san,
            verdict,
            equivalent=presented_equivalent,
            lost_forced_mate=lost_forced_mate,
        ),
        verdict=verdict,
        equivalent=presented_equivalent,
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
    if (
        not include_refutation_findings
        or response.verdict not in {"mistake", "blunder"}
        or len(analyzed.moves) < 2
    ):
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
        if len(deduplicated) == MAX_TEACHING_FINDINGS:
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


def _explanation_plans(
    selected: tuple[_SelectedFinding, ...],
    findings: list[ReviewFindingResponse],
) -> tuple[_ExplanationPlan, ...]:
    plans: list[_ExplanationPlan] = []
    for item, response in zip(selected, findings, strict=True):
        evidence = item.finding.evidence[0]
        arrows = tuple(_evidence_arrows(item.finding, scope=item.scope))
        plans.append(
            _ExplanationPlan(
                topic=_topic_for_finding(item.finding),
                finding=item.finding,
                evidence_ids=tuple(response.evidence_ids),
                scope=item.scope,
                ply=_scoped_ply(evidence, item.scope),
                markers=tuple(_evidence_markers(item.finding)),
                arrows=arrows,
                badge=_evidence_badge(item.finding, arrows, scope=item.scope),
            )
        )
    return tuple(plans)


def _scoped_ply(evidence: Evidence, scope: _FindingScope) -> int:
    return evidence.ply + (1 if scope == "attempt_refutation" else 0)


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
        ply=_scoped_ply(evidence, scope),
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
    plans: tuple[_ExplanationPlan, ...],
    best_move: ReviewMove,
    attempt: ReviewAttemptResponse | None,
    attempt_evidence_id: str | None,
    best_evidence_id: str,
) -> list[ReviewAnnotation]:
    notes: list[ReviewAnnotation] = []
    if attempt is not None:
        if attempt_evidence_id is None:
            raise ValueError("Attempt commentary requires engine evidence")
        notes.append(
            _attempt_annotation(
                attempt,
                annotation_id="explanation-1",
                evidence_id=attempt_evidence_id,
            )
        )

    visible_plans = _distinct_plans(plans, limit=2)
    if (
        attempt is not None
        and attempt.verdict in {"mistake", "blunder"}
        and len(attempt.line.moves) > 1
        and not _shows_first_refutation(visible_plans, attempt.line.moves[1].uci)
    ):
        notes.append(
            _reply_annotation(
                attempt,
                annotation_id=f"explanation-{len(notes) + 1}",
                evidence_id=attempt_evidence_id,
            )
        )
    for plan in visible_plans:
        if plan.finding is not None:
            text = f"{plan.finding.evidence[0].summary} {plan.topic.idea}"
        elif plan.topic.id == "mate-technique":
            text = f"{best_move.san} starts Stockfish's shortest checked mating line."
        else:
            text = (
                f"The clearest first move is {best_move.san}. The checked line supports "
                "the move, but not a more specific tactical label."
            )
        notes.append(
            ReviewAnnotation(
                id=f"explanation-{len(notes) + 1}",
                label=_plan_label(plan, has_attempt=attempt is not None),
                text=text,
                scope=plan.scope,
                ply=plan.ply,
                markers=list(plan.markers),
                arrows=list(plan.arrows),
                badge=plan.badge,
                evidence_ids=list(plan.evidence_ids),
            )
        )

    has_root_best_diagram = any(
        plan.scope == "best_line" and plan.ply == 0 and (plan.markers or plan.arrows)
        for plan in visible_plans
    )
    if (
        attempt is not None
        and not attempt.equivalent
        and attempt.move.uci != best_move.uci
        and not has_root_best_diagram
    ):
        notes.append(
            ReviewAnnotation(
                id=f"explanation-{len(notes) + 1}",
                label="Better move",
                text=f"Compare the attempt with {best_move.san}, Stockfish's first choice.",
                scope="best_line",
                ply=0,
                markers=[_destination_marker(best_move.uci, role="focus")],
                arrows=[_move_arrow(best_move.uci, role="engine")],
                badge=ReviewDiagramBadge(
                    kind="engine",
                    square=best_move.uci[2:4],
                    role="engine",
                    arrow_index=0,
                ),
                evidence_ids=[best_evidence_id],
            )
        )
    return notes


def _plan_label(plan: _ExplanationPlan, *, has_attempt: bool) -> str:
    if plan.finding is None:
        return "Engine choice"
    if not has_attempt:
        return plan.topic.name
    prefix = "Reply" if plan.scope == "attempt_refutation" else "Best"
    return f"{prefix} · {plan.topic.name}"


def _shows_first_refutation(
    plans: tuple[_ExplanationPlan, ...],
    reply_uci: str,
) -> bool:
    reply = chess.Move.from_uci(reply_uci)
    reply_from = chess.square_name(reply.from_square)
    reply_to = chess.square_name(reply.to_square)
    return any(
        plan.scope == "attempt_refutation"
        and plan.ply == 1
        and plan.arrows
        and plan.arrows[0].from_square == reply_from
        and plan.arrows[0].to_square == reply_to
        for plan in plans
    )


def _reply_annotation(
    attempt: ReviewAttemptResponse,
    *,
    annotation_id: str,
    evidence_id: str,
) -> ReviewAnnotation:
    reply = attempt.line.moves[1]
    return ReviewAnnotation(
        id=annotation_id,
        label="Strongest reply",
        text=(
            f"After {attempt.move.san}, Stockfish checks {reply.san} as the strongest reply. "
            "This continuation is hypothetical."
        ),
        scope="attempt_refutation",
        ply=1,
        markers=[_destination_marker(reply.uci, role="danger")],
        arrows=[_move_arrow(reply.uci, role="reply")],
        evidence_ids=[evidence_id],
    )


def _distinct_plans(
    plans: tuple[_ExplanationPlan, ...],
    *,
    limit: int,
) -> tuple[_ExplanationPlan, ...]:
    selected: list[_ExplanationPlan] = []
    seen: set[tuple[str, int, str | None, str | None]] = set()
    for plan in plans:
        first_arrow = plan.arrows[0] if plan.arrows else None
        key = (
            plan.scope,
            plan.ply,
            first_arrow.from_square if first_arrow else None,
            first_arrow.to_square if first_arrow else None,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(plan)
        if len(selected) == limit:
            break
    return tuple(selected)


def _attempt_annotation(
    attempt: ReviewAttemptResponse,
    *,
    annotation_id: str,
    evidence_id: str,
) -> ReviewAnnotation:
    percent = round(attempt.expected_score_loss * 100)
    if attempt.verdict == "best":
        text = f"{attempt.move.san} matches Stockfish's first choice."
    elif attempt.equivalent:
        text = f"{attempt.move.san} is effectively equivalent to Stockfish's first choice."
    elif attempt.verdict == "blunder" and percent > 0:
        text = (
            f"{attempt.move.san} loses about {percent} percentage points of expected score. "
            "The next diagram shows the hypothetical refutation."
        )
    elif attempt.verdict in {"mistake", "blunder"}:
        text = (
            f"{attempt.move.san} is a {attempt.verdict}. "
            "The next diagram shows the hypothetical refutation."
        )
    elif attempt.lost_forced_mate:
        text = (
            f"{attempt.move.san} remains favorable, but it gives up the forced mate "
            "Stockfish found in the best line."
        )
    elif percent > 0:
        text = (
            f"Stockfish rates {attempt.move.san} as {attempt.verdict}. It gives up about "
            f"{percent} percentage points of expected score."
        )
    elif attempt.centipawn_loss is not None:
        text = (
            f"{attempt.move.san} is not equivalent to the first choice: its checked "
            f"evaluation is {attempt.centipawn_loss} centipawns lower."
        )
    else:
        text = f"{attempt.move.san} is not equivalent to the engine's first choice."
    scope = "attempt_refutation" if attempt.line.role == "attempt_refutation" else "attempt_line"
    marker_role: ReviewMarkerRole = (
        "danger" if attempt.verdict in {"mistake", "blunder"} else "focus"
    )
    return ReviewAnnotation(
        id=annotation_id,
        label="Your move",
        text=text,
        scope=scope,
        ply=0,
        markers=[_destination_marker(attempt.move.uci, role=marker_role)],
        arrows=[_move_arrow(attempt.move.uci, role="played")],
        evidence_ids=[evidence_id],
    )


def _evidence_arrows(
    finding: DetectedSubject | None,
    *,
    scope: _FindingScope = "best_line",
) -> list[ReviewArrow]:
    if finding is None or not finding.evidence:
        return []
    evidence = finding.evidence[0]
    move_role: ReviewArrowRole = "reply" if scope == "attempt_refutation" else "engine"
    arrows: list[ReviewArrow] = []
    if evidence.from_square is not None and evidence.to_square is not None:
        arrows.append(
            ReviewArrow(
                from_square=evidence.from_square,
                to_square=evidence.to_square,
                role=move_role,
            )
        )
    elif evidence.moves:
        arrows.append(_move_arrow(evidence.moves[0], role=move_role))
    if finding.handler == "threat" and len(evidence.moves) > 1:
        arrows.append(_move_arrow(evidence.moves[-1], role="threat"))

    relation_role: ReviewArrowRole = (
        "ray" if finding.handler in {"pin", "xray", "discovered_attack"} else "attack"
    )
    geometric_handlers = {
        "check",
        "double_attack",
        "pin",
        "discovered_attack",
        "xray",
        "trapping",
        "chasing_targeting",
    }
    if evidence.actor is not None and finding.handler in geometric_handlers:
        for target in evidence.targets[:3]:
            if evidence.actor.square == target.square:
                continue
            relation = (evidence.actor.square, target.square)
            if any(
                arrow.from_square == relation[0] and arrow.to_square == relation[1]
                for arrow in arrows
            ):
                continue
            arrows.append(
                ReviewArrow(
                    from_square=relation[0],
                    to_square=relation[1],
                    role=relation_role,
                )
            )
            if len(arrows) == 4:
                break
    return arrows


def _evidence_badge(
    finding: DetectedSubject,
    arrows: tuple[ReviewArrow, ...],
    *,
    scope: _FindingScope,
) -> ReviewDiagramBadge | None:
    kind = _badge_for_finding(finding)
    square = _badge_square(finding, kind)
    if kind is None or square is None:
        return None
    arrow_index = _badge_arrow_index(finding, square, arrows)
    role: ReviewArrowRole = "engine"
    if finding.handler == "threat":
        role = "threat"
    elif scope == "attempt_refutation":
        role = "reply"
    return ReviewDiagramBadge(
        kind=kind,
        square=square,
        role=role,
        arrow_index=arrow_index,
    )


def _badge_arrow_index(
    finding: DetectedSubject,
    square: str,
    arrows: tuple[ReviewArrow, ...],
) -> int:
    preferred_roles: set[ReviewArrowRole]
    if finding.handler == "threat":
        preferred_roles = {"threat"}
    elif finding.handler in {
        "double_attack",
        "pin",
        "xray",
        "trapping",
        "discovered_attack",
    }:
        preferred_roles = {"attack", "ray"}
    else:
        preferred_roles = {"engine", "reply"}

    for require_preferred_role in (True, False):
        for index, arrow in enumerate(arrows):
            if require_preferred_role and arrow.role not in preferred_roles:
                continue
            if arrow.contains_square(square):
                return index
    raise RuntimeError("Review badge has no evidence-backed arrow")


def _evidence_markers(finding: DetectedSubject | None) -> list[ReviewSquareMarker]:
    if finding is None or not finding.evidence:
        return []
    evidence = finding.evidence[0]
    target_role: ReviewMarkerRole = (
        "danger" if finding.handler in {"trapping", "mate"} else "target"
    )
    markers = [
        ReviewSquareMarker(square=target.square, role=target_role)
        for target in evidence.targets[:3]
    ]
    if not markers and finding.handler in {
        "double_attack",
        "material",
        "mate",
        "pin",
        "threat",
        "trapping",
        "xray",
    }:
        markers.extend(
            ReviewSquareMarker(square=square, role=target_role) for square in evidence.squares[:3]
        )

    move_from, move_to = _evidence_move_squares(evidence)
    if finding.handler in {"clearing", "discovered_attack"} and move_from:
        markers.append(ReviewSquareMarker(square=move_from, role="vacated"))
    elif finding.handler == "interference":
        blocked = move_to or (evidence.squares[1] if len(evidence.squares) > 1 else None)
        if blocked:
            markers.append(ReviewSquareMarker(square=blocked, role="blocked"))
        if evidence.squares:
            markers.append(ReviewSquareMarker(square=evidence.squares[-1], role="target"))
    return _unique_markers(markers)


def _hint_markers(finding: DetectedSubject | None) -> list[ReviewSquareMarker]:
    if finding is None:
        return []
    evidence = finding.evidence[0]
    squares = (
        [target.square for target in evidence.targets]
        if evidence.targets
        else list(evidence.squares)
    )
    return [ReviewSquareMarker(square=square, role="focus") for square in dict.fromkeys(squares)]


def _move_arrow(
    uci: str,
    *,
    role: ReviewArrowRole,
) -> ReviewArrow:
    move = chess.Move.from_uci(uci)
    return ReviewArrow(
        from_square=chess.square_name(move.from_square),
        to_square=chess.square_name(move.to_square),
        role=role,
    )


def _destination_marker(uci: str, *, role: ReviewMarkerRole) -> ReviewSquareMarker:
    move = chess.Move.from_uci(uci)
    return ReviewSquareMarker(square=chess.square_name(move.to_square), role=role)


def _badge_for_finding(finding: DetectedSubject) -> ReviewBadge | None:
    evidence = finding.evidence[0]
    if finding.handler == "threat":
        return "mate" if evidence.kind == "mate_threat" else "capture"
    if finding.handler == "material":
        return "capture" if evidence.kind == "material_gain" else None
    badges: dict[str, ReviewBadge] = {
        "double_attack": "fork",
        "pin": "pin",
        "xray": "xray",
        "trapping": "trap",
        "eliminate_defence": "capture",
        "clearing": "clearance",
        "discovered_attack": "discovery",
        "interference": "interference",
        "luring": "attraction",
        "magnet": "attraction",
        "intermediate_move": "intermezzo",
        "mate": "mate",
        "mate_technique": "mate",
    }
    return badges.get(finding.handler)


def _badge_square(
    finding: DetectedSubject,
    badge: ReviewBadge | None,
) -> str | None:
    evidence = finding.evidence[0]
    move_from, move_to = _evidence_move_squares(evidence)
    if badge is None:
        return None
    if badge in {"fork", "engine"}:
        return move_to or (evidence.actor.square if evidence.actor else None)
    if badge in {"pin", "xray", "trap", "capture", "attraction"}:
        if evidence.targets:
            return evidence.targets[0].square
        if evidence.squares:
            if badge == "pin":
                return evidence.squares[-1]
            if badge == "xray" and len(evidence.squares) > 1:
                return evidence.squares[1]
            return evidence.squares[0]
    if badge in {"clearance", "discovery"}:
        return move_from
    return move_to


def _evidence_move_squares(evidence: Evidence) -> tuple[str | None, str | None]:
    if evidence.from_square is not None and evidence.to_square is not None:
        return evidence.from_square, evidence.to_square
    if not evidence.moves:
        return None, None
    move = chess.Move.from_uci(evidence.moves[0])
    return chess.square_name(move.from_square), chess.square_name(move.to_square)


def _unique_markers(markers: list[ReviewSquareMarker]) -> list[ReviewSquareMarker]:
    unique: list[ReviewSquareMarker] = []
    seen: set[tuple[str, ReviewMarkerRole]] = set()
    for marker in markers:
        key = (marker.square, marker.role)
        if key not in seen:
            seen.add(key)
            unique.append(marker)
    return unique


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
    centipawn_loss = best_value - attempted_value
    if centipawn_loss <= _NEAR_EQUAL_CENTIPAWNS:
        return expected_loss <= _NEAR_EQUAL_EXPECTED_LOSS
    return expected_loss <= _EQUIVALENT_EXPECTED_LOSS and centipawn_loss <= 50


def _attempt_verdict(
    *,
    attempted: str,
    best: str,
    expected_loss: float,
    equivalent: bool,
    lost_forced_mate: bool,
    score_deterioration: str | None,
    same_move_is_best: bool,
) -> str:
    if attempted == best and (same_move_is_best or equivalent):
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
            id="hint",
            label=topic.name,
            text=hint,
            scope="terminal",
            markers=[
                ReviewSquareMarker(square=square, role="danger" if is_mate else "focus")
                for square in king_squares
            ],
            evidence_ids=[evidence_id],
        ),
        explanation=[
            ReviewAnnotation(
                id="explanation-1",
                label="Final position",
                text=explanation,
                scope="terminal",
                markers=[
                    ReviewSquareMarker(square=square, role="danger" if is_mate else "focus")
                    for square in king_squares
                ],
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
