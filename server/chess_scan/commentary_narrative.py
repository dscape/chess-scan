"""Editorial coaching assembled only from checked review evidence and legal lines."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import chess

from chess_scan.chess_geometry import captured_piece_and_square, material_balance
from chess_scan.review_themes import SolutionStep, SolutionTrace
from chess_scan.schemas import (
    ENGINE_ONLY_EVIDENCE_KINDS,
    CoachingMoveSegment,
    CoachingSection,
    CoachingSegment,
    CoachingTextSegment,
    EngineScore,
    PositionReviewResponse,
    ReviewAnnotation,
    ReviewEvidenceResponse,
    ReviewLineResponse,
    review_line_for_scope,
)

CoachingFocus = Literal["cause", "concept", "comparison"]

_SAN_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"O-O(?:-O)?|"
    r"[KQRBN](?:[a-h1-8]{0,2})?x?[a-h][1-8](?:=[QRBN])?|"
    r"[a-h]x[a-h][1-8](?:=[QRBN])?|"
    r"[a-h][18]=[QRBN]"
    r")[+#]?(?![A-Za-z0-9])"
)
_QUIET_PAWN_REFERENCE = re.compile(r"(?<![A-Za-z0-9])[a-h][1-8](?:=[QRBN])?[+#]?(?![A-Za-z0-9])")
_PAWN_MOVE_CONTEXT = re.compile(
    r"(?:after|allows|before|beginning with|begins with|continues with|move is|"
    r"play|played|plays|reply is|then|with)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _MoveContext:
    captured_piece: chess.Piece | None
    captured_square: chess.Square | None
    gives_check: bool


def build_coaching_sections(
    review: PositionReviewResponse,
    lessons: Sequence[ReviewAnnotation],
    *,
    focus: CoachingFocus,
) -> list[CoachingSection]:
    if not lessons:
        return []

    evidence_by_id = {item.id: item for item in review.evidence}
    primary = lessons[0]
    primary_evidence = _primary_evidence(primary, evidence_by_id)
    sections = [_opening_section(review, primary, primary_evidence)]

    continuation = _continuation_section(review, primary, primary_evidence)
    alternative = _alternative_section(review)
    secondary_section: CoachingSection | None = None
    if len(lessons) > 1:
        secondary = lessons[1]
        secondary_section = _idea_section(
            review,
            secondary,
            _primary_evidence(secondary, evidence_by_id),
        )
    if focus == "concept" and secondary_section is not None:
        sections.append(secondary_section)
    practice = (
        _practice_section(review, primary, primary_evidence)
        if review.best_move is not None
        else None
    )
    if focus == "concept" and practice is not None:
        sections.append(practice)
    if focus == "comparison" and alternative is not None:
        sections.append(alternative)
        if continuation is not None:
            sections.append(continuation)
    else:
        if continuation is not None:
            sections.append(continuation)
        if alternative is not None:
            sections.append(alternative)

    if focus != "concept" and secondary_section is not None:
        sections.append(secondary_section)
    if focus != "concept" and practice is not None:
        sections.append(practice)
    return sections[:5]


def _opening_section(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> CoachingSection:
    evidence_ids = _section_evidence_ids(
        review,
        lesson,
        evidence,
        include_attempt=review.attempt is not None,
    )
    if review.attempt is None:
        move = _move_at_scope(
            review,
            _lesson_scope(lesson, evidence),
            lesson.ply,
            role="line",
        )
        segments = _lesson_segments(review, lesson, evidence, move)
        return CoachingSection(
            kind="diagnosis",
            title=_concept_title(lesson),
            segments=segments,
            evidence_ids=evidence_ids,
        )

    attempt = review.attempt
    capture = _first_relevant_capture(review, lesson, evidence)
    if capture is not None:
        ply, context = capture
        line = attempt.line
        captured_name = chess.piece_name(context.captured_piece.piece_type)
        captured_square = chess.square_name(context.captured_square)
        connection = " immediately allows " if ply == 1 else " leads to "
        segments = [
            _move(line, 0, role="attempt"),
            _text(connection),
            _move(line, ply, role="reply"),
            _text(
                f" to take the {captured_name} on {captured_square}. "
                "That forcing capture changes the material balance."
            ),
            *_score_comparison_segments(review),
        ]
        return CoachingSection(
            kind="diagnosis",
            title="Why the move loses material",
            segments=segments,
            evidence_ids=evidence_ids,
        )

    is_refutation = (
        attempt.verdict in {"mistake", "blunder"}
        and _lesson_scope(lesson, evidence) == "attempt_refutation"
    )
    selected_move = _move_at_scope(
        review,
        _lesson_scope(lesson, evidence),
        lesson.ply,
        role="reply" if is_refutation else "line",
    )
    if is_refutation and selected_move is not None and selected_move.ply > 0:
        copy, move_prefixed = _move_prefixed_copy(lesson.text, selected_move.move.san)
        segments = [
            _move(attempt.line, 0, role="attempt"),
            _text(" allows "),
            selected_move,
        ]
        if move_prefixed:
            segments.append(_text(", which "))
            segments.extend(
                _structured_copy_segments(
                    review,
                    lesson,
                    evidence,
                    copy,
                    start_ply=selected_move.ply + 1,
                )
            )
        else:
            segments.append(_text(". "))
            segments.extend(_structured_copy_segments(review, lesson, evidence, lesson.text))
    else:
        segments = _lesson_segments(review, lesson, evidence, selected_move)
    segments.extend(_score_comparison_segments(review))
    return CoachingSection(
        kind="diagnosis",
        title=_attempt_title(review, lesson, evidence),
        segments=segments,
        evidence_ids=evidence_ids,
    )


def _continuation_section(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> CoachingSection | None:
    line = _line_for_lesson(review, lesson, evidence)
    if line is None:
        line = review.attempt.line if review.attempt is not None else None
    if line is None:
        return None
    evidence_id = _engine_evidence_id(
        review,
        "engine_comparison"
        if line.role in {"attempt_line", "attempt_refutation"}
        else "engine_candidate",
    )
    segments: list[CoachingSegment] = [_text("The checked line runs ")]
    visible_plies = min(
        len(line.moves),
        max(5, lesson.ply + 1, _evidence_final_ply(line, evidence) + 1),
    )
    for ply in range(visible_plies):
        if ply:
            segments.append(_text(" → "))
        role: Literal["attempt", "reply", "line", "better"] = "line"
        if line.role in {"attempt_line", "attempt_refutation"}:
            role = "attempt" if ply == 0 else "reply" if ply == 1 else "line"
        segments.append(_move(line, ply, role=role))
    segments.append(_text("."))
    return CoachingSection(
        kind="continuation",
        title="The line to calculate",
        segments=segments,
        evidence_ids=_unique_ids([evidence_id, *lesson.evidence_ids]),
    )


def _alternative_section(review: PositionReviewResponse) -> CoachingSection | None:
    if review.attempt is None or not review.lines:
        return None
    line = review.lines[0]
    if review.attempt.move == line.moves[0]:
        return None
    engine_ids = [
        _engine_evidence_id(review, "engine_candidate"),
        _engine_evidence_id(review, "engine_comparison"),
    ]
    if review.attempt.equivalent:
        segments: list[CoachingSegment] = [
            _move(review.attempt.line, 0, role="attempt"),
            _text(" is effectively equivalent to Stockfish's first choice "),
            _move(line, 0, role="line"),
            _text("."),
        ]
        title = "Another sound approach"
    else:
        segments = [
            _text("Stockfish prefers "),
            _move(line, 0, role="better"),
            _text("."),
        ]
        title = "A better first move"
    if len(line.moves) > 1:
        segments.append(_text(" The checked continuation continues "))
        for ply in range(1, min(4, len(line.moves))):
            if ply > 1:
                segments.append(_text(" → "))
            segments.append(_move(line, ply, role="line"))
        segments.append(_text("."))
    return CoachingSection(
        kind="alternative",
        title=title,
        segments=segments,
        evidence_ids=_unique_ids(engine_ids),
    )


def _idea_section(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> CoachingSection:
    move = _move_at_scope(
        review,
        _lesson_scope(lesson, evidence),
        lesson.ply,
        role="line",
    )
    segments = _lesson_segments(review, lesson, evidence, move)
    line = _line_for_lesson(review, lesson, evidence)
    if line is not None and evidence.proof == "line_consequence":
        proof_plies = _evidence_line_plies(line, evidence)
        if len(proof_plies) > 1 or (proof_plies and proof_plies[-1] > lesson.ply):
            segments.append(_text(" Checked sequence: "))
            complete_proof_plies = range(proof_plies[0], proof_plies[-1] + 1)
            for index, ply in enumerate(complete_proof_plies):
                if index:
                    segments.append(_text(" → "))
                segments.append(_move(line, ply, role=_line_move_role(line, ply)))
            segments.append(_text("."))
    return CoachingSection(
        kind="idea",
        title=_concept_title(lesson),
        segments=segments,
        evidence_ids=_section_evidence_ids(
            review,
            lesson,
            evidence,
            include_attempt=False,
        ),
    )


def _practice_section(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> CoachingSection:
    evidence_ids = _section_evidence_ids(
        review,
        lesson,
        evidence,
        include_attempt=review.attempt is not None,
    )
    if review.attempt is not None and len(review.attempt.line.moves) > 1:
        reply_context = _move_context(review.fen, review.attempt.line, 1)
        attempt_move = _move(review.attempt.line, 0, role="attempt")
        reply_move = _move(review.attempt.line, 1, role="reply")
        if reply_context.gives_check:
            return CoachingSection(
                kind="practice",
                title="Start with forcing replies",
                segments=[
                    _text("After considering "),
                    attempt_move,
                    _text(", calculate every check first. The line begins with "),
                    reply_move,
                    _text("."),
                ],
                evidence_ids=evidence_ids,
            )
        if reply_context.captured_piece is not None:
            return CoachingSection(
                kind="practice",
                title="A captures-first check",
                segments=[
                    _text("Before committing to "),
                    attempt_move,
                    _text(", list every forcing capture for the opponent. Here, "),
                    reply_move,
                    _text(" is the reply to calculate before quieter plans."),
                ],
                evidence_ids=evidence_ids,
            )
        if review.attempt.lost_forced_mate or evidence.kind in {
            "answer_check",
            "check",
            "king_route_restricted",
        }:
            return CoachingSection(
                kind="practice",
                title="Keep king safety in the candidate list",
                segments=[
                    _text("Before choosing "),
                    attempt_move,
                    _text(", compare the opponent's forcing checks and king-entry squares."),
                ],
                evidence_ids=evidence_ids,
            )
        return CoachingSection(
            kind="practice",
            title="Test the opponent's strongest reply",
            segments=[
                _text("After choosing "),
                attempt_move,
                _text(", test the checked reply "),
                reply_move,
                _text(" before settling on your evaluation."),
            ],
            evidence_ids=evidence_ids,
        )

    move = _move_at_scope(
        review,
        _lesson_scope(lesson, evidence),
        lesson.ply,
        role="line",
    )
    segments = [_text("Name the concrete change created by the candidate move")]
    if move is not None:
        segments.extend([_text(" "), move])
    segments.append(_text(": checks, captures, threats, and newly controlled squares"))
    line = _line_for_lesson(review, lesson, evidence)
    if line is not None and lesson.ply + 1 < len(line.moves):
        segments.append(_text(", then test the opponent's checked response."))
    else:
        segments.append(_text("."))
    return CoachingSection(
        kind="practice",
        title="Turn the idea into a calculation habit",
        segments=segments,
        evidence_ids=evidence_ids,
    )


def _first_relevant_capture(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> tuple[int, _MoveContext] | None:
    if (
        review.attempt is None
        or review.attempt.verdict not in {"mistake", "blunder"}
        or _lesson_scope(lesson, evidence) != "attempt_refutation"
        or evidence.kind not in {"material_gain", "material_gain_line"}
    ):
        return None
    trace = _line_trace(review.fen, review.attempt.line)
    moving_side = trace.player
    if (
        material_balance(trace.steps[-1].after, moving_side)
        - material_balance(trace.initial, moving_side)
        >= 0
    ):
        return None
    preferred = lesson.ply
    plies = [preferred, *range(1, min(5, len(review.attempt.line.moves)))]
    seen: set[int] = set()
    for ply in plies:
        if ply in seen or not 0 <= ply < len(review.attempt.line.moves):
            continue
        seen.add(ply)
        context = _step_context(trace.steps[ply])
        if context.captured_piece is not None and context.captured_piece.color == moving_side:
            return ply, context
    return None


def _line_trace(fen: str, line: ReviewLineResponse) -> SolutionTrace:
    return SolutionTrace.build(chess.Board(fen), [move.uci for move in line.moves])


def _move_context(fen: str, line: ReviewLineResponse, ply: int) -> _MoveContext:
    return _step_context(_line_trace(fen, line).steps[ply])


def _step_context(step: SolutionStep) -> _MoveContext:
    captured_piece, captured_square = captured_piece_and_square(step.before, step.move)
    return _MoveContext(
        captured_piece=captured_piece,
        captured_square=captured_square if captured_piece is not None else None,
        gives_check=step.after.is_check(),
    )


def _score_comparison_segments(review: PositionReviewResponse) -> list[CoachingSegment]:
    if review.attempt is None or not review.lines:
        return []
    best_line = review.lines[0]
    if review.attempt.equivalent:
        if review.attempt.move == best_line.moves[0]:
            return [
                _text(" Stockfish keeps "),
                _move(best_line, 0, role="line"),
                _text(f" as its first choice at {_score_label(best_line.score, review.fen)}."),
            ]
        return [
            _text(" Stockfish treats "),
            _move(review.attempt.line, 0, role="attempt"),
            _text(" as effectively equivalent to "),
            _move(best_line, 0, role="line"),
            _text(" at this depth."),
        ]
    return [
        _text(" Stockfish's evaluation shifts from "),
        _move(best_line, 0, role="better"),
        _text(f" at {_score_label(best_line.score, review.fen)} to "),
        _move(review.attempt.line, 0, role="attempt"),
        _text(f" at {_score_label(review.attempt.line.score, review.fen)}."),
    ]


def _score_label(score: EngineScore, fen: str) -> str:
    value = score.value if chess.Board(fen).turn == chess.WHITE else -score.value
    if score.kind == "mate":
        side = "White" if value > 0 else "Black"
        return f"{side} mates in {abs(value)}"
    value /= 100
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}{abs(value):.1f}"


def _primary_evidence(
    lesson: ReviewAnnotation,
    evidence_by_id: dict[str, ReviewEvidenceResponse],
) -> ReviewEvidenceResponse:
    return next(
        evidence_by_id[evidence_id]
        for evidence_id in lesson.evidence_ids
        if evidence_by_id[evidence_id].kind not in ENGINE_ONLY_EVIDENCE_KINDS
    )


def _line_for_lesson(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> ReviewLineResponse | None:
    return review_line_for_scope(
        _lesson_scope(lesson, evidence),
        best_line=review.lines[0] if review.lines else None,
        attempt_line=review.attempt.line if review.attempt else None,
    )


def _evidence_line_plies(
    line: ReviewLineResponse,
    evidence: ReviewEvidenceResponse,
) -> list[int]:
    cursor = 0
    matched_plies: list[int] = []
    canonical_moves = [move.uci for move in line.moves]
    for evidence_move in evidence.moves:
        try:
            matched_ply = canonical_moves.index(evidence_move, cursor)
        except ValueError as error:
            if evidence.proof == "line_consequence":
                raise ValueError(
                    "Line-consequence evidence is absent from its checked line"
                ) from error
            return []
        matched_plies.append(matched_ply)
        cursor = matched_ply + 1
    return matched_plies


def _evidence_final_ply(
    line: ReviewLineResponse,
    evidence: ReviewEvidenceResponse,
) -> int:
    matched_plies = _evidence_line_plies(line, evidence)
    return max(evidence.ply, matched_plies[-1] if matched_plies else evidence.ply)


def _lesson_scope(
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> str:
    return evidence.scope if lesson.scope == "root" else lesson.scope


def _move_at_scope(
    review: PositionReviewResponse,
    scope: str,
    ply: int,
    *,
    role: Literal["attempt", "reply", "line", "better"],
) -> CoachingMoveSegment | None:
    line = review_line_for_scope(
        scope,
        best_line=review.lines[0] if review.lines else None,
        attempt_line=review.attempt.line if review.attempt else None,
    )
    if line is None or ply >= len(line.moves):
        return None
    return _move(line, ply, role=role)


def _move(
    line: ReviewLineResponse,
    ply: int,
    *,
    role: Literal["attempt", "reply", "line", "better"],
) -> CoachingMoveSegment:
    return CoachingMoveSegment(
        type="move",
        scope=_coaching_scope(line.role),
        ply=ply,
        role=role,
        move=line.moves[ply],
    )


def _coaching_scope(
    role: str,
) -> Literal["best_line", "attempt_line", "attempt_refutation"]:
    if role in {"best_candidate", "alternative_candidate"}:
        return "best_line"
    if role == "attempt_line":
        return "attempt_line"
    if role == "attempt_refutation":
        return "attempt_refutation"
    raise ValueError(f"Unsupported coaching line role: {role}")


def _attempt_title(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
) -> str:
    if review.attempt is None:
        return _concept_title(lesson)
    is_refutation = (
        review.attempt.verdict in {"mistake", "blunder"}
        and _lesson_scope(lesson, evidence) == "attempt_refutation"
    )
    if not is_refutation:
        return _concept_title(lesson)
    if review.attempt.lost_forced_mate:
        return "Why the move gives up the mating line"
    if evidence.kind in {"answer_check", "check", "king_route_restricted"}:
        return "The king-safety problem"
    return "What the move allows"


def _concept_title(lesson: ReviewAnnotation) -> str:
    label = lesson.label.split("·")[-1].strip().lower()
    titles = {
        "breakthrough": "The breakthrough",
        "clearance": "The point of the clearance",
        "create a threat": "The threat to calculate",
        "defence": "Meeting the immediate threat",
        "development": "Finish development",
        "discovered attack": "The discovered attack",
        "gain space": "Gaining space",
        "giving check": "A forcing check",
        "magnet": "The tactical magnet",
        "passed pawn": "The passed pawn",
        "pawn break": "The point of the pawn break",
        "piece activity": "The point of piece activity",
        "prophylaxis": "The prophylactic idea",
        "remove the defender": "Removing the defender",
        "saving the piece": "Meeting the immediate threat",
        "strong square": "The strong square",
        "support a pawn advance": "Preparing the pawn advance",
        "temporary sacrifice": "Why the temporary sacrifice works",
        "weak pawn": "The long-term target",
        "winning material": "The material sequence",
    }
    if label in titles:
        return titles[label]
    return f"The point of the {label}" if label else "The positional idea"


def _lesson_segments(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
    move: CoachingMoveSegment | None,
) -> list[CoachingSegment]:
    if move is None:
        return _structured_copy_segments(review, lesson, evidence, lesson.text)
    copy, move_prefixed = _move_prefixed_copy(lesson.text, move.move.san)
    if move_prefixed:
        return [
            move,
            _text(" "),
            *_structured_copy_segments(
                review,
                lesson,
                evidence,
                copy,
                start_ply=move.ply + 1,
            ),
        ]
    if any(san == move.move.san for _start, _end, san in _san_references(lesson.text)):
        return _structured_copy_segments(review, lesson, evidence, lesson.text)
    return [
        _text("After "),
        move,
        _text(", "),
        *_structured_copy_segments(
            review,
            lesson,
            evidence,
            _lower_first(lesson.text),
            start_ply=move.ply + 1,
        ),
    ]


def _structured_copy_segments(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
    copy: str,
    *,
    start_ply: int | None = None,
) -> list[CoachingSegment]:
    line = _line_for_lesson(review, lesson, evidence)
    line_plies: dict[str, list[int]] = {}
    if line is not None and evidence.proof != "counterfactual":
        for ply, reviewed_move in enumerate(line.moves):
            line_plies.setdefault(reviewed_move.san, []).append(ply)
    elif line is not None and lesson.ply < len(line.moves):
        anchor = line.moves[lesson.ply]
        line_plies[anchor.san] = [lesson.ply]

    segments: list[CoachingSegment] = []
    cursor = 0
    line_cursor = lesson.ply if start_ply is None else start_ply
    for start, end, san in _san_references(copy):
        _append_text(segments, copy[cursor:start])
        matching_plies = line_plies.get(san, [])
        ply = next((candidate for candidate in matching_plies if candidate >= line_cursor), None)
        if line is not None and ply is not None:
            segments.append(_move(line, ply, role=_line_move_role(line, ply)))
            line_cursor = ply + 1
        elif evidence.proof == "counterfactual":
            _append_text(segments, _described_move(san, evidence.proof))
        else:
            raise ValueError(f"Lesson SAN {san} is absent from its checked line")
        cursor = end
    _append_text(segments, copy[cursor:])
    return segments or [_text("The checked idea changes the position.")]


def _line_move_role(
    line: ReviewLineResponse,
    ply: int,
) -> Literal["attempt", "reply", "line", "better"]:
    if line.role not in {"attempt_line", "attempt_refutation"}:
        return "line"
    return "attempt" if ply == 0 else "reply" if ply == 1 else "line"


def _described_move(san: str, proof: str) -> str:
    move = san.rstrip("+#")
    prefix = "the counterfactual " if proof == "counterfactual" else "a "
    if move.startswith("O-O"):
        return f"{prefix}castling move"
    destination = re.search(r"([a-h][1-8])(?:=[QRBN])?$", move)
    if destination is None:
        return f"{prefix}cited move"
    piece_names = {
        "K": "king",
        "Q": "queen",
        "R": "rook",
        "B": "bishop",
        "N": "knight",
    }
    piece = piece_names.get(move[0], "pawn")
    action = "capture on" if "x" in move else "move to"
    return f"{prefix}{piece} {action} {destination.group(1)}"


def _append_text(segments: list[CoachingSegment], value: str) -> None:
    if not value:
        return
    if segments and isinstance(segments[-1], CoachingTextSegment):
        segments[-1] = _text(segments[-1].text + value)
    else:
        segments.append(_text(value))


def unstructured_san_references(text: str) -> tuple[str, ...]:
    return tuple(san for _start, _end, san in _san_references(text))


def _san_references(text: str) -> list[tuple[int, int, str]]:
    references = [
        (match.start(), match.end(), match.group(0)) for match in _SAN_REFERENCE.finditer(text)
    ]
    occupied = [(start, end) for start, end, _san in references]
    for match in _QUIET_PAWN_REFERENCE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        san = match.group(0)
        before = text[: match.start()].rstrip()
        if (
            not san.endswith(("+", "#"))
            and "=" not in san
            and match.start() > 0
            and not (before and before[-1] in ".:;!?")
            and _PAWN_MOVE_CONTEXT.search(before) is None
        ):
            continue
        references.append((match.start(), match.end(), san))
    return sorted(references)


def _move_prefixed_copy(text: str, san: str) -> tuple[str, bool]:
    stripped = text.strip()
    references = _san_references(stripped)
    if not references or references[0] != (0, len(san), san):
        return stripped, False
    stripped = stripped[len(san) :].lstrip()
    if not stripped:
        return "changes the position.", True
    return _lower_first(stripped), True


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def _section_evidence_ids(
    review: PositionReviewResponse,
    lesson: ReviewAnnotation,
    evidence: ReviewEvidenceResponse,
    *,
    include_attempt: bool,
) -> list[str]:
    evidence_ids = list(lesson.evidence_ids)
    scope = _lesson_scope(lesson, evidence)
    if scope == "best_line":
        evidence_ids.append(_engine_evidence_id(review, "engine_candidate"))
    elif scope in {"attempt_line", "attempt_refutation"}:
        evidence_ids.append(_engine_evidence_id(review, "engine_comparison"))
    if include_attempt:
        evidence_ids.extend(
            [
                _engine_evidence_id(review, "engine_candidate"),
                _engine_evidence_id(review, "engine_comparison"),
            ]
        )
    return _unique_ids(evidence_ids)


def _engine_evidence_id(review: PositionReviewResponse, kind: str) -> str:
    return next(item.id for item in review.evidence if item.kind == kind)


def _unique_ids(evidence_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(evidence_ids))


def _text(value: str) -> CoachingTextSegment:
    return CoachingTextSegment(type="text", text=value)
