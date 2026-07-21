"""Deterministic chess evidence for a single-position review.

The detectors never repair a position and never infer prose. They validate the engine line with
python-chess and emit only facts that can be tied to pieces, squares, and legal moves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import chess

from chess_scan.review_themes import (
    PieceRef,
    Proof,
    ThemeEvidence,
    detect_solution_themes,
)

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: str
    summary: str
    squares: tuple[str, ...] = ()
    moves: tuple[str, ...] = ()
    proof: Proof = "legal_geometry"
    ply: int = 0
    actor: PieceRef | None = None
    targets: tuple[PieceRef, ...] = ()
    from_square: str | None = None
    to_square: str | None = None

    def __post_init__(self) -> None:
        if self.proof not in {"legal_geometry", "line_consequence"}:
            raise ValueError(f"Unknown evidence proof: {self.proof}")
        for square in (*self.squares, self.from_square, self.to_square):
            if square is not None:
                _validate_evidence_square(square)
        if (self.from_square is None) != (self.to_square is None):
            raise ValueError("Evidence move geometry requires both endpoint squares")
        for move in self.moves:
            try:
                parsed = chess.Move.from_uci(move)
            except ValueError as exc:
                raise ValueError(f"Evidence contains an invalid move: {move}") from exc
            if parsed == chess.Move.null() or parsed.drop is not None:
                raise ValueError(f"Evidence contains an invalid move: {move}")


@dataclass(frozen=True, slots=True)
class DetectedSubject:
    handler: str
    confidence: float
    evidence: tuple[Evidence, ...]
    validated_theme: bool = False


@dataclass(frozen=True, slots=True)
class AnalyzedLine:
    moves: tuple[chess.Move, ...]
    boards: tuple[chess.Board, ...]


@dataclass(frozen=True, slots=True)
class ReviewContext:
    board: chess.Board
    line: AnalyzedLine

    @property
    def move(self) -> chess.Move:
        return self.line.moves[0]

    @property
    def after(self) -> chess.Board:
        return self.line.boards[0]

    @property
    def mover(self) -> chess.Color:
        return self.board.turn


def build_analyzed_line(
    board: chess.Board,
    uci_moves: list[str],
) -> AnalyzedLine:
    current = board.copy(stack=False)
    moves: list[chess.Move] = []
    boards: list[chess.Board] = []
    for uci in uci_moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"Engine line contains an invalid move: {uci}") from exc
        if move not in current.legal_moves:
            raise ValueError(f"Engine line contains an illegal move: {uci}")
        current.push(move)
        moves.append(move)
        boards.append(current.copy(stack=False))
    if not moves:
        raise ValueError("Engine line must contain at least one legal move")
    return AnalyzedLine(tuple(moves), tuple(boards))


def detect_subjects(
    context: ReviewContext,
    *,
    theme_evidence: Sequence[ThemeEvidence] | None = None,
) -> tuple[DetectedSubject, ...]:
    return _detect_subjects(
        context,
        DETECTOR_REGISTRY,
        theme_evidence=theme_evidence,
        include_legacy_themes=True,
    )


def teaching_subjects(
    context: ReviewContext,
    *,
    theme_evidence: Sequence[ThemeEvidence] | None = None,
) -> tuple[DetectedSubject, ...]:
    return _detect_subjects(
        context,
        TEACHING_DETECTOR_REGISTRY,
        theme_evidence=theme_evidence,
        include_legacy_themes=False,
    )


def detect_primary_subject(context: ReviewContext) -> DetectedSubject | None:
    findings = teaching_subjects(context)
    return findings[0] if findings else None


_CONTEXT_ONLY_HANDLERS = {
    "material_advantage",
    "mate_technique",
    "mobility",
    "pawn_endgame",
    "pawn_race",
    "pawn_square",
    "queen_endgame",
    "queen_pawn",
    "rook_endgame",
    "rook_pawn",
    "wrong_bishop",
}


_THEME_HANDLERS = {
    "double_attack",
    "pin",
    "eliminate_defence",
    "discovered_attack",
    "xray",
    "intermediate_move",
    "trapping",
    "interference",
    "luring",
    "magnet",
    "clearing",
}


def _detect_subjects(
    context: ReviewContext,
    registrations: Sequence[DetectorRegistration],
    *,
    theme_evidence: Sequence[ThemeEvidence] | None,
    include_legacy_themes: bool,
) -> tuple[DetectedSubject, ...]:
    best_by_handler = {
        finding.handler: finding
        for finding in _theme_findings(context, theme_evidence=theme_evidence)
    }
    for registration in registrations:
        handler, _detector = registration
        if handler in _THEME_HANDLERS and (not include_legacy_themes or handler in best_by_handler):
            continue
        finding = _run_detector(registration, context)
        if finding is None:
            continue
        previous = best_by_handler.get(finding.handler)
        if previous is None or finding.confidence > previous.confidence:
            best_by_handler[finding.handler] = finding
    return tuple(sorted(best_by_handler.values(), key=_subject_sort_key))


def _theme_findings(
    context: ReviewContext,
    *,
    theme_evidence: Sequence[ThemeEvidence] | None,
) -> tuple[DetectedSubject, ...]:
    evidence = (
        detect_solution_themes(context.board, context.line.moves)
        if theme_evidence is None
        else theme_evidence
    )
    return tuple(
        _theme_finding(context, item)
        for item in evidence
        if not _routine_opening_clearance(context, item)
    )


def _routine_opening_clearance(context: ReviewContext, evidence: ThemeEvidence) -> bool:
    if evidence.theme != "clearance" or evidence.ply != 0 or len(context.board.piece_map()) < 28:
        return False
    piece = context.board.piece_at(context.move.from_square)
    return (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_file(context.move.from_square) in {3, 4}
        and chess.square_rank(context.move.from_square) in {1, 6}
        and not context.board.is_capture(context.move)
        and not context.after.is_check()
    )


def _theme_finding(context: ReviewContext, evidence: ThemeEvidence) -> DetectedSubject:
    handler = _handler_for_theme(evidence)
    kind = evidence.theme
    if evidence.theme == "fork" and evidence.actor is not None:
        kind = f"two_targets_{evidence.actor.piece}"
    return DetectedSubject(
        handler=handler,
        confidence=1.0,
        evidence=(
            Evidence(
                kind=kind,
                summary=_theme_summary(context, evidence),
                squares=evidence.squares,
                moves=evidence.moves,
                proof=evidence.proof,
                ply=evidence.ply,
                actor=evidence.actor,
                targets=evidence.targets,
                from_square=evidence.from_square,
                to_square=evidence.to_square,
            ),
        ),
        validated_theme=True,
    )


def _handler_for_theme(evidence: ThemeEvidence) -> str:
    if evidence.theme == "attraction":
        attracted = evidence.targets[0] if evidence.targets else None
        return "magnet" if attracted is not None and attracted.piece == "king" else "luring"
    return {
        "capturingDefender": "eliminate_defence",
        "clearance": "clearing",
        "discoveredAttack": "discovered_attack",
        "fork": "double_attack",
        "interference": "interference",
        "intermezzo": "intermediate_move",
        "pin": "pin",
        "trappedPiece": "trapping",
        "xRayAttack": "xray",
    }[evidence.theme]


def _theme_summary(context: ReviewContext, evidence: ThemeEvidence) -> str:
    move = context.line.moves[evidence.ply]
    before = context.board if evidence.ply == 0 else context.line.boards[evidence.ply - 1]
    san = before.san(move)
    target_names = [_piece_label(target) for target in evidence.targets]
    if evidence.theme == "fork" and len(target_names) >= 2:
        return f"{san} attacks {target_names[0]} and {target_names[1]} at once."
    if evidence.theme == "pin" and target_names:
        return f"The piece on {evidence.targets[0].square} is pinned and cannot move freely."
    if evidence.theme == "discoveredAttack":
        return f"{san} clears a line for a discovered attack."
    if evidence.theme == "xRayAttack":
        return "The continuation exploits an attack through an intervening enemy piece."
    if evidence.theme == "intermezzo":
        return f"{san} inserts a forcing move before the available capture."
    if evidence.theme == "clearance":
        return f"{san} clears a square or line used by the continuation."
    if evidence.theme == "interference":
        return f"{san} interrupts a defensive line."
    if evidence.theme == "attraction" and target_names:
        return f"{san} draws {target_names[0]} onto the square needed for the follow-up."
    if evidence.theme == "capturingDefender" and len(target_names) >= 2:
        return f"{san} removes {target_names[0]}, leaving {target_names[1]} vulnerable."
    if evidence.theme == "trappedPiece" and target_names:
        return f"The continuation wins {target_names[0]}, which has no safe escape."
    raise RuntimeError(f"Unsupported theme evidence: {evidence.theme}")


def _piece_label(piece: PieceRef) -> str:
    article = "the"
    return f"{article} {piece.piece} on {piece.square}"


def _mate(context: ReviewContext) -> DetectedSubject | None:
    after = context.after
    if after.is_checkmate():
        return _finding(
            "mate",
            1.0,
            "mate",
            f"{context.board.san(context.move)} ends the game by checkmate.",
            moves=(context.move.uci(),),
            squares=(chess.square_name(context.move.to_square),),
        )
    if len(context.line.moves) >= 3 and context.line.boards[2].is_checkmate():
        return _finding(
            "mate",
            0.95,
            "mating_line",
            "The checked engine line reaches mate on the next move by the attacking side.",
            moves=tuple(move.uci() for move in context.line.moves[:3]),
        )
    return None


def _defence(context: ReviewContext) -> DetectedSubject | None:
    board, move, after = context.board, context.move, context.after
    san = board.san(move)
    if board.is_check() and not after.is_check():
        return _finding(
            "defence",
            1.0,
            "answer_check",
            f"{san} gets the king out of check.",
            moves=(move.uci(),),
        )

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        return None
    enemy = not context.mover
    attacked_before = board.is_attacked_by(enemy, move.from_square)
    attacked_after = after.is_attacked_by(enemy, move.to_square)
    if not attacked_before or attacked_after:
        return None
    attackers = board.attackers(enemy, move.from_square)
    method = "moving out of attack"
    if board.is_capture(move):
        method = "capturing the attacker" if move.to_square in attackers else "capturing"
    return _finding(
        "defence",
        0.9,
        "defensive_move",
        f"{san} saves an attacked {chess.piece_name(moving_piece.piece_type)} by {method}.",
        squares=(chess.square_name(move.from_square), chess.square_name(move.to_square)),
        moves=(move.uci(),),
    )


def _material(context: ReviewContext) -> DetectedSubject | None:
    board, move = context.board, context.move
    captured = _captured_piece(board, move)
    mover = board.piece_at(move.from_square)
    gain = _settled_material_gain(context)
    if gain <= 0:
        return None
    if captured is not None:
        defenders = board.attackers(captured.color, move.to_square)
        if not defenders:
            return _finding(
                "material",
                0.98,
                "material_gain",
                f"{board.san(move)} captures an unprotected "
                f"{chess.piece_name(captured.piece_type)}.",
                squares=(chess.square_name(move.to_square),),
                moves=(move.uci(),),
            )
        return _finding(
            "material",
            0.85,
            "material_gain_line",
            f"The checked line beginning with {board.san(move)} gains about {gain} "
            f"point{'s' if gain != 1 else ''} of material.",
            moves=tuple(move.uci() for move in context.line.moves),
        )
    if mover is not None and gain > 0:
        return _finding(
            "material",
            0.85,
            "material_gain_line",
            f"The checked line gains about {gain} point{'s' if gain != 1 else ''} of material.",
            moves=tuple(move.uci() for move in context.line.moves),
        )
    return None


def _double_attack(context: ReviewContext) -> DetectedSubject | None:
    after, move, mover = context.after, context.move, context.mover
    targets: list[tuple[chess.Square, chess.Piece]] = []
    for square in after.attacks(move.to_square):
        piece = after.piece_at(square)
        if piece is not None and piece.color != mover:
            targets.append((square, piece))
    valuable = [
        target
        for target in targets
        if target[1].piece_type == chess.KING or PIECE_VALUES[target[1].piece_type] >= 1
    ]
    if len(valuable) < 2:
        return None
    valuable.sort(
        key=lambda target: (
            target[1].piece_type != chess.KING,
            -PIECE_VALUES[target[1].piece_type],
            target[0],
        )
    )
    if (
        not any(piece.piece_type == chess.KING for _, piece in valuable)
        and sum(PIECE_VALUES[piece.piece_type] for _, piece in valuable) < 5
    ):
        return None
    target_squares = {square for square, _ in valuable}
    checking_fork = after.is_check() and any(
        piece.piece_type == chess.KING for _, piece in valuable
    )
    exploited = _line_captures_square(context, target_squares, start_index=2)
    if not checking_fork or not exploited:
        return None
    names = [
        "king" if piece.piece_type == chess.KING else chess.piece_name(piece.piece_type)
        for _, piece in valuable[:2]
    ]
    squares = tuple(chess.square_name(square) for square, _ in valuable[:2])
    moving_piece = after.piece_at(move.to_square)
    piece_name = chess.piece_name(moving_piece.piece_type) if moving_piece else "piece"
    return _finding(
        "double_attack",
        0.96,
        f"two_targets_{piece_name}",
        f"{context.board.san(move)} attacks the {names[0]} and the {names[1]} at once.",
        squares=squares,
        moves=(move.uci(),),
        actor=(
            _piece_reference(moving_piece, move.to_square) if moving_piece is not None else None
        ),
        targets=tuple(_piece_reference(piece, square) for square, piece in valuable[:2]),
    )


def _pin(context: ReviewContext) -> DetectedSubject | None:
    board, after, enemy = context.board, context.after, not context.mover
    new_pins = [
        square
        for square, piece in after.piece_map().items()
        if piece.color == enemy
        and piece.piece_type != chess.KING
        and after.is_pinned(enemy, square)
        and not board.is_pinned(enemy, square)
    ]
    if new_pins:
        square = new_pins[0]
        if not _line_captures_square(context, {square}, start_index=2):
            return None
        return _finding(
            "pin",
            0.96,
            "new_pin",
            f"{board.san(context.move)} pins the "
            f"{chess.piece_name(after.piece_type_at(square))} on "
            f"{chess.square_name(square)}.",
            squares=(chess.square_name(square),),
            moves=(context.move.uci(),),
        )

    attacked_pins = [
        square
        for square in after.attacks(context.move.to_square)
        if after.color_at(square) == enemy and after.is_pinned(enemy, square)
    ]
    if attacked_pins:
        square = attacked_pins[0]
        if not _line_captures_square(context, {square}, start_index=2):
            return None
        return _finding(
            "pin",
            0.92,
            "attack_pinned_piece",
            f"{board.san(context.move)} attacks the pinned piece on "
            f"{chess.square_name(square)} again.",
            squares=(chess.square_name(square),),
            moves=(context.move.uci(),),
        )
    return None


def _eliminate_defence(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3:
        return None
    board = context.board
    first, reply, follow_up = context.line.moves[:3]
    before_follow_up = context.line.boards[1]
    captured_target = _captured_piece(before_follow_up, follow_up)
    if captured_target is None or captured_target.color == context.mover:
        return None
    target_square = follow_up.to_square
    original_target = board.piece_at(target_square)
    after_first = context.line.boards[0]
    if original_target != captured_target or after_first.piece_at(target_square) != original_target:
        return None
    defenders = board.attackers(not context.mover, target_square)
    if not defenders:
        return None

    first_capture = _captured_piece(board, first)
    if first_capture is not None and first.to_square in defenders:
        method = "captures the defender"
        defender_square = first.to_square
    elif (
        reply.from_square in defenders
        and after_first.is_capture(reply)
        and reply.to_square == first.to_square
        and reply.to_square not in before_follow_up.attackers(not context.mover, target_square)
    ):
        method = "lures the defender away"
        defender_square = reply.from_square
    else:
        interfered = next(
            (
                defender
                for defender in defenders
                if _is_slider(board.piece_at(defender))
                and first.to_square in chess.SquareSet(chess.between(defender, target_square))
                and defender not in after_first.attackers(not context.mover, target_square)
            ),
            None,
        )
        if interfered is None:
            return None
        method = "interferes with the defender"
        defender_square = interfered
    return _finding(
        "eliminate_defence",
        0.94,
        "defender_removed",
        f"The first move {method}, so the "
        f"{chess.piece_name(captured_target.piece_type)} can be taken.",
        squares=(chess.square_name(defender_square), chess.square_name(target_square)),
        moves=(first.uci(), reply.uci(), follow_up.uci()),
    )


def _discovered_attack(context: ReviewContext) -> DetectedSubject | None:
    board, after, move = context.board, context.after, context.move
    if after.is_check():
        discovered_checkers = [
            square
            for square in after.checkers()
            if square != move.to_square and _is_slider(after.piece_at(square))
        ]
        if discovered_checkers:
            checker = discovered_checkers[0]
            return _finding(
                "discovered_attack",
                0.99,
                "discovered_check",
                f"Moving from {chess.square_name(move.from_square)} uncovers check from "
                f"{chess.square_name(checker)}.",
                squares=(chess.square_name(move.from_square), chess.square_name(checker)),
                moves=(move.uci(),),
            )

    for square, piece in after.piece_map().items():
        if piece.color != context.mover or square == move.to_square or not _is_slider(piece):
            continue
        before_targets = _enemy_targets(board, square, context.mover)
        after_targets = _enemy_targets(after, square, context.mover)
        new_targets = after_targets - before_targets
        if new_targets:
            target = next(iter(new_targets))
            if not _line_captures_square(context, {target}, start_index=2):
                continue
            return _finding(
                "discovered_attack",
                0.94,
                "uncovered_line",
                f"{board.san(move)} uncovers an attack from "
                f"{chess.square_name(square)} to {chess.square_name(target)}.",
                squares=(chess.square_name(square), chess.square_name(target)),
                moves=(move.uci(),),
            )
    return None


def _xray(context: ReviewContext) -> DetectedSubject | None:
    after, move = context.after, context.move
    piece = after.piece_at(move.to_square)
    if piece is None or not _is_slider(piece):
        return None
    pair = _first_two_enemy_pieces_on_ray(
        after,
        move.to_square,
        context.mover,
        piece.piece_type,
    )
    if pair is None or len(context.line.moves) < 3:
        return None
    front, rear = pair
    follow_board = context.line.boards[1]
    follow_up = context.line.moves[2]
    if (
        follow_up.from_square != move.to_square
        or not follow_board.is_capture(follow_up)
        or follow_up.to_square not in {front, rear}
    ):
        return None
    return _finding(
        "xray",
        0.9,
        "pieces_on_line",
        f"The pieces on {chess.square_name(front)} and {chess.square_name(rear)} "
        "stand on the same line as the attacking piece.",
        squares=(
            chess.square_name(move.to_square),
            chess.square_name(front),
            chess.square_name(rear),
        ),
        moves=(move.uci(),),
    )


def _intermediate_move(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3 or not context.after.is_check():
        return None
    board = context.board
    available_captures = [move for move in board.legal_moves if board.is_capture(move)]
    follow_up_board = context.line.boards[1]
    follow_up = context.line.moves[2]
    capture_targets = {move.to_square for move in available_captures}
    if (
        not available_captures
        or not follow_up_board.is_capture(follow_up)
        or follow_up.to_square not in capture_targets
    ):
        return None
    return _finding(
        "intermediate_move",
        0.86,
        "forcing_move_first",
        "A forcing check is played before returning to the available capture.",
        moves=tuple(move.uci() for move in context.line.moves[:3]),
    )


def _promotion(context: ReviewContext) -> DetectedSubject | None:
    move = context.move
    if move.promotion is None:
        return None
    if move.promotion == chess.QUEEN:
        return None
    queen_promotion = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    if queen_promotion not in context.board.legal_moves:
        return None
    queen_board = context.board.copy(stack=False)
    queen_board.push(queen_promotion)
    chosen_is_distinct = (context.after.is_check() and not queen_board.is_check()) or (
        not context.after.is_stalemate() and queen_board.is_stalemate()
    )
    if not chosen_is_distinct:
        return None
    promoted = chess.piece_name(move.promotion)
    summary = f"{context.board.san(move)} uses underpromotion to a {promoted}."
    return _finding(
        "promotion",
        1.0,
        "promotion",
        summary,
        squares=(chess.square_name(move.to_square),),
        moves=(move.uci(),),
    )


def _threat(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3:
        return None
    future_board = context.line.boards[1]
    future_move = context.line.moves[2]
    if future_board.is_capture(future_move):
        captured = _captured_piece(future_board, future_move)
        target_square = future_move.to_square
        target_after_first = context.after.piece_at(target_square)
        newly_attacked = (
            target_after_first is not None
            and target_after_first == captured
            and target_after_first.color != context.mover
            and context.after.is_attacked_by(context.mover, target_square)
            and not context.board.is_attacked_by(context.mover, target_square)
        )
        if captured and PIECE_VALUES[captured.piece_type] >= 3 and newly_attacked:
            return _finding(
                "threat",
                0.82,
                "material_threat",
                f"The move creates a threat to win the "
                f"{chess.piece_name(captured.piece_type)} on "
                f"{chess.square_name(future_move.to_square)}.",
                squares=(chess.square_name(future_move.to_square),),
                moves=(context.move.uci(), future_move.uci()),
            )
    if context.line.boards[2].is_checkmate():
        return _finding(
            "threat",
            0.95,
            "mate_threat",
            "The first move creates a mating threat.",
            moves=(context.move.uci(), future_move.uci()),
        )
    return None


def _trapping(context: ReviewContext) -> DetectedSubject | None:
    after = context.after
    opponent = after.turn
    attacked = after.attacks(context.move.to_square)
    legal_moves = tuple(after.legal_moves)
    for square in attacked:
        piece = after.piece_at(square)
        if piece is None or piece.color != opponent or PIECE_VALUES[piece.piece_type] < 3:
            continue
        escapes = [move for move in legal_moves if move.from_square == square]
        if len(escapes) <= 1 and _line_captures_square(context, {square}, start_index=2):
            return _finding(
                "trapping",
                0.86,
                "restricted_mobility",
                f"The {chess.piece_name(piece.piece_type)} on "
                f"{chess.square_name(square)} is attacked and has {len(escapes)} legal "
                f"escape square{'s' if len(escapes) != 1 else ''}.",
                squares=(chess.square_name(square),),
                moves=(context.move.uci(),),
            )
    return None


def _interference(context: ReviewContext) -> DetectedSubject | None:
    board, after, move = context.board, context.after, context.move
    if board.is_capture(move):
        return None
    enemy = not context.mover
    for target, target_piece in board.piece_map().items():
        if target_piece.color != enemy:
            continue
        for defender in board.attackers(enemy, target):
            defender_piece = board.piece_at(defender)
            if not _is_slider(defender_piece):
                continue
            between = chess.SquareSet(chess.between(defender, target))
            if (
                move.to_square in between
                and defender not in after.attackers(enemy, target)
                and _line_captures_square(context, {target}, start_index=2)
            ):
                return _finding(
                    "interference",
                    0.9,
                    "line_blocked",
                    f"{board.san(move)} breaks the defensive line to {chess.square_name(target)}.",
                    squares=(
                        chess.square_name(defender),
                        chess.square_name(move.to_square),
                        chess.square_name(target),
                    ),
                    moves=(move.uci(),),
                )
    return None


def _luring(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3:
        return None
    first, reply, follow_up = context.line.moves[:3]
    reply_board = context.line.boards[0]
    if not reply_board.is_capture(reply) or reply.to_square != first.to_square:
        return None
    reply_piece = reply_board.piece_at(reply.from_square)
    final_board = context.line.boards[1]
    captured = _captured_piece(final_board, follow_up)
    direct_capture = captured is not None and follow_up.to_square == reply.to_square
    king_follow_up = (
        reply_piece is not None
        and reply_piece.piece_type == chess.KING
        and final_board.gives_check(follow_up)
    )
    defender_lured = captured is not None and reply.from_square in context.board.attackers(
        not context.mover, follow_up.to_square
    )
    if not (direct_capture or king_follow_up or defender_lured):
        return None
    return _finding(
        "luring",
        0.9,
        "piece_lured",
        "The first move lures the replying piece to "
        f"{chess.square_name(reply.to_square)} before the follow-up.",
        squares=(chess.square_name(reply.to_square),),
        moves=(first.uci(), reply.uci(), follow_up.uci()),
    )


def _magnet(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3:
        return None
    first = context.move
    reply_board = context.line.boards[0]
    reply = context.line.moves[1]
    replying_piece = reply_board.piece_at(reply.from_square)
    if (
        replying_piece is None
        or replying_piece.piece_type != chess.KING
        or not reply_board.is_capture(reply)
        or reply.to_square != first.to_square
    ):
        return None
    follow_board = context.line.boards[1]
    follow_up = context.line.moves[2]
    if not (follow_board.is_capture(follow_up) or follow_board.gives_check(follow_up)):
        return None
    return _finding(
        "magnet",
        0.92,
        "king_lured",
        f"The sacrifice draws the king to {chess.square_name(reply.to_square)} for the follow-up.",
        squares=(chess.square_name(reply.to_square),),
        moves=tuple(move.uci() for move in context.line.moves[:3]),
    )


def _blocking(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3 or not context.line.boards[2].is_checkmate():
        return None
    reply_board = context.line.boards[0]
    reply = context.line.moves[1]
    piece = reply_board.piece_at(reply.from_square)
    defending_king = reply_board.king(reply_board.turn)
    if piece is None or piece.piece_type == chess.KING or defending_king is None:
        return None
    king_escape = chess.Move(defending_king, reply.to_square)
    if king_escape not in reply_board.legal_moves:
        return None
    return _finding(
        "blocking",
        0.86,
        "blocked_flight_square",
        f"The reply places a piece on {chess.square_name(reply.to_square)}, "
        "restricting its own king before mate.",
        squares=(chess.square_name(reply.to_square), chess.square_name(defending_king)),
        moves=tuple(move.uci() for move in context.line.moves[:3]),
    )


def _chasing_targeting(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3:
        return None
    first, reply, follow_up = context.line.moves[:3]
    after = context.after
    attacked = after.attacks(first.to_square)
    if reply.from_square not in attacked:
        return None
    reply_piece = after.piece_at(reply.from_square)
    if reply_piece is None or reply.to_square == reply.from_square:
        return None
    follow_board = context.line.boards[1]
    if follow_up.from_square != first.to_square or not (
        follow_board.is_capture(follow_up) or follow_board.gives_check(follow_up)
    ):
        return None
    return _finding(
        "chasing_targeting",
        0.84,
        "piece_chased",
        f"The first move chases the {chess.piece_name(reply_piece.piece_type)} "
        "before the next target is attacked.",
        squares=(chess.square_name(reply.from_square), chess.square_name(reply.to_square)),
        moves=(first.uci(), reply.uci(), follow_up.uci()),
    )


def _clearing(context: ReviewContext) -> DetectedSubject | None:
    if len(context.line.moves) < 3:
        return None
    first, _, follow_up = context.line.moves[:3]
    follow_board = context.line.boards[1]
    if not (follow_board.is_capture(follow_up) or follow_board.gives_check(follow_up)):
        return None
    if follow_up.to_square == first.from_square:
        return _finding(
            "clearing",
            0.9,
            "square_cleared",
            f"The first move clears {chess.square_name(first.from_square)} for the follow-up.",
            squares=(chess.square_name(first.from_square),),
            moves=(first.uci(), follow_up.uci()),
        )
    follow_piece = follow_board.piece_at(follow_up.from_square)
    if _is_slider(follow_piece) and first.from_square in chess.SquareSet(
        chess.between(follow_up.from_square, follow_up.to_square)
    ):
        return _finding(
            "clearing",
            0.88,
            "line_cleared",
            f"The first move clears the line used by {follow_board.san(follow_up)}.",
            squares=(chess.square_name(first.from_square),),
            moves=(first.uci(), follow_up.uci()),
        )
    return None


def _breakthrough(context: ReviewContext) -> DetectedSubject | None:
    piece = context.board.piece_at(context.move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return None
    before_passed = _passed_pawns(context.board, context.mover)
    after_passed = _passed_pawns(context.after, context.mover)
    created = after_passed - before_passed
    if not created:
        return None
    square = next(iter(created))
    return _finding(
        "breakthrough",
        0.9,
        "passed_pawn_created",
        f"{context.board.san(context.move)} creates a passed pawn on {chess.square_name(square)}.",
        squares=(chess.square_name(square),),
        moves=(context.move.uci(),),
    )


def _check(context: ReviewContext) -> DetectedSubject | None:
    if not context.after.is_check():
        return None
    return _finding(
        "check",
        1.0,
        "check",
        f"{context.board.san(context.move)} gives check.",
        moves=(context.move.uci(),),
        squares=(chess.square_name(context.move.to_square),),
    )


def _draw(context: ReviewContext) -> DetectedSubject | None:
    board = context.board
    if board.is_stalemate():
        summary = "The position is stalemate."
    elif board.is_insufficient_material():
        summary = "The position is drawn because there is insufficient mating material."
    elif context.after.is_stalemate():
        summary = f"{board.san(context.move)} produces stalemate."
    elif context.after.is_insufficient_material():
        summary = f"{board.san(context.move)} leaves insufficient mating material."
    else:
        return None
    return _finding("draw", 1.0, "draw", summary, moves=(context.move.uci(),))


def _passed_pawn(context: ReviewContext) -> DetectedSubject | None:
    passed = _passed_pawns(context.after, context.mover)
    if context.move.to_square not in passed:
        return None
    square = context.move.to_square
    return _finding(
        "passed_pawn",
        0.88,
        "passed_pawn",
        f"The pawn on {chess.square_name(square)} has no opposing pawn blocking "
        "or controlling its path to promotion.",
        squares=(chess.square_name(square),),
    )


def _activity(context: ReviewContext) -> DetectedSubject | None:
    piece = context.board.piece_at(context.move.from_square)
    if piece is None:
        return None
    before = len(context.board.attacks(context.move.from_square))
    after = len(context.after.attacks(context.move.to_square))
    if after < before + 3:
        return None
    return _finding(
        "activity",
        0.72,
        "mobility_increase",
        f"The moved {chess.piece_name(piece.piece_type)} increases its reach "
        f"from {before} to {after} squares.",
        squares=(
            chess.square_name(context.move.from_square),
            chess.square_name(context.move.to_square),
        ),
        moves=(context.move.uci(),),
    )


def _opening(context: ReviewContext) -> DetectedSubject | None:
    board = context.board
    if len(board.piece_map()) < 24 or board.fullmove_number > 20:
        return None
    piece = board.piece_at(context.move.from_square)
    starting_squares = (
        {chess.B1, chess.G1, chess.C1, chess.F1}
        if context.mover == chess.WHITE
        else {chess.B8, chess.G8, chess.C8, chess.F8}
    )
    central_pawn = (
        piece is not None
        and piece.piece_type == chess.PAWN
        and context.move.from_square
        in ({chess.D2, chess.E2} if context.mover == chess.WHITE else {chess.D7, chess.E7})
    )
    developing_piece = (
        piece is not None
        and piece.piece_type in {chess.KNIGHT, chess.BISHOP}
        and context.move.from_square in starting_squares
    )
    if not central_pawn and not developing_piece:
        return None
    summary = (
        f"{board.san(context.move)} claims central space and opens lines for development."
        if central_pawn
        else (
            f"{board.san(context.move)} develops the "
            f"{chess.piece_name(piece.piece_type)} from its starting square."
        )
    )
    return _finding(
        "opening",
        0.9,
        "center_control" if central_pawn else "development",
        summary,
        squares=(
            chess.square_name(context.move.from_square),
            chess.square_name(context.move.to_square),
        ),
        moves=(context.move.uci(),),
        actor=_piece_reference(piece, context.move.to_square),
    )


def _mate_technique(context: ReviewContext) -> DetectedSubject | None:
    pieces = context.board.piece_map().values()
    non_kings = [piece for piece in pieces if piece.piece_type != chess.KING]
    if len(non_kings) != 1 or non_kings[0].piece_type not in {chess.QUEEN, chess.ROOK}:
        return None
    name = chess.piece_name(non_kings[0].piece_type)
    return _finding(
        "mate_technique",
        0.95,
        f"{name}_mate",
        f"This is a king-and-{name} mating position.",
    )


def _pawn_endgame(context: ReviewContext) -> DetectedSubject | None:
    non_pawns = [
        piece
        for piece in context.board.piece_map().values()
        if piece.piece_type not in {chess.KING, chess.PAWN}
    ]
    if non_pawns:
        return None
    return _finding("pawn_endgame", 0.98, "pawn_endgame", "Only kings and pawns remain.")


def _pawn_square(context: ReviewContext) -> DetectedSubject | None:
    pawns = list(context.board.pieces(chess.PAWN, context.mover))
    enemy_king = context.board.king(not context.mover)
    if len(pawns) != 1 or enemy_king is None:
        return None
    pawn = pawns[0]
    moves_to_promote = 7 - chess.square_rank(pawn) if context.mover else chess.square_rank(pawn)
    distance = chess.square_distance(enemy_king, pawn)
    return _finding(
        "pawn_square",
        0.78,
        "pawn_race_distance",
        f"The pawn is {moves_to_promote} rank "
        f"move{'s' if moves_to_promote != 1 else ''} from promotion and the "
        f"opposing king is {distance} king move{'s' if distance != 1 else ''} away.",
        squares=(chess.square_name(pawn), chess.square_name(enemy_king)),
    )


def _mobility(context: ReviewContext) -> DetectedSubject | None:
    side = context.mover
    own = _pseudo_mobility(context.after, side)
    enemy = _pseudo_mobility(context.after, not side)
    if own <= enemy:
        return None
    return _finding(
        "mobility",
        0.65,
        "mobility",
        f"The active side's pieces reach {own} squares compared with {enemy} for the opponent.",
    )


def _rook_pawn(context: ReviewContext) -> DetectedSubject | None:
    pawns = context.board.pieces(chess.PAWN, chess.WHITE) | context.board.pieces(
        chess.PAWN, chess.BLACK
    )
    rook_pawns = [square for square in pawns if chess.square_file(square) in {0, 7}]
    if context.move.from_square not in rook_pawns:
        return None
    return _finding(
        "rook_pawn",
        0.7,
        "rook_pawn",
        "A rook pawn is a central feature of this position.",
        squares=(
            chess.square_name(context.move.from_square),
            chess.square_name(context.move.to_square),
        ),
    )


def _weak_pawn(context: ReviewContext) -> DetectedSubject | None:
    weaknesses: list[tuple[chess.Square, str, chess.Color]] = []
    for color in chess.COLORS:
        pawns = context.board.pieces(chess.PAWN, color)
        files = [chess.square_file(square) for square in pawns]
        for square in pawns:
            file = chess.square_file(square)
            if files.count(file) > 1:
                weaknesses.append((square, "doubled", color))
            elif not any(abs(other - file) == 1 for other in files):
                weaknesses.append((square, "isolated", color))
    targeted = [
        weakness
        for weakness in weaknesses
        if weakness[2] != context.mover
        and context.after.is_attacked_by(context.mover, weakness[0])
        and not context.board.is_attacked_by(context.mover, weakness[0])
    ]
    if not targeted:
        return None
    square, kind, _ = targeted[0]
    return _finding(
        "weak_pawn",
        0.72,
        "weak_pawn",
        f"The pawn on {chess.square_name(square)} is {kind}.",
        squares=(chess.square_name(square),),
    )


def _material_advantage(context: ReviewContext) -> DetectedSubject | None:
    balance = _material_balance(context.board, context.mover)
    if balance < 2:
        return None
    return _finding(
        "material_advantage",
        0.9,
        "material_balance",
        f"The moving side has a material advantage of about {balance} points.",
    )


def _king_attack(context: ReviewContext) -> DetectedSubject | None:
    enemy_king = context.after.king(not context.mover)
    if enemy_king is None:
        return None
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king])
    attacked_before = sum(context.board.is_attacked_by(context.mover, square) for square in ring)
    attacked = sum(context.after.is_attacked_by(context.mover, square) for square in ring)
    if (attacked < 3 or attacked <= attacked_before) and not context.after.is_check():
        return None
    return _finding(
        "king_attack",
        min(0.95, 0.62 + attacked * 0.07),
        "king_ring_pressure",
        f"{attacked} squares around the opposing king are attacked.",
        squares=(chess.square_name(enemy_king),),
    )


def _seventh_rank(context: ReviewContext) -> DetectedSubject | None:
    target_rank = 6 if context.mover == chess.WHITE else 1
    rooks = [
        square
        for square in context.after.pieces(chess.ROOK, context.mover)
        if chess.square_rank(square) == target_rank
    ]
    if not rooks or context.move.to_square not in rooks:
        return None
    square = context.move.to_square
    return _finding(
        "seventh_rank",
        0.88,
        "rook_seventh_rank",
        f"The rook on {chess.square_name(square)} has reached the seventh rank.",
        squares=(chess.square_name(square),),
    )


def _queen_pawn(context: ReviewContext) -> DetectedSubject | None:
    pieces = [
        piece for piece in context.board.piece_map().values() if piece.piece_type != chess.KING
    ]
    queens = [piece for piece in pieces if piece.piece_type == chess.QUEEN]
    if len(queens) != 1 or any(
        piece.piece_type not in {chess.QUEEN, chess.PAWN} for piece in pieces
    ):
        return None
    if not context.board.pieces(chess.PAWN, not queens[0].color):
        return None
    return _finding("queen_pawn", 0.95, "queen_against_pawn", "A queen is playing against pawns.")


def _pawn_race(context: ReviewContext) -> DetectedSubject | None:
    white = _most_advanced_pawn(context.board, chess.WHITE)
    black = _most_advanced_pawn(context.board, chess.BLACK)
    if white is None or black is None:
        return None
    white_distance = 7 - chess.square_rank(white)
    black_distance = chess.square_rank(black)
    if max(white_distance, black_distance) > 4:
        return None
    return _finding(
        "pawn_race",
        0.82,
        "pawn_race",
        "Both sides have advanced passed-pawn candidates: "
        f"{chess.square_name(white)} and {chess.square_name(black)}.",
        squares=(chess.square_name(white), chess.square_name(black)),
    )


def _strong_square(context: ReviewContext) -> DetectedSubject | None:
    enemy = not context.mover
    for square, piece in context.after.piece_map().items():
        if piece.color != context.mover or piece.piece_type not in {chess.KNIGHT, chess.BISHOP}:
            continue
        rank = chess.square_rank(square)
        if (context.mover and rank < 4) or (not context.mover and rank > 3):
            continue
        own_pawn_support = any(
            context.after.piece_type_at(attacker) == chess.PAWN
            for attacker in context.after.attackers(context.mover, square)
        )
        enemy_pawn_can_chase = any(
            abs(chess.square_file(pawn) - chess.square_file(square)) == 1
            and (
                chess.square_rank(pawn) < rank
                if enemy == chess.WHITE
                else chess.square_rank(pawn) > rank
            )
            for pawn in context.after.pieces(chess.PAWN, enemy)
        )
        if square == context.move.to_square and own_pawn_support and not enemy_pawn_can_chase:
            return _finding(
                "strong_square",
                0.82,
                "outpost",
                f"The {chess.piece_name(piece.piece_type)} on "
                f"{chess.square_name(square)} is supported by a pawn and cannot be "
                "chased by an enemy pawn.",
                squares=(chess.square_name(square),),
            )
    return None


def _open_file(context: ReviewContext) -> DetectedSubject | None:
    pawn_files = {
        chess.square_file(square)
        for color in chess.COLORS
        for square in context.board.pieces(chess.PAWN, color)
    }
    for square in context.after.pieces(chess.ROOK, context.mover):
        file = chess.square_file(square)
        if square == context.move.to_square and file not in pawn_files:
            return _finding(
                "open_file",
                0.86,
                "open_file",
                f"The rook on {chess.square_name(square)} occupies an open file.",
                squares=(chess.square_name(square),),
            )
    return None


def _wrong_bishop(context: ReviewContext) -> DetectedSubject | None:
    board = context.board
    non_kings = [piece for piece in board.piece_map().values() if piece.piece_type != chess.KING]
    if any(piece.piece_type not in {chess.BISHOP, chess.PAWN} for piece in non_kings):
        return None
    for color in chess.COLORS:
        bishops = board.pieces(chess.BISHOP, color)
        rook_pawns = [
            square
            for square in board.pieces(chess.PAWN, color)
            if chess.square_file(square) in {0, 7}
        ]
        enemy_king = board.king(not color)
        for bishop in bishops:
            for pawn in rook_pawns:
                promotion = chess.square(
                    chess.square_file(pawn),
                    7 if color == chess.WHITE else 0,
                )
                if enemy_king is None or chess.square_distance(enemy_king, promotion) > 1:
                    continue
                bishop_color = (chess.square_file(bishop) + chess.square_rank(bishop)) % 2
                promotion_color = (chess.square_file(promotion) + chess.square_rank(promotion)) % 2
                if bishop_color == promotion_color:
                    continue
                return _finding(
                    "wrong_bishop",
                    0.95,
                    "wrong_bishop",
                    f"The bishop on {chess.square_name(bishop)} cannot control the "
                    f"promotion square {chess.square_name(promotion)} for the rook pawn.",
                    squares=(
                        chess.square_name(bishop),
                        chess.square_name(pawn),
                        chess.square_name(promotion),
                    ),
                )
    return None


def _signature_evaluator(
    handler: str,
    required: set[chess.PieceType],
    *,
    both_sides: chess.PieceType | None = None,
) -> Callable[[ReviewContext], DetectedSubject | None]:
    def evaluate(context: ReviewContext) -> DetectedSubject | None:
        pieces = [
            piece for piece in context.board.piece_map().values() if piece.piece_type != chess.KING
        ]
        present = {piece.piece_type for piece in pieces}
        if not required.issubset(present):
            return None
        if any(piece_type not in required | {chess.PAWN} for piece_type in present):
            return None
        if both_sides is not None and any(
            not context.board.pieces(both_sides, color) for color in chess.COLORS
        ):
            return None
        return _finding(
            handler,
            0.7,
            "material_signature",
            f"The remaining material matches the {handler.replace('_', ' ')} subject.",
        )

    return evaluate


Detector = Callable[[ReviewContext], DetectedSubject | None]
DetectorRegistration = tuple[str, Detector]

DETECTORS: tuple[DetectorRegistration, ...] = (
    ("mate", _mate),
    ("defence", _defence),
    ("material", _material),
    ("double_attack", _double_attack),
    ("pin", _pin),
    ("eliminate_defence", _eliminate_defence),
    ("discovered_attack", _discovered_attack),
    ("xray", _xray),
    ("intermediate_move", _intermediate_move),
    ("promotion", _promotion),
    ("threat", _threat),
    ("trapping", _trapping),
    ("interference", _interference),
    ("luring", _luring),
    ("magnet", _magnet),
    ("blocking", _blocking),
    ("chasing_targeting", _chasing_targeting),
    ("clearing", _clearing),
    ("breakthrough", _breakthrough),
)

POSITION_EVALUATORS: tuple[DetectorRegistration, ...] = (
    ("check", _check),
    ("draw", _draw),
    ("passed_pawn", _passed_pawn),
    ("activity", _activity),
    ("opening", _opening),
    ("mate_technique", _mate_technique),
    ("pawn_endgame", _pawn_endgame),
    ("pawn_square", _pawn_square),
    ("mobility", _mobility),
    ("rook_pawn", _rook_pawn),
    ("weak_pawn", _weak_pawn),
    ("material_advantage", _material_advantage),
    ("king_attack", _king_attack),
    ("seventh_rank", _seventh_rank),
    ("queen_pawn", _queen_pawn),
    ("pawn_race", _pawn_race),
    ("strong_square", _strong_square),
    ("open_file", _open_file),
    (
        "rook_endgame",
        _signature_evaluator("rook_endgame", {chess.ROOK}, both_sides=chess.ROOK),
    ),
    ("wrong_bishop", _wrong_bishop),
    (
        "queen_endgame",
        _signature_evaluator("queen_endgame", {chess.QUEEN}, both_sides=chess.QUEEN),
    ),
)

DETECTOR_REGISTRY = DETECTORS + POSITION_EVALUATORS
TEACHING_DETECTOR_REGISTRY = tuple(
    registration
    for registration in DETECTOR_REGISTRY
    if registration[0] not in _CONTEXT_ONLY_HANDLERS
)
DETECTABLE_HANDLERS = {handler for handler, _detector in DETECTOR_REGISTRY}


SUBJECT_PRIORITY = {
    "mate": 0,
    "double_attack": 2,
    "eliminate_defence": 3,
    "material": 4,
    "discovered_attack": 5,
    "pin": 6,
    "xray": 7,
    "intermediate_move": 8,
    "promotion": 9,
    "interference": 10,
    "luring": 11,
    "blocking": 12,
    "magnet": 13,
    "clearing": 14,
    "chasing_targeting": 15,
    "trapping": 16,
    "breakthrough": 17,
    "threat": 18,
    "check": 19,
    "defence": 20,
    "passed_pawn": 30,
    "king_attack": 31,
    "seventh_rank": 32,
    "open_file": 33,
    "strong_square": 34,
    "material_advantage": 35,
}


def _validate_evidence_square(square: str) -> None:
    try:
        chess.parse_square(square)
    except ValueError as exc:
        raise ValueError(f"Evidence contains an invalid square: {square}") from exc


def _run_detector(
    registration: DetectorRegistration,
    context: ReviewContext,
) -> DetectedSubject | None:
    handler, detector = registration
    finding = detector(context)
    if finding is not None and finding.handler != handler:
        raise RuntimeError(f"Detector registered as {handler} returned {finding.handler}")
    return finding


def _subject_sort_key(finding: DetectedSubject) -> tuple[int, float, str]:
    return (
        SUBJECT_PRIORITY.get(finding.handler, 100),
        -finding.confidence,
        finding.handler,
    )


def _finding(
    handler: str,
    confidence: float,
    kind: str,
    summary: str,
    *,
    squares: tuple[str, ...] = (),
    moves: tuple[str, ...] = (),
    actor: PieceRef | None = None,
    targets: tuple[PieceRef, ...] = (),
) -> DetectedSubject:
    return DetectedSubject(
        handler=handler,
        confidence=confidence,
        evidence=(
            Evidence(
                kind,
                summary,
                squares,
                moves,
                actor=actor,
                targets=targets,
            ),
        ),
    )


def _piece_reference(piece: chess.Piece, square: chess.Square) -> PieceRef:
    return PieceRef(
        color="white" if piece.color == chess.WHITE else "black",
        piece=chess.piece_name(piece.piece_type),
        square=chess.square_name(square),
    )


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _line_captures_square(
    context: ReviewContext,
    squares: set[chess.Square],
    *,
    start_index: int,
) -> bool:
    for index in range(start_index, len(context.line.moves), 2):
        board = context.line.boards[index - 1]
        move = context.line.moves[index]
        if board.is_capture(move) and move.to_square in squares:
            return True
    return False


def _settled_material_gain(context: ReviewContext) -> int:
    initial = _material_balance(context.board, context.mover)
    final = context.line.boards[-1]
    final_gain = _material_balance(final, context.mover) - initial
    if final_gain <= 0 or final.turn == context.mover:
        return final_gain

    settled = context.line.boards[-2] if len(context.line.boards) > 1 else context.board
    settled_gain = _material_balance(settled, context.mover) - initial
    if settled_gain > 0:
        return settled_gain

    last_move = context.line.moves[-1]
    before_last = settled
    if not before_last.is_capture(last_move):
        return settled_gain
    recapture = any(
        final.is_capture(reply) and reply.to_square == last_move.to_square
        for reply in final.legal_moves
    )
    return settled_gain if recapture else final_gain


def _material_balance(board: chess.Board, color: chess.Color) -> int:
    own = sum(
        len(board.pieces(piece_type, color)) * value for piece_type, value in PIECE_VALUES.items()
    )
    enemy = sum(
        len(board.pieces(piece_type, not color)) * value
        for piece_type, value in PIECE_VALUES.items()
    )
    return own - enemy


def _is_slider(piece: chess.Piece | None) -> bool:
    return piece is not None and piece.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}


def _enemy_targets(
    board: chess.Board, square: chess.Square, color: chess.Color
) -> set[chess.Square]:
    return {target for target in board.attacks(square) if board.color_at(target) == (not color)}


def _first_two_enemy_pieces_on_ray(
    board: chess.Board,
    origin: chess.Square,
    color: chess.Color,
    piece_type: chess.PieceType,
) -> tuple[chess.Square, chess.Square] | None:
    diagonals = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    orthogonals = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if piece_type == chess.BISHOP:
        directions = diagonals
    elif piece_type == chess.ROOK:
        directions = orthogonals
    else:
        directions = diagonals + orthogonals

    for file_step, rank_step in directions:
        found: list[chess.Square] = []
        file = chess.square_file(origin) + file_step
        rank = chess.square_rank(origin) + rank_step
        while 0 <= file < 8 and 0 <= rank < 8:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == color:
                    break
                found.append(square)
                if len(found) == 2:
                    return found[0], found[1]
            file += file_step
            rank += rank_step
    return None


def _passed_pawns(board: chess.Board, color: chess.Color) -> set[chess.Square]:
    enemy_pawns = board.pieces(chess.PAWN, not color)
    result: set[chess.Square] = set()
    for square in board.pieces(chess.PAWN, color):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        blocked = False
        for enemy in enemy_pawns:
            enemy_file = chess.square_file(enemy)
            enemy_rank = chess.square_rank(enemy)
            ahead = enemy_rank > rank if color == chess.WHITE else enemy_rank < rank
            if ahead and abs(enemy_file - file) <= 1:
                blocked = True
                break
        if not blocked:
            result.add(square)
    return result


def _pseudo_mobility(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.attacks(square)) for square in chess.scan_forward(board.occupied_co[color])
    )


def _most_advanced_pawn(board: chess.Board, color: chess.Color) -> chess.Square | None:
    passed = _passed_pawns(board, color)
    if not passed:
        return None
    key = (
        chess.square_rank if color == chess.WHITE else lambda square: 7 - chess.square_rank(square)
    )
    return max(passed, key=key)
