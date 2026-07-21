"""High-precision tactical evidence extracted from a legal solution line.

The functions in this module identify relationships that are visible on the board and then
confirmed by the continuation. They do not evaluate move quality and do not produce prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import chess

ColorName = Literal["white", "black"]
Proof = Literal["legal_geometry", "line_consequence"]

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 99,
}
SLIDERS = {chess.BISHOP, chess.ROOK, chess.QUEEN}
SUPPORTED_LICHESS_THEMES = frozenset(
    {
        "attraction",
        "capturingDefender",
        "clearance",
        "discoveredAttack",
        "fork",
        "interference",
        "intermezzo",
        "pin",
        "trappedPiece",
        "xRayAttack",
    }
)


@dataclass(frozen=True, slots=True)
class PieceRef:
    color: ColorName
    piece: str
    square: str


@dataclass(frozen=True, slots=True)
class ThemeEvidence:
    theme: str
    ply: int
    proof: Proof
    actor: PieceRef | None
    from_square: str | None = None
    to_square: str | None = None
    targets: tuple[PieceRef, ...] = ()
    squares: tuple[str, ...] = ()
    moves: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SolutionStep:
    ply: int
    before: chess.Board
    move: chess.Move
    after: chess.Board


@dataclass(frozen=True, slots=True)
class SolutionTrace:
    initial: chess.Board
    player: chess.Color
    steps: tuple[SolutionStep, ...]
    previous: SolutionStep | None = None

    @classmethod
    def build(
        cls,
        board: chess.Board,
        moves: Sequence[chess.Move | str],
        *,
        previous: tuple[chess.Board, chess.Move | str] | None = None,
    ) -> SolutionTrace:
        previous_step = _previous_step(board, previous)
        current = board.copy(stack=False)
        steps: list[SolutionStep] = []
        for ply, raw_move in enumerate(moves):
            try:
                move = chess.Move.from_uci(raw_move) if isinstance(raw_move, str) else raw_move
            except ValueError as exc:
                raise ValueError(f"Solution contains an invalid move: {raw_move}") from exc
            if move not in current.legal_moves:
                raise ValueError(f"Solution contains an illegal move: {move.uci()}")
            before = current.copy(stack=False)
            current.push(move)
            steps.append(SolutionStep(ply, before, move, current.copy(stack=False)))
        if not steps:
            raise ValueError("Solution must contain at least one move")
        return cls(board.copy(stack=False), board.turn, tuple(steps), previous_step)

    def player_steps(self) -> tuple[SolutionStep, ...]:
        return self.steps[::2]


def detect_solution_themes(
    board: chess.Board,
    moves: Sequence[chess.Move | str],
    *,
    previous: tuple[chess.Board, chess.Move | str] | None = None,
) -> tuple[ThemeEvidence, ...]:
    """Return one earliest concrete proof for each supported tactical theme."""

    trace = SolutionTrace.build(board, moves, previous=previous)
    return _detect_trace_themes(trace)


def detect_solution_theme_modes(
    board: chess.Board,
    moves: Sequence[chess.Move | str],
    *,
    previous: tuple[chess.Board, chess.Move | str],
) -> tuple[tuple[ThemeEvidence, ...], tuple[ThemeEvidence, ...]]:
    """Return setup-aware and history-free evidence while sharing the solution trace."""

    trace = SolutionTrace.build(board, moves, previous=previous)
    common = tuple(
        finding
        for detector in _THEME_DETECTORS
        if detector is not _intermezzo
        if (finding := detector(trace)) is not None
    )
    contextual_intermezzo = _intermezzo(trace)
    contextual = common + ((contextual_intermezzo,) if contextual_intermezzo else ())
    return _sorted_evidence(contextual), _sorted_evidence(common)


def detected_lichess_themes(
    board: chess.Board,
    moves: Sequence[chess.Move | str],
    *,
    previous: tuple[chess.Board, chess.Move | str] | None = None,
) -> frozenset[str]:
    return frozenset(
        evidence.theme for evidence in detect_solution_themes(board, moves, previous=previous)
    )


def _fork(trace: SolutionTrace) -> ThemeEvidence | None:
    player_steps = trace.player_steps()
    for step in player_steps[:-1]:
        actor = step.after.piece_at(step.move.to_square)
        if (
            actor is None
            or actor.piece_type == chess.KING
            or _is_bad_spot(step.after, step.move.to_square)
        ):
            continue
        targets: list[PieceRef] = []
        opponent_attackers = step.after.attackers(not trace.player, step.move.to_square)
        for square in step.after.attacks(step.move.to_square):
            target = step.after.piece_at(square)
            if target is None or target.color == trace.player or target.piece_type == chess.PAWN:
                continue
            more_valuable = PIECE_VALUES[target.piece_type] > PIECE_VALUES[actor.piece_type]
            safely_loose = _is_hanging(step.after, square) and square not in opponent_attackers
            if more_valuable or safely_loose:
                targets.append(_piece_ref(target, square))
        if len(targets) >= 2:
            return _move_evidence(
                "fork",
                step,
                actor=_piece_ref(actor, step.move.to_square),
                targets=tuple(targets),
            )
    return None


def _pin(trace: SolutionTrace) -> ThemeEvidence | None:
    for step in trace.player_steps():
        board = step.after
        for square, pinned in board.piece_map().items():
            if pinned.color == trace.player:
                continue
            pin_ray = board.pin(pinned.color, square)
            if pin_ray == chess.BB_ALL:
                continue

            for attacked_square in board.attacks(square):
                target = board.piece_at(attacked_square)
                if (
                    target is not None
                    and target.color == trace.player
                    and attacked_square not in pin_ray
                    and (
                        PIECE_VALUES[target.piece_type] > PIECE_VALUES[pinned.piece_type]
                        or _is_hanging(board, attacked_square)
                    )
                ):
                    return _pin_evidence(trace, step, square, pinned, attacked_square)

            for attacker_square in board.attackers(trace.player, square):
                if attacker_square not in pin_ray:
                    continue
                attacker = board.piece_at(attacker_square)
                if attacker is None:
                    continue
                lower_attacker = PIECE_VALUES[pinned.piece_type] > PIECE_VALUES[attacker.piece_type]
                pseudo_escape = any(
                    move.from_square == square and move.to_square not in pin_ray
                    for move in board.pseudo_legal_moves
                )
                pinned_is_lost = (
                    _is_hanging(board, square)
                    and square not in board.attackers(pinned.color, attacker_square)
                    and pseudo_escape
                )
                if lower_attacker or pinned_is_lost:
                    return _pin_evidence(trace, step, square, pinned, attacker_square)
    return None


def _xray(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(2, len(trace.steps), 2):
        step = trace.steps[index]
        previous_reply = trace.steps[index - 1]
        previous_move = trace.steps[index - 2]
        if not step.before.is_capture(step.move):
            continue
        reply_piece = previous_reply.after.piece_at(previous_reply.move.to_square)
        if (
            reply_piece is None
            or reply_piece.piece_type == chess.KING
            or previous_reply.move.to_square != step.move.to_square
            or previous_move.move.to_square != previous_reply.move.to_square
            or previous_reply.move.from_square
            not in chess.SquareSet(chess.between(step.move.from_square, step.move.to_square))
        ):
            continue
        actor = step.before.piece_at(step.move.from_square)
        target = _captured_piece(step.before, step.move)
        return _move_evidence(
            "xRayAttack",
            step,
            actor=_piece_ref(actor, step.move.from_square) if actor else None,
            targets=(_piece_ref(target, step.move.to_square),) if target else (),
            proof="line_consequence",
            moves=_move_span(trace, index - 2, index),
            extra_squares=(chess.square_name(previous_reply.move.from_square),),
        )
    return None


def _trapped_piece(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(2, len(trace.steps), 2):
        capture_step = trace.steps[index]
        target = _captured_piece(capture_step.before, capture_step.move)
        if target is None or target.piece_type == chess.PAWN:
            continue
        reply = trace.steps[index - 1]
        target_square = (
            reply.move.from_square
            if reply.move.to_square == capture_step.move.to_square
            else capture_step.move.to_square
        )
        position_after_trap = trace.steps[index - 2].after
        trapped = position_after_trap.piece_at(target_square)
        if (
            trapped is None
            or trapped.color == trace.player
            or not _is_trapped(position_after_trap, target_square)
        ):
            continue
        actor = capture_step.before.piece_at(capture_step.move.from_square)
        return _move_evidence(
            "trappedPiece",
            capture_step,
            actor=_piece_ref(actor, capture_step.move.from_square) if actor else None,
            targets=(_piece_ref(trapped, target_square),),
            proof="line_consequence",
            moves=_move_span(trace, index - 2, index),
        )
    return None


def _discovered_attack(trace: SolutionTrace) -> ThemeEvidence | None:
    for step in trace.player_steps():
        if step.after.is_check() and step.move.to_square not in step.after.checkers():
            hidden_checkers = list(step.after.checkers())
            if hidden_checkers:
                checker_square = hidden_checkers[0]
                actor = step.after.piece_at(checker_square)
                king_square = step.after.king(not trace.player)
                targets = (
                    (_piece_ref(step.after.piece_at(king_square), king_square),)
                    if king_square is not None
                    else ()
                )
                return _move_evidence(
                    "discoveredAttack",
                    step,
                    actor=_piece_ref(actor, checker_square) if actor else None,
                    targets=targets,
                    extra_squares=(chess.square_name(checker_square),),
                )

    for index in range(2, len(trace.steps), 2):
        capture_step = trace.steps[index]
        if not capture_step.before.is_capture(capture_step.move):
            continue
        previous_reply = trace.steps[index - 1]
        previous_move = trace.steps[index - 2]
        between = chess.SquareSet(
            chess.between(capture_step.move.from_square, capture_step.move.to_square)
        )
        if previous_reply.move.to_square == capture_step.move.to_square:
            continue
        if (
            previous_move.move.from_square not in between
            or previous_move.move.to_square
            in {capture_step.move.from_square, capture_step.move.to_square}
            or previous_move.before.is_castling(previous_move.move)
        ):
            continue
        actor = capture_step.before.piece_at(capture_step.move.from_square)
        target = _captured_piece(capture_step.before, capture_step.move)
        return _move_evidence(
            "discoveredAttack",
            previous_move,
            actor=_piece_ref(actor, capture_step.move.from_square) if actor else None,
            targets=(_piece_ref(target, capture_step.move.to_square),) if target else (),
            proof="line_consequence",
            moves=_move_span(trace, index - 2, index),
            extra_squares=(chess.square_name(previous_move.move.from_square),),
        )
    return None


def _interference(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(2, len(trace.steps), 2):
        capture_step = trace.steps[index]
        target = _captured_piece(capture_step.before, capture_step.move)
        if target is None or not _is_hanging(capture_step.before, capture_step.move.to_square):
            continue
        reply = trace.steps[index - 1]
        previous_move = trace.steps[index - 2]

        if reply.move.to_square != capture_step.move.to_square:
            before_interference = previous_move.before
            if _line_was_interfered(
                before_interference,
                capture_step.move.to_square,
                previous_move.move.to_square,
            ):
                actor = previous_move.after.piece_at(previous_move.move.to_square)
                return _move_evidence(
                    "interference",
                    previous_move,
                    actor=_piece_ref(actor, previous_move.move.to_square) if actor else None,
                    targets=(_piece_ref(target, capture_step.move.to_square),),
                    proof="line_consequence",
                    moves=_move_span(trace, index - 2, index),
                )

        before_reply = previous_move.after
        if _line_was_interfered(
            before_reply,
            capture_step.move.to_square,
            reply.move.to_square,
        ):
            actor = reply.after.piece_at(reply.move.to_square)
            return ThemeEvidence(
                theme="interference",
                ply=reply.ply,
                proof="line_consequence",
                actor=_piece_ref(actor, reply.move.to_square) if actor else None,
                from_square=chess.square_name(reply.move.from_square),
                to_square=chess.square_name(reply.move.to_square),
                targets=(_piece_ref(target, capture_step.move.to_square),),
                squares=(
                    chess.square_name(reply.move.to_square),
                    chess.square_name(capture_step.move.to_square),
                ),
                moves=_move_span(trace, index - 2, index),
            )
    return None


def _intermezzo(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(2, len(trace.steps), 2):
        capture_step = trace.steps[index]
        if not capture_step.before.is_capture(capture_step.move):
            continue
        reply = trace.steps[index - 1]
        inserted = trace.steps[index - 2]
        earlier_reply = trace.steps[index - 3] if index >= 3 else trace.previous
        if earlier_reply is None:
            continue
        capture_square = capture_step.move.to_square
        reply_defended_target = reply.move.from_square in inserted.after.attackers(
            not trace.player, capture_square
        )
        if reply_defended_target or inserted.move.to_square == capture_square:
            continue
        if (
            earlier_reply.move.to_square != capture_square
            or not earlier_reply.before.is_capture(earlier_reply.move)
            or capture_step.move not in earlier_reply.after.legal_moves
        ):
            continue
        actor = inserted.before.piece_at(inserted.move.from_square)
        target = _captured_piece(capture_step.before, capture_step.move)
        return _move_evidence(
            "intermezzo",
            inserted,
            actor=_piece_ref(actor, inserted.move.from_square) if actor else None,
            targets=(_piece_ref(target, capture_square),) if target else (),
            proof="line_consequence",
            moves=_move_span(trace, index - 2, index),
        )
    return None


def _clearance(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(2, len(trace.steps), 2):
        line_step = trace.steps[index]
        if line_step.before.piece_at(line_step.move.to_square) is not None:
            continue
        slider = line_step.after.piece_at(line_step.move.to_square)
        if slider is None or slider.piece_type not in SLIDERS:
            continue
        reply = trace.steps[index - 1]
        clearing = trace.steps[index - 2]
        cleared_square = clearing.move.from_square
        clears_route = (
            cleared_square == line_step.move.to_square
            or cleared_square
            in chess.SquareSet(chess.between(line_step.move.from_square, line_step.move.to_square))
        )
        reply_piece = reply.after.piece_at(reply.move.to_square)
        if (
            clearing.move.promotion is not None
            or clearing.move.to_square in {line_step.move.from_square, line_step.move.to_square}
            or line_step.before.is_check()
            or (
                line_step.after.is_check()
                and reply_piece is not None
                and reply_piece.piece_type == chess.KING
            )
            or not clears_route
        ):
            continue
        destination_was_empty = clearing.before.piece_at(clearing.move.to_square) is None
        moved_piece_exposed = _is_bad_spot(clearing.after, clearing.move.to_square)
        if not destination_was_empty and not moved_piece_exposed:
            continue
        actor = clearing.before.piece_at(clearing.move.from_square)
        return _move_evidence(
            "clearance",
            clearing,
            actor=_piece_ref(actor, clearing.move.from_square) if actor else None,
            proof="line_consequence",
            moves=_move_span(trace, index - 2, index),
            extra_squares=(chess.square_name(cleared_square),),
        )
    return None


def _attraction(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(0, len(trace.steps) - 2, 2):
        offer = trace.steps[index]
        reply = trace.steps[index + 1]
        follow_up = trace.steps[index + 2]
        if reply.move.to_square != offer.move.to_square:
            continue
        attracted = reply.after.piece_at(reply.move.to_square)
        if attracted is None or attracted.piece_type not in {chess.KING, chess.QUEEN, chess.ROOK}:
            continue
        if follow_up.move.to_square not in follow_up.after.attackers(
            trace.player, reply.move.to_square
        ):
            continue
        if attracted.piece_type != chess.KING:
            recaptured = (
                index + 4 < len(trace.steps)
                and trace.steps[index + 4].move.to_square == reply.move.to_square
            )
            if not recaptured:
                continue
        actor = offer.before.piece_at(offer.move.from_square)
        return _move_evidence(
            "attraction",
            offer,
            actor=_piece_ref(actor, offer.move.from_square) if actor else None,
            targets=(_piece_ref(attracted, reply.move.to_square),),
            proof="line_consequence",
            moves=_move_span(trace, index, min(index + 4, len(trace.steps) - 1)),
        )
    return None


def _capturing_defender(trace: SolutionTrace) -> ThemeEvidence | None:
    for index in range(2, len(trace.steps), 2):
        payoff = trace.steps[index]
        reply = trace.steps[index - 1]
        removal = trace.steps[index - 2]
        captured = _captured_piece(payoff.before, payoff.move)
        payoff_piece = payoff.before.piece_at(payoff.move.from_square)
        if captured is None or payoff_piece is None:
            continue
        if not payoff.after.is_checkmate() and (
            payoff_piece.piece_type == chess.KING
            or PIECE_VALUES[captured.piece_type] > PIECE_VALUES[payoff_piece.piece_type]
            or not _is_hanging(payoff.before, payoff.move.to_square)
            or reply.move.to_square == payoff.move.to_square
        ):
            continue
        if removal.after.is_check() or removal.move.to_square == payoff.move.from_square:
            continue
        defender = removal.before.piece_at(removal.move.to_square)
        if (
            defender is None
            or removal.move.to_square
            not in removal.before.attackers(defender.color, payoff.move.to_square)
            or removal.before.is_check()
        ):
            continue
        actor = removal.before.piece_at(removal.move.from_square)
        return _move_evidence(
            "capturingDefender",
            removal,
            actor=_piece_ref(actor, removal.move.from_square) if actor else None,
            targets=(
                _piece_ref(defender, removal.move.to_square),
                _piece_ref(captured, payoff.move.to_square),
            ),
            proof="line_consequence",
            moves=_move_span(trace, index - 2, index),
        )
    return None


def _pin_evidence(
    trace: SolutionTrace,
    step: SolutionStep,
    square: chess.Square,
    pinned: chess.Piece,
    related_square: chess.Square,
) -> ThemeEvidence:
    actor_square = _pinning_piece_square(step.after, square, pinned.color)
    actor = step.after.piece_at(actor_square) if actor_square is not None else None
    return _move_evidence(
        "pin",
        step,
        actor=(
            _piece_ref(actor, actor_square)
            if actor is not None and actor_square is not None
            else None
        ),
        targets=(_piece_ref(pinned, square),),
        extra_squares=(chess.square_name(related_square),),
    )


def _line_was_interfered(
    board: chess.Board,
    target_square: chess.Square,
    blocking_square: chess.Square,
) -> bool:
    target = board.piece_at(target_square)
    if target is None:
        return False
    for defender_square in board.attackers(target.color, target_square):
        defender = board.piece_at(defender_square)
        if (
            defender is not None
            and defender.piece_type in SLIDERS
            and blocking_square in chess.SquareSet(chess.between(defender_square, target_square))
        ):
            return True
    return False


def _pinning_piece_square(
    board: chess.Board,
    pinned_square: chess.Square,
    pinned_color: chess.Color,
) -> chess.Square | None:
    king_square = board.king(pinned_color)
    if king_square is None:
        return None
    for square, piece in board.piece_map().items():
        if piece.color == pinned_color or piece.piece_type not in SLIDERS:
            continue
        if pinned_square in chess.SquareSet(chess.between(square, king_square)):
            return square
    return None


def _is_hanging(board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    if board.attackers(piece.color, square):
        return False
    for attacker_square in board.attackers(not piece.color, square):
        attacker = board.piece_at(attacker_square)
        if attacker is None or attacker.piece_type not in SLIDERS:
            continue
        without_attacker = board.copy(stack=False)
        without_attacker.remove_piece_at(attacker_square)
        if without_attacker.attackers(piece.color, square):
            return False
    return True


def _is_bad_spot(board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    attackers = board.attackers(not piece.color, square)
    if not attackers:
        return False
    if _is_hanging(board, square):
        return True
    return any(
        (attacker := board.piece_at(attacker_square)) is not None
        and attacker.piece_type != chess.KING
        and PIECE_VALUES[attacker.piece_type] < PIECE_VALUES[piece.piece_type]
        for attacker_square in attackers
    )


def _is_trapped(board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if (
        piece is None
        or piece.color != board.turn
        or piece.piece_type in {chess.PAWN, chess.KING}
        or board.is_check()
        or board.is_pinned(piece.color, square)
        or not _is_bad_spot(board, square)
    ):
        return False
    for move in board.legal_moves:
        if move.from_square != square:
            continue
        captured = _captured_piece(board, move)
        if (
            captured is not None
            and PIECE_VALUES[captured.piece_type] >= PIECE_VALUES[piece.piece_type]
        ):
            return False
        after = board.copy(stack=False)
        after.push(move)
        if not _is_bad_spot(after, move.to_square):
            return False
    return True


def _previous_step(
    presented: chess.Board,
    previous: tuple[chess.Board, chess.Move | str] | None,
) -> SolutionStep | None:
    if previous is None:
        return None
    before, raw_move = previous
    before = before.copy(stack=False)
    try:
        move = chess.Move.from_uci(raw_move) if isinstance(raw_move, str) else raw_move
    except ValueError as exc:
        raise ValueError(f"Previous context contains an invalid move: {raw_move}") from exc
    if move not in before.legal_moves:
        raise ValueError(f"Previous context contains an illegal move: {move.uci()}")
    after = before.copy(stack=False)
    after.push(move)
    if after.fen() != presented.fen():
        raise ValueError("Previous move does not lead to the presented position")
    return SolutionStep(-1, before, move, after)


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _move_evidence(
    theme: str,
    step: SolutionStep,
    *,
    actor: PieceRef | None,
    targets: tuple[PieceRef, ...] = (),
    proof: Proof = "legal_geometry",
    moves: tuple[str, ...] | None = None,
    extra_squares: tuple[str, ...] = (),
) -> ThemeEvidence:
    from_square = chess.square_name(step.move.from_square)
    to_square = chess.square_name(step.move.to_square)
    squares = tuple(
        dict.fromkeys(
            (from_square, to_square, *(target.square for target in targets), *extra_squares)
        )
    )
    return ThemeEvidence(
        theme=theme,
        ply=step.ply,
        proof=proof,
        actor=actor,
        from_square=from_square,
        to_square=to_square,
        targets=targets,
        squares=squares,
        moves=moves or (step.move.uci(),),
    )


def _move_span(trace: SolutionTrace, start: int, end: int) -> tuple[str, ...]:
    return tuple(step.move.uci() for step in trace.steps[max(0, start) : end + 1])


def _detect_trace_themes(trace: SolutionTrace) -> tuple[ThemeEvidence, ...]:
    evidence = tuple(
        finding
        for detector in _THEME_DETECTORS
        if detector is not _intermezzo or trace.previous is not None
        if (finding := detector(trace)) is not None
    )
    return _sorted_evidence(evidence)


def _sorted_evidence(evidence: Sequence[ThemeEvidence]) -> tuple[ThemeEvidence, ...]:
    return tuple(sorted(evidence, key=lambda item: (item.ply, item.theme)))


def _piece_ref(piece: chess.Piece | None, square: chess.Square) -> PieceRef:
    if piece is None:
        raise ValueError(f"Expected a piece on {chess.square_name(square)}")
    return PieceRef(
        color="white" if piece.color == chess.WHITE else "black",
        piece=chess.piece_name(piece.piece_type),
        square=chess.square_name(square),
    )


_THEME_DETECTORS = (
    _attraction,
    _capturing_defender,
    _clearance,
    _discovered_attack,
    _fork,
    _interference,
    _intermezzo,
    _pin,
    _trapped_piece,
    _xray,
)
