"""Versioned learning-topic registry for position reviews.

The registry uses general chess terminology and an independent Chess Scan taxonomy. Every
curriculum entry is classified explicitly so that non-detectable lesson formats never become
fabricated detector findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TopicCapability(StrEnum):
    DETECTOR = "detector"
    EVALUATOR = "evaluator"
    POLICY = "policy"
    HISTORY_REQUIRED = "history_required"
    UNSUPPORTED = "unsupported"


TOPIC_REGISTRY_VERSION = "chess-scan-curriculum-1"


class Course(StrEnum):
    BASIC = "basic"
    PLUS = "plus"


@dataclass(frozen=True, slots=True)
class ReviewTopic:
    id: str
    name: str
    level: int
    course: Course
    capability: TopicCapability
    handler: str


def topics_for_level(level: int, *, include_plus: bool = True) -> tuple[ReviewTopic, ...]:
    if level not in range(1, 7):
        raise ValueError("Study level must be between 1 and 6")
    return tuple(
        topic
        for topic in REVIEW_TOPICS
        if topic.level <= level and (include_plus or topic.course is Course.BASIC)
    )


def topic_by_id(topic_id: str) -> ReviewTopic:
    try:
        return TOPICS_BY_ID[topic_id]
    except KeyError as exc:
        raise KeyError(f"Unknown review topic: {topic_id}") from exc


def _topic(
    level: int,
    course: Course,
    slug: str,
    name: str,
    capability: TopicCapability,
    handler: str,
) -> ReviewTopic:
    return ReviewTopic(
        id=f"level-{level}.{course.value}.{slug}",
        name=name,
        level=level,
        course=course,
        capability=capability,
        handler=handler,
    )


D = TopicCapability.DETECTOR
E = TopicCapability.EVALUATOR
P = TopicCapability.POLICY
H = TopicCapability.HISTORY_REQUIRED
U = TopicCapability.UNSUPPORTED
B = Course.BASIC
X = Course.PLUS

REVIEW_TOPICS = (
    # Level 1 — rules, board vision, one-move material, mate, and defence.
    _topic(1, B, "board-and-pieces", "The board and the pieces", P, "rules"),
    _topic(1, B, "piece-movement", "How the pieces move", P, "rules"),
    _topic(1, B, "attack-and-capture", "Attack and capture", D, "material"),
    _topic(1, B, "pawn", "The pawn", P, "rules"),
    _topic(1, B, "defending", "Defending", D, "defence"),
    _topic(1, B, "check", "Check", E, "check"),
    _topic(1, B, "mate-one", "Mate in one", D, "mate"),
    _topic(1, B, "mate-two", "Mate patterns", D, "mate"),
    _topic(1, B, "castling", "Castling", P, "rules"),
    _topic(1, B, "profitable-exchange", "The profitable exchange", D, "material"),
    _topic(1, B, "twofold-attack", "The twofold attack", D, "double_attack"),
    _topic(1, B, "draws", "Draws", E, "draw"),
    _topic(1, B, "mating-with-queen", "Mating with the queen", E, "mate_technique"),
    _topic(1, B, "en-passant", "Capturing en passant", P, "rules"),
    _topic(1, B, "notation", "Notation", P, "notation"),
    _topic(1, X, "winning-material", "Winning material", D, "material"),
    _topic(1, X, "defending", "Defending", D, "defence"),
    _topic(1, X, "mate", "Mate", D, "mate"),
    _topic(1, X, "board-vision", "Board vision", P, "board_vision"),
    _topic(1, X, "defending-against-mate", "Defending against mate", D, "defence"),
    _topic(1, X, "draws", "Draws", E, "draw"),
    _topic(1, X, "creating-mate", "Creating mate", D, "mate"),
    _topic(1, X, "passed-pawn", "The passed pawn", E, "passed_pawn"),
    # Level 2 — targets and the foundational tactical building blocks.
    _topic(2, B, "activity", "Activity", E, "activity"),
    _topic(2, B, "double-attack-one", "Double attack: queen", D, "double_attack"),
    _topic(2, B, "double-attack-two", "Double attack: knight", D, "double_attack"),
    _topic(2, B, "pin", "The pin", D, "pin"),
    _topic(2, B, "eliminating-defence", "Eliminating the defence", D, "eliminate_defence"),
    _topic(2, B, "golden-rules", "The opening rules", E, "opening"),
    _topic(2, B, "mate-in-two", "Mate in two", D, "mate"),
    _topic(2, B, "double-attack-pieces", "Double attack: pieces", D, "double_attack"),
    _topic(2, B, "mating-with-rook", "Mating with the rook", E, "mate_technique"),
    _topic(2, B, "discovered-attack", "Discovered attack", D, "discovered_attack"),
    _topic(2, B, "defending-against-mate", "Defending against mate", D, "defence"),
    _topic(2, B, "intermediate-move", "The intermediate move", D, "intermediate_move"),
    _topic(2, B, "solving-tests", "Solving mixed positions", P, "mixed_review"),
    _topic(2, X, "mate", "Mate", D, "mate"),
    _topic(2, X, "pawn-endings", "Pawn endings", E, "pawn_endgame"),
    _topic(2, X, "opening", "The opening", H, "opening_history"),
    _topic(2, X, "defending", "Defending", D, "defence"),
    _topic(2, X, "route-planner", "Route planner", P, "route_planner"),
    _topic(2, X, "working-out-mate", "Working out mate", P, "thinking_ahead"),
    _topic(2, X, "stalemate", "Stalemate", E, "draw"),
    _topic(2, X, "winning-material", "Winning material", D, "material"),
    _topic(2, X, "playing-rules", "Playing rules", P, "rules"),
    # Level 3 — tactical characteristics, defence, plans, and elementary endings.
    _topic(3, B, "completing-opening", "Completing the opening", E, "opening"),
    _topic(3, B, "discovered-double-check", "Discovered and double check", D, "discovered_attack"),
    _topic(3, B, "attacking-pinned-piece", "Attacking a pinned piece", D, "pin"),
    _topic(3, B, "mate-access", "Mate after gaining access", D, "mate"),
    _topic(3, B, "square-of-pawn", "The square of the pawn", E, "pawn_square"),
    _topic(3, B, "eliminating-defence", "Eliminating the defence", D, "eliminate_defence"),
    _topic(3, B, "defending-double-attack", "Defending against a double attack", D, "defence"),
    _topic(3, B, "mini-plans", "Mini-plans", U, "mini_plan"),
    _topic(3, B, "draws", "Draws", E, "draw"),
    _topic(3, B, "x-ray", "X-ray", D, "xray"),
    _topic(3, B, "opening", "The opening", H, "opening_history"),
    _topic(3, B, "defending-pin", "Defending against a pin", D, "defence"),
    _topic(3, B, "mobility", "Mobility", E, "mobility"),
    _topic(3, B, "key-squares-one", "Key squares", U, "key_square"),
    _topic(3, B, "pinned-pieces", "Pinned pieces", D, "pin"),
    _topic(3, B, "threats", "Threats", D, "threat"),
    _topic(3, B, "key-squares-two", "Key squares: applying the method", U, "key_square"),
    _topic(3, X, "x-ray-effect", "The X-ray effect", D, "xray"),
    _topic(3, X, "pinned-pieces", "Pinned pieces", D, "pin"),
    _topic(3, X, "rook-pawn", "The rook pawn", E, "rook_pawn"),
    _topic(3, X, "intermediate-move", "The intermediate move", D, "intermediate_move"),
    _topic(3, X, "opening-vulnerability", "Vulnerability in the opening", H, "opening_history"),
    _topic(3, X, "mini-plans", "Mini-plans", U, "mini_plan"),
    _topic(3, X, "mate", "Mate", D, "mate"),
    _topic(3, X, "elimination-defence", "Elimination of the defence", D, "eliminate_defence"),
    _topic(3, X, "underpromotion", "Underpromotion", D, "promotion"),
    _topic(3, X, "development", "Development", E, "opening"),
    _topic(3, X, "pinning", "Pinning", D, "pin"),
    _topic(3, X, "defending-mate", "Defending against mate", D, "defence"),
    _topic(3, X, "square-of-pawn", "The square of the pawn", E, "pawn_square"),
    _topic(3, X, "discovered-attack", "The discovered attack", D, "discovered_attack"),
    # Level 4 — preparatory moves, attack, positional targets, and endgame strategy.
    _topic(4, B, "opening-advantage", "Opening advantage", H, "opening_history"),
    _topic(4, B, "interfering", "Interfering", D, "interference"),
    _topic(4, B, "luring", "Luring", D, "luring"),
    _topic(4, B, "blocking", "Blocking", D, "blocking"),
    _topic(4, B, "thinking-ahead", "Thinking ahead", P, "thinking_ahead"),
    _topic(4, B, "pin-luring", "The pin: luring", D, "pin"),
    _topic(4, B, "passed-pawn", "The passed pawn", E, "passed_pawn"),
    _topic(4, B, "eliminating-defence", "Eliminating the defence", D, "eliminate_defence"),
    _topic(4, B, "magnet", "The magnet", D, "magnet"),
    _topic(4, B, "weak-pawns", "Weak pawns", E, "weak_pawn"),
    _topic(4, B, "material-advantage", "Material advantage", E, "material_advantage"),
    _topic(4, B, "chasing-targeting", "Chasing and targeting", D, "chasing_targeting"),
    _topic(4, B, "attacking-king", "Attacking the king", E, "king_attack"),
    _topic(4, B, "seventh-rank", "The seventh rank", E, "seventh_rank"),
    _topic(4, B, "endgame-strategy", "Endgame strategy", U, "endgame_strategy"),
    _topic(4, B, "clearing", "Clearing", D, "clearing"),
    _topic(4, B, "queen-against-pawn", "Queen against pawn", E, "queen_pawn"),
    _topic(4, X, "attacking-king", "Attacking the king", E, "king_attack"),
    _topic(4, X, "opening-vulnerability", "Vulnerability in the opening", H, "opening_history"),
    _topic(4, X, "interfering", "Interfering", D, "interference"),
    _topic(4, X, "blocking", "Blocking", D, "blocking"),
    _topic(4, X, "draws", "Draws", E, "draw"),
    _topic(4, X, "trapping", "Trapping", D, "trapping"),
    _topic(4, X, "mini-plans", "Mini-plans", U, "mini_plan"),
    _topic(4, X, "pawn-endings", "Pawn endings", E, "pawn_endgame"),
    _topic(4, X, "discovered-attack", "The discovered attack", D, "discovered_attack"),
    _topic(4, X, "endgame-technique", "Endgame technique", U, "endgame_strategy"),
    _topic(4, X, "chess-problems", "Chess problems", P, "chess_problem"),
    # Level 5 — strategic subjects and more advanced endings.
    _topic(5, B, "material-and-time", "Material and time", E, "material_advantage"),
    _topic(5, B, "mate", "Mate", D, "mate"),
    _topic(5, B, "breakthrough", "Breakthrough", D, "breakthrough"),
    _topic(5, B, "using-pawns", "How to use pawns", U, "pawn_structure"),
    _topic(5, B, "pawn-race", "Pawn race", E, "pawn_race"),
    _topic(5, B, "seventh-rank", "The seventh rank", E, "seventh_rank"),
    _topic(5, B, "discovered-attack", "Discovered attack", D, "discovered_attack"),
    _topic(5, B, "pin", "The pin", D, "pin"),
    _topic(5, B, "opening", "The opening", H, "opening_history"),
    _topic(5, B, "rook-against-pawn", "Rook against pawn", U, "rook_pawn_endgame"),
    _topic(5, B, "strong-square", "Strong square", E, "strong_square"),
    _topic(5, B, "defending", "Defending", D, "defence"),
    _topic(5, B, "rook-ending", "Rook ending", E, "rook_endgame"),
    _topic(5, B, "attacking-king", "Attacking the king", E, "king_attack"),
    _topic(5, B, "open-file", "Open file", E, "open_file"),
    _topic(5, B, "draws", "Draws", E, "draw"),
    _topic(5, X, "activity", "Activity", E, "activity"),
    _topic(5, X, "pawn-endings", "Pawn endings", E, "pawn_endgame"),
    _topic(5, X, "king-middle", "King in the middle", E, "king_attack"),
    _topic(5, X, "wrong-bishop", "The wrong bishop", E, "wrong_bishop"),
    _topic(5, X, "vulnerability", "Vulnerability", U, "vulnerability"),
    _topic(5, X, "queen-endings", "Queen endings", E, "queen_endgame"),
    _topic(5, X, "defending", "Defending", D, "defence"),
    _topic(5, X, "eternal-pins", "Eternal pins", D, "pin"),
    _topic(5, X, "bishop-pawns", "Bishop against pawns", U, "bishop_pawn_endgame"),
    _topic(5, X, "zugzwang", "Zugzwang", U, "zugzwang"),
    # Level 6 — independent study across advanced strategy and endings.
    _topic(6, B, "king-middle", "King in the middle", E, "king_attack"),
    _topic(6, B, "passed-pawn", "The passed pawn", E, "passed_pawn"),
    _topic(6, B, "strategy", "Strategy", U, "strategy"),
    _topic(6, B, "mobility", "Mobility", E, "mobility"),
    _topic(6, B, "draws", "Draws", E, "draw"),
    _topic(6, B, "opening", "The opening", H, "opening_history"),
    _topic(6, B, "tactics", "Tactics", P, "mixed_review"),
    _topic(6, B, "pawn-endings", "Pawn endings", E, "pawn_endgame"),
    _topic(6, B, "bishop-or-knight", "Bishop or knight", U, "bishop_knight"),
    _topic(6, B, "attacking-king", "Attacking the king", E, "king_attack"),
    _topic(6, B, "endgame-advantage", "Endgame advantage", U, "endgame_strategy"),
    _topic(6, B, "bishops", "Bishops", U, "bishops"),
    _topic(6, B, "defending", "Defending", D, "defence"),
    _topic(6, B, "rook-endings", "Rook endings", E, "rook_endgame"),
)

TOPICS_BY_ID = {topic.id: topic for topic in REVIEW_TOPICS}

if len(TOPICS_BY_ID) != len(REVIEW_TOPICS):
    raise RuntimeError("Review topic IDs must be unique")
