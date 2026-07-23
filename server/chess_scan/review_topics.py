# ruff: noqa: E501
"""Human-authored copy for topics a single position can prove."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReviewTopic:
    id: str
    name: str
    hint: str
    idea: str


_TOPIC_COPY = {
    "activity": (
        "Piece activity",
        "Look for the piece that can join the game with tempo or reach a more useful square.",
        "Active pieces create threats and restrict replies. The move improves a piece while making the opponent react.",
    ),
    "blocking": (
        "Blocking",
        "Find the important line between the attacking piece and its target. Can anything step into it?",
        "The move interrupts a line of attack. Once the route is blocked, the opponent's pieces stop working together.",
    ),
    "breakthrough": (
        "Breakthrough",
        "The position is closed for now. Look for a forcing move that opens one route through it.",
        "The move gives up stability to open a file, rank, or diagonal that matters more than the material invested.",
    ),
    "chasing_targeting": (
        "Chasing a target",
        "Which enemy piece has the fewest comfortable squares? Gain time by making it move again.",
        "The target is forced away while the attacking piece improves. That extra tempo is the point of the move.",
    ),
    "check": (
        "Giving check",
        "Checks force an immediate reply. Look for one that improves your position or sets up the next forcing move.",
        "The move gives check, so the opponent must answer the king threat before pursuing another plan.",
    ),
    "clearing": (
        "Clearance",
        "One of your own pieces is occupying a useful line or square. Ask where it could move with tempo.",
        "The first piece moves away so another piece can use the line or square it leaves behind.",
    ),
    "defence": (
        "Defence",
        "Before attacking, identify what is already under threat and find the least passive way to save it.",
        "The move meets the immediate threat without surrendering the initiative, often by creating a threat in return.",
    ),
    "discovered_attack": (
        "Discovered attack",
        "Two of your pieces are lined up. Can the front piece move with tempo and uncover the one behind it?",
        "Moving the front piece releases a hidden line of attack, so one move creates pressure from two pieces.",
    ),
    "double_attack": (
        "Double attack",
        "Look for one move that asks two questions at once. Checks are useful because they force the first answer.",
        "The move attacks two important targets at the same time. The forced reply leaves too little time to save both.",
    ),
    "draw": (
        "Drawing resource",
        "Do not assume the position must be won or lost. Check stalemate, repetition, and forcing exchanges.",
        "The position contains a concrete resource that removes the opponent's winning chances.",
    ),
    "eliminate_defence": (
        "Remove the defender",
        "A target looks protected. Identify the piece doing the protecting and ask whether it can be exchanged or driven away.",
        "The first move removes a key defender, and the formerly protected target can no longer be held.",
    ),
    "interference": (
        "Interference",
        "Find two enemy pieces that depend on the same line. Can you place something between them?",
        "The move cuts the connection between two defending pieces, leaving one of their jobs uncovered.",
    ),
    "intermediate_move": (
        "In-between move",
        "Before making the available capture, look for a check or threat that the opponent must answer first.",
        "The move postpones an available capture to insert a more urgent threat and change the move order.",
    ),
    "king_attack": (
        "King attack",
        "Count the attackers and defenders around the king, then look for a way to bring in one more piece with tempo.",
        "The move increases the number or quality of attackers around the king faster than the defence can reorganise.",
    ),
    "luring": (
        "Luring",
        "Is a defender safe only because it stands on the right square? Look for a forcing offer that draws it away.",
        "The opponent is tempted or forced onto a square where a second tactical idea becomes possible.",
    ),
    "magnet": (
        "Magnet",
        "Can a forcing offer pull the king onto a less safe square? Calculate the checks that follow.",
        "The offered piece acts like a magnet, drawing the king onto the exact square needed for the follow-up.",
    ),
    "mate": (
        "Checkmate",
        "The king has fewer safe squares than it appears. Start with forcing checks and test every capture and block.",
        "The move closes the king's remaining escape route while keeping every answer covered.",
    ),
    "mate_technique": (
        "Mating technique",
        "Look first at checks and king-restricting moves. Verify that every legal reply stays covered.",
        "The checked line preserves the mating net while controlling the king's legal replies.",
    ),
    "material": (
        "Winning material",
        "Look for loose pieces and forcing captures. Count what is taken, what can recapture, and what remains attacked.",
        "The sequence wins material because the opponent cannot restore the balance after the forcing moves.",
    ),
    "material_advantage": (
        "Use the extra material",
        "When ahead, reduce counterplay. Look for a safe exchange or a way to make every piece useful.",
        "The move turns an existing material edge into a simpler position with fewer chances for the opponent.",
    ),
    "mobility": (
        "Mobility",
        "Compare which pieces have useful squares and which are boxed in. Improve the least effective piece.",
        "The move increases useful choices while taking squares away from the opponent's pieces.",
    ),
    "open_file": (
        "Open file",
        "Find the file with no pawn blocking it. Which heavy piece can claim it or enter the opponent's position?",
        "A rook or queen uses the open file as an entry route, creating pressure that pawns cannot easily chase away.",
    ),
    "opening": (
        "Development",
        "Improve an undeveloped piece, contest the centre, or make the king safer—preferably two of those at once.",
        "Several pieces are still on their starting squares, so development and king safety frame the choice. First deal with anything urgent, then improve the least useful piece.",
    ),
    "passed_pawn": (
        "Passed pawn",
        "Find the pawn with no enemy pawn in front of it or on a neighbouring file. Can it advance with tempo?",
        "The passed pawn creates a promotion threat that forces enemy pieces into a passive defensive role.",
    ),
    "pawn_break": (
        "Pawn break",
        "Look for a pawn advance that challenges the opponent's chain and changes which files or diagonals can open.",
        "The pawn advance directly challenges an enemy pawn, forcing the structure to change or concede space.",
    ),
    "pawn_endgame": (
        "Pawn ending",
        "King activity comes first. Calculate opposition, pawn races, and whether each pawn move can be taken back.",
        "In a pawn ending, one king step or pawn tempo can decide who reaches the critical square first.",
    ),
    "pawn_race": (
        "Pawn race",
        "Count both races all the way to promotion, including checks after a new queen appears.",
        "Both sides have advanced pawn candidates. Compare the exact promotion tempi and any checks before deciding which race is favorable.",
    ),
    "pawn_square": (
        "The pawn's square",
        "Picture the pawn's square to see whether the king can catch it without calculating every step.",
        "The king's route and the pawn's promotion race are decided by whether the king can enter the pawn's square.",
    ),
    "pin": (
        "Pin",
        "Look along files, ranks, and diagonals for a piece that cannot move without exposing something more valuable.",
        "The pinned piece is not free to move, so it can be attacked again or treated as an unreliable defender.",
    ),
    "promotion": (
        "Promotion",
        "A pawn is close to the back rank. Check every promotion piece—not only the queen—and note any forcing check.",
        "Promotion changes the material immediately, and the choice of piece can create the exact check or defence required.",
    ),
    "prophylaxis": (
        "Prophylaxis",
        "Before improving your own plan, identify the opponent's easiest entry or freeing move and take it away.",
        "The move controls a square the opponent could otherwise use, limiting their plan before continuing with its own.",
    ),
    "queen_endgame": (
        "Queen ending",
        "Keep checking distance, king safety, and passed pawns in view. A forcing check can change everything.",
        "Queen endings revolve around checks and tempo; the move improves the position while limiting counterchecks.",
    ),
    "queen_pawn": (
        "Queen against pawn",
        "Use checks to approach the pawn and force the king in front of it before bringing your own king closer.",
        "The queen gains time through checks, then blocks the pawn so the king can finish the job.",
    ),
    "rook_endgame": (
        "Rook ending",
        "Activate the rook. Checks from the side or behind a passed pawn are usually more useful than passive defence.",
        "The rook becomes active behind or beside the pawns, where it can attack and check at the same time.",
    ),
    "rook_pawn": (
        "Rook pawn",
        "The edge of the board changes the geometry. Check whether the king has enough room to support promotion.",
        "A rook pawn has fewer neighbouring squares, so king placement and the corner square become decisive.",
    ),
    "seventh_rank": (
        "The seventh rank",
        "Look for an entry onto the opponent's second rank, where pawns and the king can be attacked from the side.",
        "A heavy piece on the seventh rank attacks several targets at once and restricts the enemy king.",
    ),
    "space": (
        "Gain space",
        "Which pawn can advance safely to claim useful squares and give your pieces more room?",
        "The pawn advance claims territory on one flank and restricts where the opposing pieces can operate.",
    ),
    "strong_square": (
        "Strong square",
        "Find a square in enemy territory that cannot be challenged by a pawn. Which piece would be hardest to remove there?",
        "The piece occupies a stable outpost where it controls important squares without being chased by a pawn.",
    ),
    "supports_pawn_advance": (
        "Support a pawn advance",
        "Can one piece move to protect the square where a pawn wants to arrive?",
        "The piece improves while supporting an advanced pawn square, making the planned push harder to challenge.",
    ),
    "temporary_sacrifice": (
        "Temporary sacrifice",
        "Calculate beyond the first material loss. Can forcing replies recover the investment with a better position?",
        "The line allows material to be taken temporarily, then uses forcing play to restore the balance or emerge ahead.",
    ),
    "threat": (
        "Create a threat",
        "If no forcing move works immediately, find a move that makes one unavoidable on the next turn.",
        "The move creates a concrete next-move threat, forcing the opponent to abandon their own plan.",
    ),
    "trapping": (
        "Trap a piece",
        "Which enemy piece is short of safe squares? Cover its exits before trying to attack it directly.",
        "The move removes an escape square, and the target no longer has a safe route out.",
    ),
    "weak_pawn": (
        "Weak pawn",
        "Find the pawn that cannot be defended by another pawn. Can you pile up on it without creating a weakness of your own?",
        "The move fixes a pawn as a long-term target and brings another attacker into the pressure.",
    ),
    "wrong_bishop": (
        "Wrong bishop",
        "Compare the bishop's colour with the pawn's promotion corner before assuming the extra pawn wins.",
        "The bishop cannot control the rook pawn's promotion corner, which gives the defending king a drawing fortress.",
    ),
    "xray": (
        "X-ray",
        "Look through the first piece on a file, rank, or diagonal. What becomes exposed if that piece moves?",
        "The long-range piece attacks through an intervening piece, so moving or exchanging the blocker reveals the target behind it.",
    ),
}

REVIEW_TOPICS = {
    handler: ReviewTopic(handler.replace("_", "-"), *copy) for handler, copy in _TOPIC_COPY.items()
}

FORK_TOPICS = {
    "two_targets_knight": ReviewTopic(
        "knight-fork",
        "Knight fork",
        "Look for a checking jump that also attacks a valuable piece. The check may make the second target impossible to save.",
        "The knight lands with two attacks at once. Because one target is the king, the other target must wait.",
    ),
    "two_targets_pawn": ReviewTopic(
        "pawn-fork",
        "Pawn fork",
        "A pawn attack can be easy to overlook. Find an advance that attacks two pieces from its new square.",
        "The pawn advances with tempo and attacks two pieces at once, so at least one target is lost.",
    ),
}

DEFAULT_TOPIC = ReviewTopic(
    "position",
    "Find the best move",
    "Start with checks, captures, and direct threats, then compare the opponent's strongest reply.",
    "The engine prefers this move, but the checked line does not prove one of the supported tactical labels.",
)


def topic_for(handler: str | None, evidence_kind: str = "") -> ReviewTopic:
    if handler is None:
        return DEFAULT_TOPIC
    if handler == "double_attack" and evidence_kind in FORK_TOPICS:
        return FORK_TOPICS[evidence_kind]
    try:
        return REVIEW_TOPICS[handler]
    except KeyError as exc:
        raise ValueError(f"Unknown review topic handler: {handler}") from exc
