"""Reproducible comparison of position reviews with derived expert commentary claims."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import chess
import chess.pgn
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from chess_scan.board import validate_full_fen
from chess_scan.config import PROJECT_ROOT
from chess_scan.expert_sources import canonical_expert_source_url
from chess_scan.model_artifact import is_sha256, sha256_file
from chess_scan.review import build_position_review
from chess_scan.review_themes import SolutionTrace
from chess_scan.schemas import (
    PositionReviewRequest,
    PositionReviewResponse,
    ReviewAnalysisInput,
    ReviewAnnotation,
    ReviewEvidenceResponse,
    ReviewFindingResponse,
    ReviewLineResponse,
    review_line_for_scope,
)

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "benchmarks" / "expert-commentary-v2.json"
QUALITY_LABELS = ("strong", "partial", "miss")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_CLAIM_EVIDENCE_KINDS = {
    "answers-check": {"answer_check"},
    "attacks-pawn-chain": {"pawn_chain_challenged"},
    "attacks-weak-pawn": {"material_threat", "weak_pawn"},
    "attacks-weak-point": {"material_threat", "weak_pawn"},
    "creates-threat": {"material_threat", "mate_threat"},
    "controls-square": {"key_square_control"},
    "exploits-pin": {"pin"},
    "forcing-reply": {"check"},
    "improves-piece": {"mobility_increase"},
    "limits-king-entry": {"king_route_restricted"},
    "limits-plan": {"key_square_control"},
    "moves-out-of-attack": {"defensive_move"},
    "occupies-outpost": {"outpost"},
    "opens-file": {"clearance", "open_file"},
    "opens-line": {"clearance", "discoveredAttack", "open_file"},
    "opens-lines": {"clearance", "discoveredAttack", "open_file"},
    "passed-pawn-advance": {"passed_pawn", "passed_pawn_created"},
    "pawn-break": {"pawn_chain_challenged"},
    "prevents-tactic": {"key_square_control"},
    "space-gain": {"space_gain"},
    "supports-pawn-advance": {"advanced_pawn_square_supported"},
    "temporary-sacrifice": {"temporary_sacrifice"},
    "threatens-capture": {"material_threat"},
    "traps-piece": {"restricted_mobility"},
    "vacates-square": {"clearance"},
    "wins-material": {"material_gain_line"},
}
_UNSUPPORTED_REFERENCE_PREDICATES = {
    "activates-pieces",
    "adds-attack",
    "attacks-two-pawns",
    "avoids-weak-pawn",
    "controls-pawn-break",
    "creates-luft",
    "exploits-blocked-line",
    "favorable-minor-piece-balance",
    "fixes-pawn",
    "fixes-pawns",
    "forces-simplification",
    "forces-trade",
    "improves-king",
    "maintains-tension",
    "play-both-flanks",
    "positional-compensation",
    "prepares-exchange-sacrifice",
    "prevents-passed-pawn",
    "prevents-pawn-advance",
    "prevents-plan",
    "restricts-development",
    "restricts-pawn",
    "restricts-piece",
    "restricts-pieces",
    "supports-king-attack",
    "supports-pawn",
    "supports-piece-square",
    "supports-reroute",
    "targets-uncastled-king",
    "uses-development-lead",
    "winning-continuation",
}
_REFERENCE_PREDICATES = frozenset(_CLAIM_EVIDENCE_KINDS) | _UNSUPPORTED_REFERENCE_PREDICATES
_PieceQualifier = Literal["pawn", "knight", "bishop", "rook", "queen", "king"]


class _ReferenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    predicate: str
    move: str
    proof: Literal["fen", "line", "counterfactual"]
    squares: list[str] = Field(default_factory=list, max_length=16)
    moves: list[str] = Field(default_factory=list, max_length=16)
    piece: _PieceQualifier | None = None
    pieces: list[_PieceQualifier] = Field(default_factory=list, max_length=6)
    flank: Literal["queenside", "center", "kingside"] | None = None
    features: list[Literal["center", "piece-activity", "bishop-pair", "activity"]] = Field(
        default_factory=list, max_length=8
    )


_Split = Literal["development", "validation", "shadow"]
_Quality = Literal["strong", "partial", "miss"]
_Verdict = Literal["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]


class _StrictManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _SourcePolicy(_StrictManifestModel):
    annotation_text_redistributed: bool
    game_scores_redistributed: bool
    derived_claims_committed: bool
    article_and_chapter_urls_retained: bool


class _Selection(_StrictManifestModel):
    cases: int = Field(ge=1)
    splits: dict[_Split, int]
    source_collections: int | dict[_Split, int]
    game_groups: int | dict[_Split, int]
    grouping: Literal["game_and_source_collection"]
    validation_policy: str | None = None
    development_only_reason: str | None = None
    shadow_policy: str | None = None


class _EnginePolicy(_StrictManifestModel):
    name: str = Field(min_length=1)
    score_pov: Literal["side_to_move"]
    root_target_ms: int = Field(gt=0)
    stability_retry_max_ms: int = Field(gt=0)
    attempt_max_ms: int = Field(gt=0)
    analysis_committed: bool


class _SourceCollection(_StrictManifestModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    study_url: str = Field(min_length=1)
    study_pgn_url: str = Field(min_length=1)
    study_pgn_sha256: str
    split: _Split


class _CaseSource(_StrictManifestModel):
    article_url: str = Field(min_length=1)
    chapter_url: str = Field(min_length=1)
    chapter_pgn_url: str = Field(min_length=1)
    chapter_pgn_sha256: str
    annotation_context_sha256: str
    annotator: str = Field(min_length=1)


class _Game(_StrictManifestModel):
    white: str = Field(min_length=1)
    black: str = Field(min_length=1)
    round: str = Field(min_length=1)


class _Position(_StrictManifestModel):
    fen: str = Field(min_length=1, max_length=120)
    move_number: int = Field(ge=1)
    side_to_move: Literal["white", "black"]
    annotated_move_uci: str
    annotated_move_san: str = Field(min_length=1)


class _PartialLine(_StrictManifestModel):
    best_move: str
    attempt_reply: str


class _Reference(_StrictManifestModel):
    acceptable_attempt_verdicts: list[_Verdict] = Field(min_length=1)
    acceptable_best_moves: list[str] = Field(min_length=1)
    acceptable_primary_concepts: list[str] = Field(min_length=1)
    acceptable_secondary_concepts: list[str]
    disallowed_primary_concepts: list[str]
    claims: list[_ReferenceClaim] = Field(min_length=1)
    history_dependent_claims: list[str]
    baseline_quality: _Quality
    partial_line: _PartialLine | None = None


class _CaseGroup(_StrictManifestModel):
    source_collection_id: str = Field(min_length=1)
    game_id: str = Field(min_length=1)


class _AnalysisRuntime(_StrictManifestModel):
    root_ms: int = Field(gt=0)
    attempt_ms: int = Field(gt=0)


class _CommentaryCase(_StrictManifestModel):
    id: str = Field(min_length=1)
    split: _Split
    source: _CaseSource
    game: _Game
    position: _Position
    reference: _Reference
    analysis: ReviewAnalysisInput
    analysis_runtime: _AnalysisRuntime | None = None
    group: _CaseGroup

    @model_validator(mode="after")
    def validate_attempt(self) -> _CommentaryCase:
        if (
            self.analysis.attempt is None
            or self.analysis.attempt.move != self.position.annotated_move_uci
        ):
            raise ValueError("Expert commentary analysis attempt differs from its annotated move")
        return self


class _BaselineGate(_StrictManifestModel):
    minimum_cases: int = Field(ge=1)
    minimum_verdict_agreement: float = Field(ge=0, le=1)
    minimum_best_move_support: float = Field(ge=0, le=1)
    minimum_strong: int = Field(ge=0)
    minimum_covered: int = Field(ge=0)
    maximum_disallowed_primary: int = Field(ge=0)
    maximum_later_primary: int = Field(ge=0)


class _CommentaryManifest(_StrictManifestModel):
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_policy: _SourcePolicy
    selection: _Selection
    engine: _EnginePolicy
    quality_scale: list[_Quality] = Field(min_length=3, max_length=3)
    cases: list[_CommentaryCase] = Field(min_length=1)
    source_collections: list[_SourceCollection] = Field(min_length=1)
    baseline_gates: dict[_Split, _BaselineGate]


def load_commentary_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if isinstance(manifest, dict) and "source_collections" not in manifest:
        raise ValueError("Archived expert commentary manifests are not runnable")
    _validate_commentary_manifest(manifest)
    return manifest


def validate_commentary_manifests_disjoint(
    candidate: dict[str, Any],
    prior_manifests: list[dict[str, Any]],
) -> None:
    candidate_collections = _collection_identities(candidate)
    candidate_chapters = {case["source"]["chapter_url"] for case in candidate["cases"]}
    candidate_hashes = {case["source"]["chapter_pgn_sha256"] for case in candidate["cases"]}
    candidate_games = {case["group"]["game_id"] for case in candidate["cases"]}
    for prior in prior_manifests:
        if candidate_collections & _collection_identities(prior):
            raise ValueError("Commentary manifests reuse a source collection")
        if candidate_chapters & {case["source"]["chapter_url"] for case in prior["cases"]}:
            raise ValueError("Commentary manifests reuse a source chapter")
        if candidate_hashes & {case["source"]["chapter_pgn_sha256"] for case in prior["cases"]}:
            raise ValueError("Commentary manifests reuse a chapter hash")
        if candidate_games & {case["group"]["game_id"] for case in prior["cases"]}:
            raise ValueError("Commentary manifests reuse a game group")


def evaluate_commentary(
    manifest: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    cases = [case for case in manifest["cases"] if case["split"] == split]
    if not cases:
        raise ValueError(f"Expert commentary split has no cases: {split}")

    results = [_evaluate_case(case) for case in cases]
    quality = Counter(result["quality"] for result in results)
    cases_count = len(results)
    verdict_matches = sum(result["verdict_matches"] for result in results)
    best_move_matches = sum(result["best_move_supported"] for result in results)
    disallowed_primary = sum(result["disallowed_primary"] for result in results)
    later_primary = sum(result["later_primary"] for result in results)
    baseline_matches = sum(result["matches_documented_baseline"] for result in results)

    return {
        "split": split,
        "cases": cases_count,
        "quality": {label: quality[label] for label in QUALITY_LABELS},
        "strong_rate": quality["strong"] / cases_count,
        "covered_rate": (quality["strong"] + quality["partial"]) / cases_count,
        "verdict_agreement": verdict_matches / cases_count,
        "best_move_support": best_move_matches / cases_count,
        "disallowed_primary": disallowed_primary,
        "later_primary": later_primary,
        "generic_fallbacks": sum(result["primary_concept"] == "position" for result in results),
        "documented_baseline_matches": baseline_matches,
        "results": results,
    }


def commentary_quality_rank(quality: str) -> int:
    if quality == "fallback":
        return 0
    try:
        return len(QUALITY_LABELS) - QUALITY_LABELS.index(quality) - 1
    except ValueError as exc:
        raise ValueError(f"Unknown commentary quality: {quality}") from exc


def commentary_gate(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    gates = manifest["baseline_gates"]
    split = metrics.get("split")
    if split not in gates:
        raise ValueError(f"Expert commentary split has no baseline gate: {split}")
    gate = gates[split]
    reasons: list[str] = []
    _minimum(reasons, "cases", metrics["cases"], gate["minimum_cases"])
    _minimum(
        reasons,
        "verdict agreement",
        metrics["verdict_agreement"],
        gate["minimum_verdict_agreement"],
    )
    _minimum(
        reasons,
        "best-move support",
        metrics["best_move_support"],
        gate["minimum_best_move_support"],
    )
    _minimum(reasons, "strong explanations", metrics["quality"]["strong"], gate["minimum_strong"])
    covered = metrics["quality"]["strong"] + metrics["quality"]["partial"]
    _minimum(reasons, "covered explanations", covered, gate["minimum_covered"])
    _maximum(
        reasons,
        "disallowed primary explanations",
        metrics["disallowed_primary"],
        gate["maximum_disallowed_primary"],
    )
    _maximum(
        reasons,
        "later-line primary explanations",
        metrics["later_primary"],
        gate["maximum_later_primary"],
    )
    return not reasons, reasons


def verify_commentary_source(
    case: dict[str, Any],
    path: Path,
    *,
    actual_file_hash: str | None = None,
) -> dict[str, str]:
    source = case["source"]
    expected_file_hash = source["chapter_pgn_sha256"]
    if actual_file_hash is None:
        actual_file_hash = sha256_file(path)
    if actual_file_hash != expected_file_hash:
        raise ValueError(
            f"Expert source changed for {case['id']}: "
            f"expected {expected_file_hash}, got {actual_file_hash}"
        )

    with path.open() as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"Expert source contains no game: {case['id']}")
    game_metadata = case["game"]
    for field, header in (("white", "White"), ("black", "Black"), ("round", "Round")):
        declared = " ".join(str(game_metadata[field]).split())
        pinned = " ".join(str(game.headers.get(header, "")).split())
        if declared != pinned:
            raise ValueError(f"Expert source {header} header changed for {case['id']}")

    position = case["position"]
    board = game.board()
    selected: chess.pgn.ChildNode | None = None
    selected_san = ""
    expected_turn = chess.WHITE if position["side_to_move"] == "white" else chess.BLACK
    for node in game.mainline():
        san = board.san(node.move)
        if (
            board.fullmove_number == position["move_number"]
            and board.turn == expected_turn
            and node.move.uci() == position["annotated_move_uci"]
        ):
            selected = node
            selected_san = san
            break
        board.push(node.move)

    if selected is None:
        raise ValueError(f"Annotated move is missing from expert source: {case['id']}")
    if board.fen() != position["fen"]:
        raise ValueError(f"Expert source FEN changed for {case['id']}")
    if selected_san != position["annotated_move_san"]:
        raise ValueError(f"Expert source SAN changed for {case['id']}")

    context = _annotation_context(selected)
    context_hash = hashlib.sha256(context.encode()).hexdigest()
    if context_hash != source["annotation_context_sha256"]:
        raise ValueError(f"Expert annotation changed for {case['id']}")
    return {"case_id": case["id"], "chapter_pgn_sha256": actual_file_hash}


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    request = commentary_case_review_request(case)
    review = build_position_review(request)
    if review.attempt is None or review.best_move is None:
        raise ValueError(f"Commentary case requires an attempt and best move: {case['id']}")

    reference = case["reference"]
    primary_concept = review.topic.id
    finding_concepts = [finding.topic.id for finding in review.findings]
    visible_findings = _visible_findings(review)
    visible_finding_concepts = [finding.topic.id for finding in visible_findings]
    verdict_matches = review.attempt.verdict in reference["acceptable_attempt_verdicts"]
    best_move_supported = review.best_move.uci in reference["acceptable_best_moves"]
    disallowed_primary = primary_concept in reference["disallowed_primary_concepts"]
    primary_evidence = _primary_evidence(review)
    primary_claim_supported = bool(
        review.findings
        and review.findings[0] in visible_findings
        and _finding_matches_reference(review, review.findings[0], reference)
    )
    later_primary = _is_later_primary(primary_evidence)

    if (
        not disallowed_primary
        and primary_concept in reference["acceptable_primary_concepts"]
        and primary_claim_supported
    ):
        quality = "strong"
    elif any(
        finding.topic.id in reference["acceptable_secondary_concepts"]
        and finding.topic.id not in reference["disallowed_primary_concepts"]
        and _finding_matches_reference(review, finding, reference)
        for finding in visible_findings
    ):
        quality = "partial"
    else:
        quality = "miss"

    return {
        "id": case["id"],
        "quality": quality,
        "documented_baseline": reference["baseline_quality"],
        "matches_documented_baseline": quality == reference["baseline_quality"],
        "verdict": review.attempt.verdict,
        "verdict_matches": verdict_matches,
        "best_move": review.best_move.uci,
        "best_move_supported": best_move_supported,
        "primary_concept": primary_concept,
        "finding_concepts": finding_concepts,
        "visible_finding_concepts": visible_finding_concepts,
        "primary_claim_supported": primary_claim_supported,
        "primary_scope": primary_evidence.scope if primary_evidence else None,
        "primary_ply": primary_evidence.ply if primary_evidence else None,
        "disallowed_primary": disallowed_primary,
        "later_primary": later_primary,
    }


def commentary_annotation_quality(
    case: dict[str, Any],
    review: PositionReviewResponse,
    annotation: ReviewAnnotation,
) -> str:
    reference = case["reference"]
    annotation_ids = set(annotation.evidence_ids)
    matching_findings = [
        finding
        for finding in review.findings
        if annotation_ids & set(finding.evidence_ids)
        and _finding_matches_reference(review, finding, reference)
    ]
    if any(
        finding.topic.id in reference["acceptable_primary_concepts"]
        and finding.topic.id not in reference["disallowed_primary_concepts"]
        for finding in matching_findings
    ):
        return "strong"
    if any(
        finding.topic.id in reference["acceptable_secondary_concepts"]
        and finding.topic.id not in reference["disallowed_primary_concepts"]
        for finding in matching_findings
    ):
        return "partial"
    return "miss"


def _visible_findings(review: PositionReviewResponse) -> list[ReviewFindingResponse]:
    visible_evidence = {
        evidence_id for annotation in review.explanation for evidence_id in annotation.evidence_ids
    }
    return [finding for finding in review.findings if visible_evidence & set(finding.evidence_ids)]


def _finding_matches_reference(
    review: PositionReviewResponse,
    finding: ReviewFindingResponse,
    reference: dict[str, Any],
) -> bool:
    evidence_by_id = {evidence.id: evidence for evidence in review.evidence}
    return any(
        _evidence_matches_claim(review, evidence_by_id[evidence_id], claim)
        for evidence_id in finding.evidence_ids
        for claim in reference["claims"]
    )


def _evidence_matches_claim(
    review: PositionReviewResponse,
    evidence: ReviewEvidenceResponse,
    claim: dict[str, Any],
) -> bool:
    allowed_kinds = _CLAIM_EVIDENCE_KINDS.get(claim["predicate"], set())
    if evidence.kind not in allowed_kinds:
        return False
    if evidence.proof == "line_consequence" and not _verified_line_consequence(review, evidence):
        return False
    if evidence.proof == "counterfactual" and not _verified_counterfactual(review, evidence):
        return False
    if _causal_move(review, evidence) != claim["move"]:
        return False
    if not _proof_matches(str(claim["proof"]), evidence):
        return False

    supported_squares = set(evidence.squares)
    if evidence.actor is not None:
        supported_squares.add(evidence.actor.square)
    supported_squares.update(target.square for target in evidence.targets)
    if evidence.from_square is not None:
        supported_squares.add(evidence.from_square)
    if evidence.to_square is not None:
        supported_squares.add(evidence.to_square)
    claim_squares = set(claim.get("squares", []))
    if not claim_squares <= supported_squares:
        return False

    claim_piece = claim.get("piece")
    if claim_piece is not None:
        evidence_pieces = {target.piece for target in evidence.targets}
        if evidence.actor is not None:
            evidence_pieces.add(evidence.actor.piece)
        if claim_piece not in evidence_pieces:
            return False

    claim_moves = claim.get("moves", [])
    if not _is_ordered_subsequence(claim_moves, evidence.moves):
        return False
    return _flank_matches(claim.get("flank"), claim["move"])


def _verified_line_consequence(
    review: PositionReviewResponse,
    evidence: ReviewEvidenceResponse,
) -> bool:
    line = _canonical_review_line(review, evidence)
    if line is None or not evidence.moves or evidence.ply >= len(line.moves):
        return False
    canonical_moves = [move.uci for move in line.moves]
    if not _is_ordered_subsequence(evidence.moves, canonical_moves):
        return False
    if canonical_moves[evidence.ply] not in evidence.moves:
        return False
    try:
        SolutionTrace.build(validate_full_fen(review.fen), canonical_moves)
    except ValueError:
        return False
    return True


def _verified_counterfactual(
    review: PositionReviewResponse,
    evidence: ReviewEvidenceResponse,
) -> bool:
    line = _canonical_review_line(review, evidence)
    if line is None or not evidence.moves or evidence.ply >= len(line.moves):
        return False
    canonical_moves = [move.uci for move in line.moves]
    if evidence.moves[0] != canonical_moves[evidence.ply]:
        return False
    try:
        SolutionTrace.build(
            validate_full_fen(review.fen),
            [*canonical_moves[: evidence.ply], *evidence.moves],
        )
    except ValueError:
        return False
    return True


def _canonical_review_line(
    review: PositionReviewResponse,
    evidence: ReviewEvidenceResponse,
) -> ReviewLineResponse | None:
    return review_line_for_scope(
        evidence.scope,
        best_line=review.lines[0] if review.lines else None,
        attempt_line=review.attempt.line if review.attempt else None,
    )


def _causal_move(
    review: PositionReviewResponse,
    evidence: ReviewEvidenceResponse,
) -> str | None:
    line = _canonical_review_line(review, evidence)
    if line is None or evidence.ply >= len(line.moves):
        return None
    return line.moves[evidence.ply].uci


def _proof_matches(claim_proof: str, evidence: ReviewEvidenceResponse) -> bool:
    if claim_proof == "fen":
        return evidence.proof in {"legal_geometry", "direct_rule"}
    if claim_proof == "line":
        return evidence.proof == "line_consequence"
    if claim_proof == "counterfactual":
        return evidence.proof == "counterfactual"
    return False


def _flank_matches(flank: object, move_uci: str) -> bool:
    if flank is None:
        return True
    file_index = chess.square_file(chess.Move.from_uci(move_uci).to_square)
    return {
        "queenside": file_index <= 2,
        "center": 2 <= file_index <= 5,
        "kingside": file_index >= 5,
    }.get(str(flank), False)


def _primary_evidence(review: PositionReviewResponse) -> ReviewEvidenceResponse | None:
    if not review.findings:
        return None
    evidence_by_id = {evidence.id: evidence for evidence in review.evidence}
    first_ids = review.findings[0].evidence_ids
    return evidence_by_id[first_ids[0]] if first_ids else None


def _is_later_primary(evidence) -> bool:
    if evidence is None:
        return False
    if evidence.scope == "best_line":
        return evidence.ply > 0
    if evidence.scope == "attempt_refutation":
        return evidence.ply > 1
    return False


def _is_ordered_subsequence(required: list[str], available: list[str]) -> bool:
    available_iterator = iter(available)
    return all(any(move == candidate for candidate in available_iterator) for move in required)


def _validate_commentary_manifest(manifest: object) -> None:
    try:
        typed_manifest = _CommentaryManifest.model_validate(manifest, strict=True)
    except ValidationError as exc:
        raise ValueError("Expert commentary manifest has invalid or unknown fields") from exc
    manifest = typed_manifest.model_dump(mode="json")
    if manifest["source_policy"] != {
        "annotation_text_redistributed": False,
        "game_scores_redistributed": False,
        "derived_claims_committed": True,
        "article_and_chapter_urls_retained": True,
    }:
        raise ValueError("Expert commentary source policy violates the data contract")
    engine = manifest["engine"]
    if (
        engine["name"] != "Stockfish 18 lite"
        or not engine["analysis_committed"]
        or engine["root_target_ms"] > engine["stability_retry_max_ms"]
        or engine["attempt_max_ms"] > engine["stability_retry_max_ms"]
    ):
        raise ValueError("Expert commentary engine policy is unsupported or contradictory")
    if manifest["quality_scale"] != list(QUALITY_LABELS):
        raise ValueError("Expert commentary quality scale is invalid")
    selection = manifest["selection"]
    collections = manifest["source_collections"]
    gates = manifest["baseline_gates"]
    cases = manifest["cases"]
    case_ids = [case["id"] for case in cases]
    if any(_SAFE_ID.fullmatch(case_id) is None for case_id in case_ids) or len(
        set(case_ids)
    ) != len(case_ids):
        raise ValueError("Expert commentary case IDs must be unique safe strings")
    version = manifest["version"]
    if _SAFE_ID.fullmatch(version) is None:
        raise ValueError("Expert commentary version must be a safe identifier")
    if selection["cases"] != len(cases):
        raise ValueError("Expert commentary case count does not match its selection")

    declared_splits = selection["splits"]
    actual_splits = Counter(case.get("split") for case in cases)
    if dict(actual_splits) != declared_splits:
        raise ValueError("Expert commentary split counts do not match their selection")
    if set(gates) != set(declared_splits):
        raise ValueError("Expert commentary gates do not match declared splits")
    if any(count < 1 for count in declared_splits.values()):
        raise ValueError("Expert commentary split counts must be positive")
    for split, gate in gates.items():
        split_cases = declared_splits[split]
        if not (
            gate["minimum_cases"] <= split_cases
            and gate["minimum_strong"] <= gate["minimum_covered"] <= split_cases
            and gate["maximum_disallowed_primary"] <= split_cases
            and gate["maximum_later_primary"] <= split_cases
        ):
            raise ValueError("Expert commentary gate thresholds contradict their split")

    collection_ids = [collection["id"] for collection in collections]
    if any(_SAFE_ID.fullmatch(collection_id) is None for collection_id in collection_ids) or len(
        set(collection_ids)
    ) != len(collection_ids):
        raise ValueError("Expert commentary collection IDs must be unique safe strings")
    collection_by_id = {collection["id"]: collection for collection in collections}
    collection_sources: dict[str, tuple[str, str]] = {}
    study_urls: set[str] = set()
    study_hashes: set[str] = set()
    for collection in collections:
        study_url = _normalized_study_url(collection["study_url"])
        study_pgn_url = canonical_expert_source_url(collection["study_pgn_url"])
        if study_pgn_url != f"{study_url}.pgn":
            raise ValueError("Expert commentary study PGN URL contradicts its collection")
        study_hash = collection["study_pgn_sha256"]
        if not is_sha256(study_hash):
            raise ValueError("Expert commentary collections require a valid study hash")
        if study_url in study_urls or study_hash in study_hashes:
            raise ValueError("Expert commentary collection sources must be unique")
        study_urls.add(study_url)
        study_hashes.add(study_hash)
        collection_sources[collection["id"]] = (study_url, study_hash)
    collection_splits: dict[tuple[str, str], set[str]] = {}
    game_groups: set[str] = set()
    chapter_hashes: set[str] = set()
    for case, typed_case in zip(cases, typed_manifest.cases, strict=True):
        group = case["group"]
        source = case["source"]
        collection_id = group["source_collection_id"]
        game_id = group["game_id"]
        if collection_id not in collection_by_id:
            raise ValueError("Expert commentary case references an unknown collection")
        collection = collection_by_id[collection_id]
        if collection["split"] != case["split"]:
            raise ValueError("Expert commentary case and collection splits disagree")
        study_url = f"{collection_sources[collection_id][0]}/"
        chapter_url = canonical_expert_source_url(source["chapter_url"])
        chapter_pgn_url = canonical_expert_source_url(source["chapter_pgn_url"])
        if not chapter_url.startswith(study_url) or chapter_pgn_url != f"{chapter_url}.pgn":
            raise ValueError("Expert commentary chapter is outside its source collection")
        if game_id in game_groups:
            raise ValueError("Expert commentary game groups must be unique")
        game_groups.add(game_id)
        chapter_hash = source["chapter_pgn_sha256"]
        annotation_hash = source["annotation_context_sha256"]
        if not is_sha256(chapter_hash) or chapter_hash in chapter_hashes:
            raise ValueError("Expert commentary chapter hashes must be unique SHA-256 values")
        if not is_sha256(annotation_hash):
            raise ValueError("Expert commentary annotations require a valid SHA-256")
        chapter_hashes.add(chapter_hash)
        runtime = case["analysis_runtime"] or {
            "root_ms": engine["root_target_ms"],
            "attempt_ms": engine["attempt_max_ms"],
        }
        if not (
            engine["root_target_ms"] <= runtime["root_ms"] <= engine["stability_retry_max_ms"]
            and runtime["attempt_ms"] <= engine["attempt_max_ms"]
        ):
            raise ValueError("Expert commentary case runtime exceeds its engine policy")
        request = _validated_case_review_request(typed_case)
        collection_splits.setdefault(collection_sources[collection_id], set()).add(
            str(case["split"])
        )
        checked_boards = _checked_analysis_boards(request)
        if request.analysis is None:
            raise ValueError("Expert commentary analysis is missing after validation")
        for claim in typed_case.reference.claims:
            _validate_reference_claim(
                claim,
                analysis=request.analysis,
                checked_boards=checked_boards,
            )
    if any(len(splits) != 1 for splits in collection_splits.values()):
        raise ValueError("Expert commentary collections cannot cross splits")

    expected_collections = selection.get("source_collections")
    expected_groups = selection.get("game_groups")
    actual_collection_counts = Counter(collection["split"] for collection in collections)
    actual_group_counts = Counter(case["split"] for case in cases)
    if not _declared_count_matches(expected_collections, actual_collection_counts):
        raise ValueError("Expert commentary collection counts do not match their selection")
    if not _declared_count_matches(expected_groups, actual_group_counts):
        raise ValueError("Expert commentary game-group counts do not match their selection")


def commentary_case_review_request(case: dict[str, Any]) -> PositionReviewRequest:
    try:
        typed_case = _CommentaryCase.model_validate(case, strict=True)
    except ValidationError as exc:
        raise ValueError("Expert commentary case has invalid or unknown fields") from exc
    return _validated_case_review_request(typed_case)


def _validated_case_review_request(case: _CommentaryCase) -> PositionReviewRequest:
    request = PositionReviewRequest(fen=case.position.fen, analysis=case.analysis)
    board = validate_full_fen(request.fen)
    if request.analysis is None or request.analysis.attempt is None:
        raise ValueError("Expert commentary analysis requires an attempted line")
    move = chess.Move.from_uci(request.analysis.attempt.move)
    if move not in board.legal_moves:
        raise ValueError("Expert commentary attempted move is illegal in its position")
    return request


def _validate_reference_claim(
    reference: _ReferenceClaim,
    *,
    analysis: ReviewAnalysisInput,
    checked_boards: tuple[chess.Board, ...],
) -> None:
    if reference.predicate not in _REFERENCE_PREDICATES:
        raise ValueError("Expert commentary claim has an unknown predicate")
    try:
        parsed_moves = [chess.Move.from_uci(move) for move in (reference.move, *reference.moves)]
        for square in reference.squares:
            chess.parse_square(square)
    except ValueError as exc:
        raise ValueError("Expert commentary claim contains invalid chess geometry") from exc
    if any(move == chess.Move.null() or move.drop is not None for move in parsed_moves):
        raise ValueError("Expert commentary claim contains an invalid move")

    if reference.proof == "line":
        checked_lines = [line.pv for line in analysis.lines]
        if analysis.attempt is not None:
            checked_lines.append(analysis.attempt.line.pv)
        required = [reference.move, *reference.moves]
        if not any(_is_ordered_subsequence(required, line) for line in checked_lines):
            raise ValueError("Expert commentary line claim is absent from checked analysis")
        return

    plan_board = _advanced_board_for_reference_move(checked_boards, parsed_moves[0])
    if plan_board is None:
        raise ValueError("Expert commentary causal move is absent from its checked position")
    for continuation in parsed_moves[1:]:
        if _push_reference_move(plan_board, continuation):
            continue
        fallback = _advanced_board_for_reference_move(checked_boards, continuation)
        if fallback is None:
            raise ValueError("Expert commentary continuation has invalid case geometry")
        plan_board = fallback


def _declared_count_matches(declared: object, actual: Counter[str]) -> bool:
    if isinstance(declared, int):
        return declared == sum(actual.values())
    return isinstance(declared, dict) and declared == dict(actual)


def _checked_analysis_boards(request: PositionReviewRequest) -> tuple[chess.Board, ...]:
    root = validate_full_fen(request.fen)
    boards = [root.copy(stack=False)]
    if request.analysis is None:
        return tuple(boards)
    lines = [line.pv for line in request.analysis.lines]
    if request.analysis.attempt is not None:
        lines.append(request.analysis.attempt.line.pv)
    for line in lines:
        trace = SolutionTrace.build(root, line)
        boards.extend(step.before for step in trace.steps)
    return tuple(boards)


def _advanced_board_for_reference_move(
    boards: tuple[chess.Board, ...],
    move: chess.Move,
) -> chess.Board | None:
    for source in boards:
        board = source.copy(stack=False)
        if _push_reference_move(board, move):
            return board
    return None


def _push_reference_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return False
    board.turn = piece.color
    if move not in board.legal_moves:
        return False
    board.push(move)
    return True


def _collection_identities(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for collection in manifest["source_collections"]:
        identities.add(("url", _normalized_study_url(collection["study_url"])))
        identities.add(("hash", str(collection["study_pgn_sha256"])))
    return identities


def _normalized_study_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expert commentary collections require a study URL")
    return canonical_expert_source_url(value).rstrip("/")


def _annotation_context(node: chess.pgn.ChildNode) -> str:
    parent_comment = " ".join((node.parent.comment if node.parent else "").split())
    comment = " ".join(node.comment.split())
    return " ".join(value for value in (parent_comment, comment) if value)


def _minimum(reasons: list[str], label: str, actual: float, expected: float) -> None:
    if actual < expected:
        reasons.append(f"{label} is {actual}, below {expected}")


def _maximum(reasons: list[str], label: str, actual: float, expected: float) -> None:
    if actual > expected:
        reasons.append(f"{label} is {actual}, above {expected}")
