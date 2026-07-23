from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chess_scan.expert_commentary import (
    DEFAULT_MANIFEST_PATH,
    commentary_annotation_quality,
    commentary_case_review_request,
    commentary_gate,
    evaluate_commentary,
    load_commentary_manifest,
    validate_commentary_manifests_disjoint,
    verify_commentary_source,
)
from chess_scan.expert_sources import canonical_expert_source_url
from chess_scan.review import build_position_review

_SHADOW_MANIFEST_PATH = DEFAULT_MANIFEST_PATH.with_name("expert-commentary-shadow-v1.json")
_ARCHIVED_MANIFEST_PATH = DEFAULT_MANIFEST_PATH.with_name("expert-commentary-v1.json")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data",
        "https://user:secret@lichess.org/study/example.pgn",
        "https://lichess.org/study/example.pgn?token=secret",
    ],
)
def test_expert_source_fetch_rejects_noncanonical_urls(url: str) -> None:
    with pytest.raises(ValueError, match="canonical Lichess HTTPS URLs"):
        canonical_expert_source_url(url)


@pytest.mark.parametrize(
    "path",
    [
        "/study/ABCDEFGH/../IJKLMNOP/QRSTUVWX",
        "/study/ABCDEFGH/%2f/QRSTUVWX",
    ],
)
def test_expert_source_urls_reject_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="noncanonical study path"):
        canonical_expert_source_url(f"https://lichess.org{path}")


def test_expert_commentary_baseline_is_reproducible() -> None:
    manifest = load_commentary_manifest()

    metrics = evaluate_commentary(manifest, split="development")

    assert metrics["cases"] == 10
    assert metrics["quality"] == {"strong": 6, "partial": 1, "miss": 3}
    assert metrics["verdict_agreement"] == 1
    assert metrics["best_move_support"] == 1
    assert metrics["disallowed_primary"] == 1
    assert metrics["later_primary"] == 0
    assert metrics["generic_fallbacks"] == 0
    assert metrics["documented_baseline_matches"] == 4
    assert commentary_gate(manifest, metrics) == (True, [])


def test_expert_commentary_validation_baseline_is_frozen() -> None:
    manifest = load_commentary_manifest()

    metrics = evaluate_commentary(manifest, split="validation")

    assert metrics["cases"] == 30
    assert metrics["quality"] == {"strong": 6, "partial": 0, "miss": 24}
    assert metrics["verdict_agreement"] == 1
    assert metrics["best_move_support"] == 1
    assert metrics["disallowed_primary"] == 3
    assert metrics["later_primary"] == 4
    assert metrics["generic_fallbacks"] == 5
    assert metrics["documented_baseline_matches"] == 15
    assert commentary_gate(manifest, metrics) == (True, [])


def test_expert_commentary_shadow_gate_is_frozen() -> None:
    manifest = load_commentary_manifest(_SHADOW_MANIFEST_PATH)

    metrics = evaluate_commentary(manifest, split="shadow")

    assert metrics["cases"] == 10
    assert metrics["quality"] == {"strong": 3, "partial": 0, "miss": 7}
    assert metrics["verdict_agreement"] == 1
    assert metrics["best_move_support"] == 1
    assert metrics["disallowed_primary"] == 1
    assert metrics["later_primary"] == 0
    assert metrics["generic_fallbacks"] == 2
    assert metrics["documented_baseline_matches"] == 10
    assert commentary_gate(manifest, metrics) == (True, [])


def test_expert_commentary_gate_rejects_regressions() -> None:
    manifest = load_commentary_manifest()
    metrics = evaluate_commentary(manifest, split="development")
    metrics["quality"]["strong"] = 5

    passed, reasons = commentary_gate(manifest, metrics)

    assert passed is False
    assert reasons == [
        "strong explanations is 5, below 6",
        "covered explanations is 6, below 7",
    ]


def test_commentary_quality_requires_visible_matching_evidence() -> None:
    manifest = load_commentary_manifest(_SHADOW_MANIFEST_PATH)
    metrics = evaluate_commentary(manifest, split="shadow")
    results = {result["id"]: result for result in metrics["results"]}

    hidden_space = results["wcc-2024-ding-gukesh-11-b4"]
    assert hidden_space["finding_concepts"] == ["clearing", "space"]
    assert hidden_space["visible_finding_concepts"] == ["clearing"]
    assert hidden_space["quality"] == "miss"

    different_move = results["wcc-2024-ding-gukesh-32-g5"]
    assert different_move["primary_concept"] == "space"
    assert different_move["primary_claim_supported"] is False
    assert different_move["quality"] == "miss"

    development = load_commentary_manifest()
    line_only = json.loads(json.dumps(development))
    case = next(
        case for case in line_only["cases"] if case["id"] == "tata-2026-van-foreest-giri-37-ra8"
    )
    case["reference"]["claims"][0]["move"] = case["position"]["annotated_move_uci"]
    result = next(
        result
        for result in evaluate_commentary(line_only, split="development")["results"]
        if result["id"] == case["id"]
    )
    assert case["reference"]["partial_line"] is not None
    assert result["quality"] == "miss"


def test_commentary_manifest_validates_split_isolation(tmp_path: Path) -> None:
    manifest = load_commentary_manifest()
    contaminated = json.loads(json.dumps(manifest))
    contaminated["source_collections"][0]["split"] = "validation"

    path = tmp_path / "contaminated-commentary-manifest.json"
    path.write_text(json.dumps(contaminated))
    try:
        with pytest.raises(ValueError, match="case and collection splits disagree"):
            load_commentary_manifest(path)
    finally:
        path.unlink(missing_ok=True)

    shadow = load_commentary_manifest(_SHADOW_MANIFEST_PATH)
    validate_commentary_manifests_disjoint(shadow, [manifest])
    with pytest.raises(ValueError, match="reuse a source collection"):
        validate_commentary_manifests_disjoint(shadow, [shadow])

    mirrored = json.loads(json.dumps(shadow))
    mirrored["source_collections"][0]["study_url"] = "https://lichess.org/study/AAAAAAAA"
    mirrored["source_collections"][0]["study_pgn_sha256"] = "1" * 64
    for index, case in enumerate(mirrored["cases"]):
        case["source"]["chapter_url"] = f"https://lichess.org/study/AAAAAAAA/{index:08d}"
        case["group"]["game_id"] = f"other:{index}"
    with pytest.raises(ValueError, match="reuse a chapter hash"):
        validate_commentary_manifests_disjoint(mirrored, [shadow])


def test_manifest_rejects_attempt_drift_and_unsafe_versions(tmp_path: Path) -> None:
    manifest = load_commentary_manifest()
    drifted = json.loads(json.dumps(manifest))
    drifted["cases"][0]["analysis"]["attempt"]["move"] = "a6b6"
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="invalid or unknown fields"):
        load_commentary_manifest(path)

    drifted = json.loads(json.dumps(manifest))
    drifted["version"] = "../escape"
    path.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="safe identifier"):
        load_commentary_manifest(path)

    unsafe_case = json.loads(json.dumps(manifest))
    unsafe_case["cases"][0]["id"] = "../escape"
    path.write_text(json.dumps(unsafe_case))
    with pytest.raises(ValueError, match="unique safe strings"):
        load_commentary_manifest(path)

    aliased = json.loads(json.dumps(manifest))
    duplicate_collection = {
        **aliased["source_collections"][0],
        "id": "aliased-collection",
    }
    aliased["source_collections"].append(duplicate_collection)
    path.write_text(json.dumps(aliased))
    with pytest.raises(ValueError, match="collection sources must be unique"):
        load_commentary_manifest(path)

    misspelled = json.loads(json.dumps(manifest))
    misspelled["cases"][0]["reference"]["claims"][0]["predicate"] = "prevent-plan"
    path.write_text(json.dumps(misspelled))
    with pytest.raises(ValueError, match="unknown predicate"):
        load_commentary_manifest(path)

    misspelled_field = json.loads(json.dumps(manifest))
    misspelled_field["cases"][0]["reference"]["claims"][0]["sqaure"] = "e4"
    path.write_text(json.dumps(misspelled_field))
    with pytest.raises(ValueError, match="invalid or unknown fields"):
        load_commentary_manifest(path)

    malformed_hash = json.loads(json.dumps(manifest))
    malformed_hash["cases"][0]["source"]["annotation_context_sha256"] = "z" * 64
    path.write_text(json.dumps(malformed_hash))
    with pytest.raises(ValueError, match="annotations require a valid SHA-256"):
        load_commentary_manifest(path)

    unreachable_line_claim = json.loads(json.dumps(manifest))
    line_claim = next(
        claim
        for case in unreachable_line_claim["cases"]
        for claim in case["reference"]["claims"]
        if claim["proof"] == "line"
    )
    line_claim.setdefault("moves", []).append("a2a3")
    path.write_text(json.dumps(unreachable_line_claim))
    with pytest.raises(ValueError, match="line claim is absent"):
        load_commentary_manifest(path)

    invalid_analysis = json.loads(json.dumps(manifest))
    invalid_analysis["cases"][0]["analysis"]["lines"][0]["wdl"] = [1001, 0, 0]
    path.write_text(json.dumps(invalid_analysis))
    with pytest.raises(ValueError, match="invalid or unknown fields"):
        load_commentary_manifest(path)

    missing_reference_field = json.loads(json.dumps(manifest))
    del missing_reference_field["cases"][0]["reference"]["acceptable_best_moves"]
    path.write_text(json.dumps(missing_reference_field))
    with pytest.raises(ValueError, match="invalid or unknown fields"):
        load_commentary_manifest(path)

    invalid_gate = json.loads(json.dumps(manifest))
    invalid_gate["baseline_gates"]["validation"]["minimum_strong"] = "5"
    path.write_text(json.dumps(invalid_gate))
    with pytest.raises(ValueError, match="invalid or unknown fields"):
        load_commentary_manifest(path)

    redistributed = json.loads(json.dumps(manifest))
    redistributed["source_policy"]["annotation_text_redistributed"] = True
    path.write_text(json.dumps(redistributed))
    with pytest.raises(ValueError, match="source policy"):
        load_commentary_manifest(path)

    unsupported_engine = json.loads(json.dumps(manifest))
    unsupported_engine["engine"]["analysis_committed"] = False
    path.write_text(json.dumps(unsupported_engine))
    with pytest.raises(ValueError, match="engine policy"):
        load_commentary_manifest(path)

    excessive_runtime = json.loads(json.dumps(manifest))
    excessive_runtime["cases"][0]["analysis_runtime"] = {
        "root_ms": excessive_runtime["engine"]["stability_retry_max_ms"] + 1,
        "attempt_ms": excessive_runtime["engine"]["attempt_max_ms"],
    }
    path.write_text(json.dumps(excessive_runtime))
    with pytest.raises(ValueError, match="runtime exceeds"):
        load_commentary_manifest(path)

    with pytest.raises(ValueError, match="Archived"):
        load_commentary_manifest(_ARCHIVED_MANIFEST_PATH)


def test_line_evidence_must_come_from_its_canonical_engine_line() -> None:
    manifest = load_commentary_manifest()
    case = next(
        case for case in manifest["cases"] if case["id"] == "tata-2026-aravindh-niemann-36-nxe5"
    )
    review = build_position_review(commentary_case_review_request(case))
    evidence = next(item for item in review.evidence if item.kind == "temporary_sacrifice")
    annotation = next(item for item in review.explanation if evidence.id in item.evidence_ids)
    tampered = review.model_copy(
        update={
            "evidence": [
                item.model_copy(update={"moves": ["a2a3"]}) if item.id == evidence.id else item
                for item in review.evidence
            ]
        }
    )

    assert commentary_annotation_quality(case, tampered, annotation) == "miss"


def test_counterfactual_evidence_must_replay_legally() -> None:
    manifest = load_commentary_manifest(_SHADOW_MANIFEST_PATH)
    case = next(case for case in manifest["cases"] if case["id"] == "wcc-2024-ding-gukesh-18-rd2")
    review = build_position_review(commentary_case_review_request(case))
    evidence = next(item for item in review.evidence if item.proof == "counterfactual")
    annotation = next(item for item in review.explanation if evidence.id in item.evidence_ids)
    matching_case = json.loads(json.dumps(case))
    matching_case["reference"]["claims"] = [
        {
            "predicate": "prevents-tactic",
            "move": "d1d2",
            "moves": ["f2e2"],
            "proof": "counterfactual",
        }
    ]

    assert commentary_annotation_quality(matching_case, review, annotation) == "strong"
    tampered = review.model_copy(
        update={
            "evidence": [
                item.model_copy(update={"moves": ["d1d2", "a1a8"]})
                if item.id == evidence.id
                else item
                for item in review.evidence
            ]
        }
    )
    assert commentary_annotation_quality(matching_case, tampered, annotation) == "miss"


def test_claim_continuations_are_order_sensitive() -> None:
    manifest = load_commentary_manifest()
    reversed_claim = json.loads(json.dumps(manifest))
    case = next(
        case for case in reversed_claim["cases"] if case["id"] == "tata-2026-oro-lami-20-nf4"
    )
    supporting_claim = next(
        claim
        for claim in case["reference"]["claims"]
        if claim["predicate"] == "supports-pawn-advance"
    )
    supporting_claim["moves"].reverse()

    result = next(
        result
        for result in evaluate_commentary(reversed_claim, split="development")["results"]
        if result["id"] == case["id"]
    )

    assert result["quality"] == "miss"
    assert result["primary_claim_supported"] is False


def test_source_verification_checks_game_position_and_annotation(tmp_path: Path) -> None:
    pgn = """[Event \"Example\"]
[White \"White\"]
[Black \"Black\"]
[Round \"1\"]
[Result \"*\"]

1. e4 {A central first move.} e5 *
"""
    source = tmp_path / "example.pgn"
    source.write_text(pgn)
    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    comment_hash = hashlib.sha256(b"A central first move.").hexdigest()
    case = {
        "id": "example",
        "source": {
            "chapter_pgn_sha256": file_hash,
            "annotation_context_sha256": comment_hash,
        },
        "game": {"white": "White", "black": "Black", "round": "1"},
        "position": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_number": 1,
            "side_to_move": "white",
            "annotated_move_uci": "e2e4",
            "annotated_move_san": "e4",
        },
    }

    assert verify_commentary_source(case, source)["case_id"] == "example"

    case["game"]["round"] = "2"
    with pytest.raises(ValueError, match="Round header changed"):
        verify_commentary_source(case, source)
    case["game"]["round"] = "1"
    case["source"]["annotation_context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="annotation changed"):
        verify_commentary_source(case, source)


def test_expert_commentary_manifest_contains_no_source_prose() -> None:
    manifest = load_commentary_manifest()
    serialized = json.dumps(manifest)

    assert manifest["source_policy"]["annotation_text_redistributed"] is False
    assert manifest["selection"]["splits"] == {"development": 10, "validation": 30}
    assert all(len(case["source"]["chapter_pgn_sha256"]) == 64 for case in manifest["cases"])
    assert all(len(case["source"]["annotation_context_sha256"]) == 64 for case in manifest["cases"])
    assert all(len(source["study_pgn_sha256"]) == 64 for source in manifest["source_collections"])
    assert len({case["id"] for case in manifest["cases"]}) == 40
    assert len({case["group"]["game_id"] for case in manifest["cases"]}) == 40
    assert not (
        {case["group"]["game_id"] for case in manifest["cases"] if case["split"] == "development"}
        & {case["group"]["game_id"] for case in manifest["cases"] if case["split"] == "validation"}
    )
    assert "human_comment" not in serialized
    assert "parent_comment" not in serialized
