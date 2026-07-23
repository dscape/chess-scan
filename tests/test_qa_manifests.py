from __future__ import annotations

import hashlib
import json
from collections import Counter

from chess_scan.config import PROJECT_ROOT


def test_online_source_inventory_is_complete_and_portable() -> None:
    path = PROJECT_ROOT / "benchmarks" / "chess-steps-online-sources.json"
    manifest = json.loads(path.read_text())
    sources = manifest["evaluated_sources"]
    assets = manifest["interactive_render_assets"]
    kinds = Counter(source["kind"] for source in sources)

    assert kinds == {
        "localized_workbook_sample_pdf": 70,
        "stepping_stones_web_image": 17,
        "correction_pdf": 1,
        "interactive_position_set": 17,
        "german_manual_sample_pdf": 6,
        "official_manual_sample_pdf": 18,
        "instructional_pdf": 55,
    }
    assert sum(source.get("positions", 0) for source in sources) == 204
    assert len(assets) == 12
    assert len({source["id"] for source in sources}) == len(sources)
    assert len({asset["id"] for asset in assets}) == len(assets)
    assert all(source["url"].startswith("https://") for source in sources + assets)
    assert all(len(source["sha256"]) == 64 for source in sources + assets)
    assert "/tmp/" not in path.read_text()
    assert "/Users/" not in path.read_text()


def test_argus_training_manifest_is_portable_and_complete() -> None:
    manifest = json.loads((PROJECT_ROOT / "benchmarks" / "argus-training-corpus.json").read_text())

    assert manifest["external_only"] is True
    assert manifest["source"]["chess_positions"] == {
        "train": 80_000,
        "test": 20_000,
        "test_real": 13,
    }
    assert manifest["prepared_splits"]["test"]["squares"] == 4_500
    assert len(manifest["source_archive_sha256"]) == 64
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    assert "/Users/" not in json.dumps(manifest)


def test_platform_training_manifest_is_portable_and_complete() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "benchmarks" / "platform-training-corpus.json").read_text()
    )

    assert manifest["version"] == "platforms-v1"
    assert manifest["records"] == 3_984
    assert manifest["platforms"] == {
        "chess.com": {"piece_styles": 38, "boards": 1_824},
        "lichess": {"piece_styles": 40, "boards": 1_920},
        "taketaketake": {"piece_styles": 5, "boards": 240},
    }
    assert len(manifest["records_sha256"]) == 64
    assert len(manifest["real_records_sha256"]) == 64
    assert len(manifest["real_squares_sha256"]) == 64
    assert all(len(record["sha256"]) == 64 for record in manifest["source_files"])
    assert "/Users/" not in json.dumps(manifest)


def test_print_regression_manifest_is_portable_and_complete() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "benchmarks" / "print-regression-corpus.json").read_text()
    )

    assert manifest["version"] == "print-regressions-v1"
    assert manifest["boards"] == manifest["groups"] == 1
    assert manifest["source_images_redistributed"] is False
    assert len(manifest["records_sha256"]) == 64
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    assert "/Users/" not in json.dumps(manifest)


def test_lichess_puzzle_manifest_is_portable_balanced_and_disjoint() -> None:
    manifest = json.loads((PROJECT_ROOT / "benchmarks" / "lichess-puzzle-corpus.json").read_text())

    assert manifest["version"] == "lichess-puzzles-2026-07-05-v1"
    assert manifest["source"]["license"] == "public-domain"
    assert len(manifest["source"]["sha256"]) == 64
    assert len(manifest["themes"]) == 10
    assert manifest["selection"]["grouping"] == "lichess_game_id"
    for split in ("development", "validation"):
        record = manifest["splits"][split]
        assert record["puzzles"] == 1_000
        assert set(record["theme_counts"].values()) == {100}
        assert len(record["sha256"]) == 64
    assert "/Users/" not in json.dumps(manifest)


def test_expert_commentary_manifest_is_portable_and_development_only() -> None:
    manifest = json.loads((PROJECT_ROOT / "benchmarks" / "expert-commentary-v1.json").read_text())

    assert manifest["version"] == "expert-commentary-2026-07-22-v1"
    assert manifest["selection"]["splits"] == {"development": 10, "validation": 0}
    assert manifest["source_policy"]["annotation_text_redistributed"] is False
    assert len(manifest["cases"]) == 10
    assert len({case["id"] for case in manifest["cases"]}) == 10
    assert len({case["source"]["chapter_url"] for case in manifest["cases"]}) == 10
    assert all(case["source"]["article_url"].startswith("https://") for case in manifest["cases"])
    assert all(
        case["source"]["chapter_pgn_url"].startswith("https://") for case in manifest["cases"]
    )
    assert "/tmp/" not in json.dumps(manifest)
    assert "/Users/" not in json.dumps(manifest)


def test_qa_results_match_v2_runtime_metadata() -> None:
    results = json.loads((PROJECT_ROOT / "benchmarks" / "qa-2026-07-18.json").read_text())
    metadata = json.loads((PROJECT_ROOT / "models" / "chess-steps-v2.json").read_text())
    artifact = PROJECT_ROOT / "models" / "chess-steps-v2.onnx"
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    accuracy = results["standard_accuracy"]
    adjudication = results["localized_disagreement_adjudication"]

    assert results["runtime_version"] == metadata["version"] == "chess-steps-v2"
    assert results["artifact_sha256"] == metadata["artifact_sha256"] == artifact_sha256
    assert accuracy["localized"]["board_exact_after_adjudication"] == 976
    assert accuracy["localized"]["exact_king_positions"] == 976
    assert accuracy["interactive"]["board_exact"] == 204
    assert accuracy["interactive"]["exact_king_positions"] == 204
    assert accuracy["german_manuals"]["board_exact_after_disagreement_adjudication"] == 63
    assert accuracy["german_manuals"]["exact_king_positions"] == 63
    assert accuracy["combined"]["standard_source_images"] == 1243
    assert accuracy["combined"]["board_exact_after_adjudication"] == 1243
    assert accuracy["combined"]["exact_king_positions"] == 1243
    assert adjudication["boards"] == len(adjudication["items"]) == 49
    assert (
        adjudication["squares"] == sum(len(item["squares"]) for item in adjudication["items"]) == 54
    )
    assert adjudication["base_model_confirmed_squares"] == 45
    assert adjudication["template_confirmed_squares"] == 9
    assert adjudication["final_model_confirmed_squares"] == 54
    assert adjudication["base_model_exact_boards"] == 40
    assert adjudication["final_model_exact_boards"] == 49
    expanded = results["expanded_disagreement_adjudication"]
    assert expanded["base_disagreement_boards"] == 18
    assert expanded["excluded_nonstandard_or_malformed"] == 10
    assert expanded["evaluable_disagreement_boards"] == 8
    assert expanded["final_model_exact_evaluable_boards"] == 6
    assert expanded["remaining_final_errors"] == 2
    manual_adjudication = results["german_manual_disagreement_adjudication"]
    assert manual_adjudication["boards"] == manual_adjudication["squares"] == 1
    assert manual_adjudication["items"][0]["adjudication"] == ("model_label_visually_confirmed")
    manual = json.loads(
        (PROJECT_ROOT / "benchmarks" / "chess-steps-german-manuals.json").read_text()
    )
    assert len(manual["sources"]) == 6
    assert len(manual["diagrams"]) == 63
    assert len(manual["excluded_diagrams"]) == 4
    queen_manifest = json.loads(
        (PROJECT_ROOT / "benchmarks" / "chess-steps-queen-colors.json").read_text()
    )
    assert len(queen_manifest["sources"]) == 167
    assert len(queen_manifest["examples"]) == 4893
    assert [stage["seed"] for stage in queen_manifest["training_stages"]] == [43, 43, 45]
    loader_seeds = [
        stage.get("loader_seed", stage["seed"]) for stage in queen_manifest["training_stages"]
    ]
    assert loader_seeds == [43, 44, 45]
    assert "/tmp/" not in json.dumps(queen_manifest)
    assert "/Users/" not in json.dumps(queen_manifest)
    assert metadata["gates"]["localized_standard_boards"] == 976
    assert metadata["reproduction"]["verified"] is True
    assert metadata["reproduction"]["gates"]["localized_final_exact_after_adjudication"] == 1.0
    assert results["model_reproduction"]["gate_equivalent"] is True
