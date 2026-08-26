#!/usr/bin/env python3
"""Report aggregate evidence for the active hidden shadow candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from chess_scan.bootstrap import initialize_database
from chess_scan.config import Settings
from chess_scan.learning import (
    MAX_BOARDS_PER_CLIENT,
    MIN_SHADOW_BOARDS,
    MIN_SHADOW_CLIENTS,
    MIN_SPECIALIST_SHADOW_BOARDS,
    MIN_SPECIALIST_SHADOW_CLIENTS,
    diverse_shadow_rows,
    has_routed_runtime_policy,
    promotion_decision,
    requires_general_label_parity,
    summarize_shadow,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args()

    database = initialize_database(Settings.load())
    cycle = database.active_learning_cycle()
    if cycle is None or cycle["state"] != "shadowing":
        raise SystemExit("No candidate is currently shadowing")
    candidate = database.get_model(str(cycle["candidate_model_version"]))
    baseline = database.get_model(str(cycle["base_model_version"]))
    metadata = json.loads(str(candidate["metadata_json"]))
    baseline_metadata = json.loads(str(baseline["metadata_json"]))
    routed_policy = has_routed_runtime_policy(metadata)
    require_general_parity = requires_general_label_parity(metadata, baseline_metadata)
    evaluated_rows = database.shadow_evaluations(str(cycle["id"]))
    rows = diverse_shadow_rows(
        evaluated_rows,
        max_per_client=MAX_BOARDS_PER_CLIENT,
    )
    summary = summarize_shadow(rows)
    evaluated_summary = summarize_shadow(evaluated_rows)
    passed, reason = promotion_decision(
        summary,
        minimum_boards=MIN_SHADOW_BOARDS,
        minimum_clients=MIN_SHADOW_CLIENTS,
        minimum_specialist_boards=(MIN_SPECIALIST_SHADOW_BOARDS if routed_policy else 0),
        minimum_specialist_clients=(MIN_SPECIALIST_SHADOW_CLIENTS if routed_policy else 0),
        require_general_label_parity=require_general_parity,
    )
    diagnostic_passed, diagnostic_reason = promotion_decision(
        evaluated_summary,
        minimum_boards=0,
        minimum_clients=0,
        minimum_specialist_boards=0,
        minimum_specialist_clients=0,
        require_general_label_parity=require_general_parity,
    )
    scores = [
        float(row["classifier_route_score"])
        for row in rows
        if row.get("classifier_route_score") is not None
    ]
    payload = {
        "cycle_id": str(cycle["id"]),
        "base_model": str(cycle["base_model_version"]),
        "candidate_model": str(cycle["candidate_model_version"]),
        "shadow_started_at": str(cycle["shadow_started_at"]),
        "eligible_now": passed,
        "decision": reason,
        "evaluated_boards": len(evaluated_rows),
        "excluded_from_promotion": len(evaluated_rows) - len(rows),
        "route_score": (
            {
                "count": len(scores),
                "min": min(scores),
                "median": median(scores),
                "max": max(scores),
            }
            if scores
            else None
        ),
        "summary": summary.as_dict(),
        "all_evaluated_diagnostic": {
            "promotion_evidence": False,
            "passed_current_metric_checks": diagnostic_passed,
            "decision": diagnostic_reason,
            "summary": evaluated_summary.as_dict(),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n")
    print(rendered)


if __name__ == "__main__":
    main()
