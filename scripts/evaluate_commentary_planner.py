#!/usr/bin/env python3
"""Stress the constrained coaching contract, with an explicit opt-in live mode."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Any

from chess_scan.commentary_planner import (
    EVIDENCE_PACKET_VERSION,
    PLANNER_VERSION,
    PROMPT_VERSION,
    CommentaryCoach,
    PlannerProviderError,
    ProviderResult,
    eligible_commentary_lessons,
    validate_position_coaching,
)
from chess_scan.config import Settings
from chess_scan.expert_commentary import (
    DEFAULT_MANIFEST_PATH,
    commentary_annotation_quality,
    commentary_case_review_request,
    commentary_quality_rank,
    load_commentary_manifest,
)
from chess_scan.model_artifact import sha256_file
from chess_scan.review import build_position_review
from qa_common import write_json


class _SelectingProvider:
    provider_name = "contract-test"

    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse
        self.model = "scripted-reverse" if reverse else "scripted"

    def complete(self, evidence_packet: dict[str, Any]) -> ProviderResult:
        allowed = list(evidence_packet["allowed_claim_ids"])
        if self.reverse:
            allowed.reverse()
        return ProviderResult(
            raw_output=json.dumps(
                {"claim_ids": allowed[:2], "focus": "cause"},
                separators=(",", ":"),
            ),
            request={"provider": "contract-test", "reverse": self.reverse},
        )


class _UnsupportedProvider:
    provider_name = "contract-test"
    model = "unsupported"

    def complete(self, evidence_packet: dict[str, Any]) -> ProviderResult:
        return ProviderResult(
            raw_output='{"claim_ids":["unsupported"],"focus":"concept"}',
            request={"provider": "contract-test", "case": "unsupported"},
        )


class _FailingProvider:
    provider_name = "contract-test"
    model = "failure"

    def complete(self, evidence_packet: dict[str, Any]) -> ProviderResult:
        raise PlannerProviderError("simulated_timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--confirm-provider-cost",
        action="store_true",
        help="Required with --live because the benchmark makes external calls.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.repetitions > 10:
        raise SystemExit("--repetitions must be between 1 and 10")
    if args.live and not args.confirm_provider_cost:
        raise SystemExit("--live requires --confirm-provider-cost")

    manifest = load_commentary_manifest(args.manifest)
    cases = [case for case in manifest["cases"] if case["split"] == args.split]
    if not cases:
        raise SystemExit(f"No commentary cases in split: {args.split}")
    reviews = [
        (
            case,
            build_position_review(
                commentary_case_review_request(case),
                review_id=f"{index:032x}",
            ),
        )
        for index, case in enumerate(cases, start=1)
    ]

    results = (
        _evaluate_live(reviews, repetitions=args.repetitions)
        if args.live
        else _evaluate_contract(reviews)
    )
    payload = {
        "dataset_version": manifest["version"],
        "manifest_sha256": sha256_file(args.manifest),
        "split": args.split,
        "planner_version": PLANNER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evidence_packet_version": EVIDENCE_PACKET_VERSION,
        **results,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("Commentary planner gate failed")


def _evaluate_contract(reviews) -> dict[str, Any]:
    accepted = 0
    eligible = 0
    grounded = 0
    outcomes: list[dict[str, Any]] = []
    for reverse in (False, True):
        coach = CommentaryCoach(_SelectingProvider(reverse=reverse))
        for case, review in reviews:
            run = coach.plan(review)
            if run.record.error_code == "no_claim_candidates":
                outcomes.append(_run_outcome(case["id"], reverse, run))
                continue
            eligible += 1
            accepted += run.response.status == "accepted"
            validate_position_coaching(
                review,
                run.response.model_dump(mode="json"),
            )
            grounded += 1
            outcomes.append(_run_outcome(case["id"], reverse, run))

    unsupported = CommentaryCoach(_UnsupportedProvider())
    failing = CommentaryCoach(_FailingProvider())
    fallback_checks = 0
    for case, review in reviews:
        for coach in (unsupported, failing):
            run = coach.plan(review)
            if run.record.error_code == "no_claim_candidates":
                outcomes.append(_run_outcome(case["id"], coach.model, run))
                continue
            fallback_checks += 1
            if run.response.status != "fallback":
                return {"passed": False, "reason": "unsafe fallback", "mode": "contract"}
            validate_position_coaching(review, run.response.model_dump(mode="json"))
            outcomes.append(_run_outcome(case["id"], coach.model, run))

    passed = eligible > 0 and accepted == eligible and grounded == eligible and fallback_checks > 0
    return {
        "passed": passed,
        "mode": "contract",
        "reviews": len(reviews),
        "eligible_generations": eligible,
        "accepted_generations": accepted,
        "grounded_generations": grounded,
        "fallback_checks": fallback_checks,
        "outcomes": outcomes,
    }


def _evaluate_live(reviews, *, repetitions: int) -> dict[str, Any]:
    settings = Settings.load()
    coach = CommentaryCoach.from_settings(settings)
    if not coach.enabled:
        raise SystemExit("Live commentary planner is not enabled")

    statuses: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    latencies: list[int] = []
    input_tokens = 0
    output_tokens = 0
    deterministic_quality: Counter[str] = Counter()
    best_available_quality: Counter[str] = Counter()
    grounded = 0
    generations = 0
    ineligible_generations = 0
    semantic_generations = 0
    semantic_non_regressions = 0
    optimal_selections = 0
    outcomes: list[dict[str, Any]] = []
    prepared = []
    for case, review in reviews:
        candidates = eligible_commentary_lessons(review)
        if not candidates:
            prepared.append((case, review, None, None))
            continue
        candidate_qualities = tuple(
            commentary_annotation_quality(case, review, annotation) for annotation in candidates
        )
        prepared.append(
            (
                case,
                review,
                candidate_qualities[0],
                max(candidate_qualities, key=commentary_quality_rank),
            )
        )

    jobs = []
    for repetition in range(1, repetitions + 1):
        for case, review, baseline_quality, best_quality in prepared:
            if baseline_quality is None or best_quality is None:
                ineligible_generations += 1
                outcomes.append(
                    {
                        "case_id": case["id"],
                        "repetition": repetition,
                        "status": "ineligible",
                    }
                )
                continue
            deterministic_quality[baseline_quality] += 1
            best_available_quality[best_quality] += 1
            jobs.append((case, review, repetition, baseline_quality, best_quality))

    aborted_reason: str | None = None
    rate_limits = 0
    executor = ThreadPoolExecutor(
        max_workers=settings.commentary_planner_max_concurrent,
        thread_name_prefix="commentary-evaluation",
    )
    job_iterator = iter(jobs)
    in_flight = {
        executor.submit(_evaluate_live_generation, coach, job)
        for job in islice(job_iterator, settings.commentary_planner_max_concurrent)
    }
    try:
        while in_flight and aborted_reason is None:
            completed, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                run, outcome, selected_quality, baseline_quality, best_quality = future.result()
                generations += 1
                statuses[run.response.status] += 1
                latencies.append(run.record.latency_ms)
                input_tokens += run.record.input_tokens or 0
                output_tokens += run.record.output_tokens or 0
                grounded += 1
                quality[selected_quality] += 1
                outcomes.append(outcome)
                if commentary_quality_rank(best_quality) > 0:
                    semantic_generations += 1
                    non_regression = commentary_quality_rank(
                        selected_quality
                    ) >= commentary_quality_rank(baseline_quality)
                    optimal = commentary_quality_rank(selected_quality) == commentary_quality_rank(
                        best_quality
                    )
                    semantic_non_regressions += non_regression
                    optimal_selections += optimal
                    if not non_regression or not optimal:
                        aborted_reason = "semantic gate is no longer attainable"
                if run.record.error_code in {
                    "http_400",
                    "http_401",
                    "http_403",
                    "http_404",
                    "provider_unavailable",
                }:
                    aborted_reason = f"non-retriable provider error: {run.record.error_code}"
                if run.record.error_code == "http_429":
                    rate_limits += 1
                    time.sleep(min(8, 2 ** (rate_limits - 1)))

            failed = generations - statuses["accepted"]
            allowed_failures = len(jobs) - math.ceil(0.95 * len(jobs))
            if failed > allowed_failures:
                aborted_reason = "acceptance gate is no longer attainable"
            if aborted_reason is None:
                for job in islice(job_iterator, len(completed)):
                    in_flight.add(executor.submit(_evaluate_live_generation, coach, job))
    finally:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        coach.close()

    accepted_rate = statuses["accepted"] / generations if generations else 0
    p95_latency = _percentile(latencies, 0.95) if latencies else 0
    maximum_latency = round(settings.commentary_planner_timeout_seconds * 1000)
    passed = (
        generations == len(jobs)
        and generations > 0
        and grounded == generations
        and accepted_rate >= 0.95
        and p95_latency <= maximum_latency
        and semantic_generations > 0
        and semantic_non_regressions == semantic_generations
        and optimal_selections == semantic_generations
    )
    result = {
        "passed": passed,
        "mode": "live",
        "provider": coach.provider_name,
        "model": coach.model,
        "reviews": len(reviews),
        "repetitions": repetitions,
        "planned_generations": len(jobs),
        "generations": generations,
        "aborted_reason": aborted_reason,
        "ineligible_generations": ineligible_generations,
        "statuses": dict(statuses),
        "accepted_rate": accepted_rate,
        "grounded_rate": grounded / generations if generations else 0,
        "selected_quality": dict(quality),
        "deterministic_quality": dict(deterministic_quality),
        "best_available_quality": dict(best_available_quality),
        "semantic_generations": semantic_generations,
        "semantic_non_regressions": semantic_non_regressions,
        "optimal_selections": optimal_selections,
        "latency_ms": {
            "median": round(statistics.median(latencies)) if latencies else 0,
            "p95": p95_latency,
            "maximum_allowed": maximum_latency,
        },
        "tokens": {"input": input_tokens, "output": output_tokens},
        "outcomes": outcomes,
    }
    return result


def _evaluate_live_generation(coach: CommentaryCoach, job):
    case, review, repetition, baseline_quality, best_quality = job
    run = coach.plan(review)
    validate_position_coaching(review, run.response.model_dump(mode="json"))
    selected_quality = _selected_quality(case, review, run.response)
    outcome = {
        **_run_outcome(case["id"], repetition, run),
        "selected_quality": selected_quality,
        "deterministic_quality": baseline_quality,
        "best_available_quality": best_quality,
    }
    return run, outcome, selected_quality, baseline_quality, best_quality


def _run_outcome(case_id: str, repetition: object, run) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "repetition": repetition,
        "status": run.response.status,
        "accepted_claim_ids": list(run.record.accepted_claim_ids),
        "error_code": run.record.error_code,
        "latency_ms": run.record.latency_ms,
        "input_tokens": run.record.input_tokens,
        "output_tokens": run.record.output_tokens,
    }


def _selected_quality(case, review, coaching) -> str:
    if coaching.status != "accepted" or not coaching.lesson_ids:
        return "fallback"
    annotations = {annotation.id: annotation for annotation in review.explanation}
    return commentary_annotation_quality(
        case,
        review,
        annotations[coaching.lesson_ids[0]],
    )


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


if __name__ == "__main__":
    main()
