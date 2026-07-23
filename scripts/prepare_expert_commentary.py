#!/usr/bin/env python3
"""Download and hash-verify external expert commentary sources without redistributing them."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from chess_scan.expert_commentary import (
    DEFAULT_MANIFEST_PATH,
    load_commentary_manifest,
    verify_commentary_source,
)
from chess_scan.expert_sources import download_verified_expert_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/qa-cache"))
    args = parser.parse_args()

    manifest = load_commentary_manifest(args.manifest)
    cache_root = args.cache_dir.expanduser().resolve()
    destination = _contained_path(cache_root, manifest["version"])
    collections = manifest.get("source_collections", [])
    cases = manifest["cases"]
    jobs = [
        ("collection", index, _prepare_collection, (destination, collection))
        for index, collection in enumerate(collections)
    ]
    jobs.extend(
        ("case", index, _prepare_case, (destination, case)) for index, case in enumerate(cases)
    )
    results = _run_bounded(jobs, max_workers=4)
    verified_collections = [results[("collection", index)] for index in range(len(collections))]
    verified = [results[("case", index)] for index in range(len(cases))]

    print(
        json.dumps(
            {
                "dataset_version": manifest["version"],
                "verified_cases": len(verified),
                "verified_collections": len(verified_collections),
                "cache_dir": str(destination),
                "collections": verified_collections,
                "sources": verified,
            },
            indent=2,
        )
    )


def _run_bounded(
    jobs: list[tuple[str, int, Any, tuple[Any, ...]]],
    *,
    max_workers: int,
) -> dict[tuple[str, int], dict[str, str]]:
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="expert-source",
    )
    iterator = iter(jobs)
    in_flight: dict[Future[dict[str, str]], tuple[str, int]] = {}

    def replenish(count: int) -> None:
        for _index in range(count):
            try:
                kind, index, operation, arguments = next(iterator)
            except StopIteration:
                return
            in_flight[executor.submit(operation, *arguments)] = (kind, index)

    replenish(max_workers)
    results: dict[tuple[str, int], dict[str, str]] = {}
    try:
        while in_flight:
            completed, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                identity = in_flight.pop(future)
                results[identity] = future.result()
            replenish(len(completed))
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results


def _prepare_collection(
    destination: Path,
    collection: dict[str, str],
) -> dict[str, str]:
    path = _contained_path(destination, "collections", f"{collection['id']}.pgn")
    download_verified_expert_source(
        collection["study_pgn_url"],
        collection["study_pgn_sha256"],
        path,
    )
    return {
        "id": collection["id"],
        "study_pgn_sha256": collection["study_pgn_sha256"],
    }


def _prepare_case(destination: Path, case: dict[str, Any]) -> dict[str, str]:
    source = case["source"]
    if not isinstance(source, dict):
        raise ValueError("Expert commentary case source must be a record")
    case_id = str(case["id"])
    path = _contained_path(destination, f"{case_id}.pgn")
    digest = download_verified_expert_source(
        str(source["chapter_pgn_url"]),
        str(source["chapter_pgn_sha256"]),
        path,
    )
    return verify_commentary_source(case, path, actual_file_hash=digest)


def _contained_path(root: Path, *parts: str) -> Path:
    destination = root.joinpath(*parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Expert commentary cache path escapes its root")
    return destination


if __name__ == "__main__":
    main()
