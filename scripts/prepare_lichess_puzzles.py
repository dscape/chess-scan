#!/usr/bin/env python3
"""Prepare two fixed, balanced 1,000-puzzle Lichess theme benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import subprocess
from pathlib import Path
from typing import Any

from chess_scan.lichess_puzzles import (
    SUPPORTED_LICHESS_THEMES,
    default_data_dir,
    lichess_game_id,
)
from chess_scan.model_artifact import sha256_file
from qa_common import download_verified

SOURCE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
SOURCE_SHA256 = "5503bfaf5534518ffe3c4c3bb0ac1ae82350d117ad1a52947796096b75e6247e"
SOURCE_BYTES = 302_111_223
SOURCE_UPDATED = "2026-07-05"
SELECTION_VERSION = "balanced-theme-hash-v1"
SELECTION_SEED = "chess-scan-lichess-themes-v1"
MIN_POPULARITY = 80
MIN_PLAYS = 50
CANDIDATES_PER_THEME = 500
PUZZLES_PER_THEME_PER_SPLIT = 100
SPLITS = ("development", "validation")
FIELDS = (
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_dir(),
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = (
        args.archive.expanduser().resolve()
        if args.archive
        else output_dir / "lichess_db_puzzle.csv.zst"
    )
    if args.download:
        download_verified(SOURCE_URL, SOURCE_SHA256, archive)
    elif not archive.is_file():
        raise FileNotFoundError(f"Lichess puzzle archive not found: {archive}")
    _verify_archive(archive)

    selected, scanned, eligible = _select(archive)
    split_manifests: dict[str, Any] = {}
    for split in SPLITS:
        path = output_dir / f"{split}.csv"
        _write_split(path, selected[split])
        split_manifests[split] = {
            "path": path.name,
            "puzzles": len(selected[split]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "theme_counts": {
                theme: sum(item[0] == theme for item in selected[split])
                for theme in sorted(SUPPORTED_LICHESS_THEMES)
            },
        }

    manifest = {
        "version": "lichess-puzzles-2026-07-05-v1",
        "source": {
            "url": SOURCE_URL,
            "updated": SOURCE_UPDATED,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "license": "public-domain",
        },
        "selection": {
            "version": SELECTION_VERSION,
            "seed": SELECTION_SEED,
            "minimum_popularity": MIN_POPULARITY,
            "minimum_plays": MIN_PLAYS,
            "candidates_per_theme": CANDIDATES_PER_THEME,
            "puzzles_per_theme_per_split": PUZZLES_PER_THEME_PER_SPLIT,
            "grouping": "lichess_game_id",
            "source_rows_scanned": scanned,
            "eligible_rows": eligible,
        },
        "themes": sorted(SUPPORTED_LICHESS_THEMES),
        "splits": split_manifests,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def _verify_archive(path: Path) -> None:
    if path.stat().st_size != SOURCE_BYTES or sha256_file(path) != SOURCE_SHA256:
        raise ValueError("Lichess puzzle archive failed checksum verification")


def _select(
    archive: Path,
) -> tuple[dict[str, list[tuple[str, dict[str, str]]]], int, int]:
    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {
        theme: [] for theme in SUPPORTED_LICHESS_THEMES
    }
    process = subprocess.Popen(
        ["zstd", "-dc", str(archive)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to open the Lichess puzzle decompressor")
    reader = csv.DictReader(process.stdout)
    if tuple(reader.fieldnames or ()) != FIELDS:
        process.kill()
        raise ValueError("Unexpected Lichess puzzle archive columns")

    scanned = 0
    eligible = 0
    for row in reader:
        scanned += 1
        if int(row["Popularity"]) < MIN_POPULARITY or int(row["NbPlays"]) < MIN_PLAYS:
            continue
        matching = set(row["Themes"].split()) & SUPPORTED_LICHESS_THEMES
        if not matching:
            continue
        eligible += 1
        for theme in matching:
            key = _selection_key(theme, row["PuzzleId"])
            item = (-key, row["PuzzleId"], row)
            heap = heaps[theme]
            if len(heap) < CANDIDATES_PER_THEME:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if process.wait() != 0:
        raise RuntimeError("Lichess puzzle decompression failed")

    ordered = {
        theme: [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
        for theme, heap in heaps.items()
    }
    selected: dict[str, list[tuple[str, dict[str, str]]]] = {split: [] for split in SPLITS}
    used: set[str] = set()
    used_games: set[str] = set()
    for split in SPLITS:
        for theme in sorted(SUPPORTED_LICHESS_THEMES):
            candidates = [
                row
                for row in ordered[theme]
                if row["PuzzleId"] not in used and lichess_game_id(row["GameUrl"]) not in used_games
            ]
            chosen = candidates[:PUZZLES_PER_THEME_PER_SPLIT]
            if len(chosen) != PUZZLES_PER_THEME_PER_SPLIT:
                raise ValueError(f"Not enough unique {theme} puzzles for {split}")
            selected[split].extend((theme, row) for row in chosen)
            used.update(row["PuzzleId"] for row in chosen)
            used_games.update(lichess_game_id(row["GameUrl"]) for row in chosen)
    return selected, scanned, eligible


def _selection_key(theme: str, puzzle_id: str) -> int:
    payload = f"{SELECTION_SEED}|{theme}|{puzzle_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8])


def _write_split(path: Path, records: list[tuple[str, dict[str, str]]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*FIELDS, "BenchmarkTheme"), lineterminator="\n")
        writer.writeheader()
        for theme, row in records:
            writer.writerow({**row, "BenchmarkTheme": theme})


if __name__ == "__main__":
    main()
