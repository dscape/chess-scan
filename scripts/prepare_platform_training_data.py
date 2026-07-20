#!/usr/bin/env python3
"""Render an external, FEN-labelled screenshot corpus from platform piece assets."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
from PIL import Image, ImageDraw, ImageFont

from chess_scan.board import CLASS_NAMES
from chess_scan.platform_data import (
    DEFAULT_EXPECTED_MANIFEST,
    PLATFORM_THEMES,
    default_data_dir,
    hex_color,
    read_records,
    sha256_file,
    verify_record_images,
)

BOARD_SIZE = 512
SQUARE_SIZE = BOARD_SIZE // 8
DEFAULT_CORPUS_VERSION = "platforms-v1"
COORDINATE_FONT_SHA256 = "d72db21f9242aedd6b917d8549ad5921766b24d5f8d0becfda2ff4c620b3c2e0"
PIECE_FILENAMES = {
    "P": "wp",
    "N": "wn",
    "B": "wb",
    "R": "wr",
    "Q": "wq",
    "K": "wk",
    "p": "bp",
    "n": "bn",
    "b": "bb",
    "r": "br",
    "q": "bq",
    "k": "bk",
}


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    assets_dir = data_dir / "assets"
    real_records_path, real_squares_path, source_files = validate_prerequisites(data_dir)
    styles = discover_styles(assets_dir, allow_inventory_change=args.allow_inventory_change)
    positions = generate_positions(args.positions, seed=args.seed)
    font = coordinate_font()

    with tempfile.TemporaryDirectory(prefix=".platform-render-", dir=data_dir) as temporary:
        generated_dir = Path(temporary) / "generated"
        records: list[dict[str, Any]] = []
        for platform, platform_styles in styles.items():
            themes = PLATFORM_THEMES[platform]
            for style_index, style in enumerate(platform_styles):
                pieces = load_piece_images(assets_dir / platform / style, platform)
                for position_index, fen in enumerate(positions):
                    orientation = "black" if (style_index + position_index) % 2 else "white"
                    theme_name, light, dark = themes[(style_index + position_index) % len(themes)]
                    labels = labels_from_fen(fen, orientation=orientation)
                    output = (
                        generated_dir
                        / "boards"
                        / platform
                        / style
                        / f"{position_index + 1:03d}-{orientation}.png"
                    )
                    render_board(
                        output,
                        labels=labels,
                        pieces=pieces,
                        orientation=orientation,
                        light=light,
                        dark=dark,
                        font=font,
                    )
                    records.append(
                        {
                            "path": str(output.relative_to(generated_dir)),
                            "platform": platform,
                            "piece_style": style,
                            "board_theme": theme_name,
                            "position_id": position_index + 1,
                            "split": "test"
                            if position_index >= args.positions - args.test_positions
                            else "train",
                            "orientation": orientation,
                            "fen": fen,
                            "labels": labels,
                            "sha256": sha256_file(output),
                        }
                    )

        records_path = generated_dir / "records.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
        )
        manifest = build_manifest(
            data_dir=data_dir,
            records=records,
            records_path=records_path,
            real_records_path=real_records_path,
            real_squares_path=real_squares_path,
            source_files=source_files,
            styles=styles,
            positions=args.positions,
            test_positions=args.test_positions,
            seed=args.seed,
            version=args.corpus_version,
        )
        (generated_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        publish_generated_corpus(data_dir, generated_dir)
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--test-positions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
        help="Version written to the generated corpus manifest",
    )
    parser.add_argument(
        "--allow-inventory-change",
        action="store_true",
        help="Allow an intentional piece-style inventory change for a new corpus manifest",
    )
    args = parser.parse_args()
    if args.positions <= args.test_positions or args.test_positions < 1:
        parser.error("positions must be greater than test-positions, which must be positive")
    if not args.corpus_version:
        parser.error("corpus-version must not be empty")
    if args.allow_inventory_change and args.corpus_version == DEFAULT_CORPUS_VERSION:
        parser.error("--allow-inventory-change requires an explicit new --corpus-version")
    return args


def validate_prerequisites(data_dir: Path) -> tuple[Path, Path, list[Path]]:
    real_records_path = data_dir / "real" / "records.jsonl"
    real_squares_path = data_dir / "real" / "squares.jsonl"
    missing = [path for path in (real_records_path, real_squares_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing curated real-data inventories: {missing}")
    for path in (real_records_path, real_squares_path):
        verify_record_images(data_dir, read_records(path))
    source_files = sorted(path for path in (data_dir / "source").glob("*") if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No source provenance files found in {data_dir / 'source'}")
    return real_records_path, real_squares_path, source_files


def discover_styles(
    assets_dir: Path,
    *,
    allow_inventory_change: bool = False,
) -> dict[str, list[str]]:
    expected = None if allow_inventory_change else json.loads(DEFAULT_EXPECTED_MANIFEST.read_text())
    expected_styles: dict[str, set[str]] = {}
    records_path = assets_dir.parent / "records.jsonl"
    if expected is not None and records_path.is_file():
        for record in read_records(records_path):
            expected_styles.setdefault(str(record["platform"]), set()).add(
                str(record["piece_style"])
            )
    styles: dict[str, list[str]] = {}
    for platform in PLATFORM_THEMES:
        platform_dir = assets_dir / platform
        if not platform_dir.is_dir():
            raise FileNotFoundError(f"Missing platform assets directory: {platform_dir}")
        available = sorted(path.name for path in platform_dir.iterdir() if path.is_dir())
        if platform == "chess.com":
            available = [style for style in available if style != "blindfold"]
        elif platform == "lichess":
            available = [style for style in available if style not in {"disguised", "mono"}]
        incomplete = {
            style: [
                symbol
                for symbol in PIECE_FILENAMES
                if not piece_path(platform_dir / style, platform, symbol).is_file()
            ]
            for style in available
        }
        incomplete = {style: symbols for style, symbols in incomplete.items() if symbols}
        if incomplete:
            raise ValueError(f"Incomplete piece styles for {platform}: {incomplete}")
        if expected is not None:
            expected_count = int(expected["platforms"][platform]["piece_styles"])
            if len(available) != expected_count or (
                platform in expected_styles and set(available) != expected_styles[platform]
            ):
                raise ValueError(
                    f"Piece-style inventory changed for {platform}; expected "
                    f"{sorted(expected_styles.get(platform, set())) or expected_count}, "
                    f"found {available}. Use --allow-inventory-change only for an "
                    "intentional corpus revision"
                )
        if not available:
            raise ValueError(f"No complete piece styles found for {platform} in {platform_dir}")
        styles[platform] = available
    return styles


def load_piece_images(assets: Path, platform: str) -> dict[str, Image.Image]:
    return {
        symbol: Image.open(piece_path(assets, platform, symbol))
        .convert("RGBA")
        .resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
        for symbol in PIECE_FILENAMES
    }


def build_manifest(
    *,
    data_dir: Path,
    records: list[dict[str, Any]],
    records_path: Path,
    real_records_path: Path,
    real_squares_path: Path,
    source_files: list[Path],
    styles: dict[str, list[str]],
    positions: int,
    test_positions: int,
    seed: int,
    version: str,
) -> dict[str, Any]:
    return {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "positions": positions,
        "test_positions": test_positions,
        "records": len(records),
        "platforms": {
            platform: {
                "piece_styles": len(platform_styles),
                "boards": sum(record["platform"] == platform for record in records),
            }
            for platform, platform_styles in styles.items()
        },
        "records_sha256": sha256_file(records_path),
        "real_records_sha256": sha256_file(real_records_path),
        "real_squares_sha256": sha256_file(real_squares_path),
        "source_files": [
            {
                "path": str(path.relative_to(data_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "notes": [
            "Images are generated training inputs and are not distributed in the "
            "source repository.",
            "Chess.com artwork remains subject to Chess.com's terms; this corpus is an "
            "operator copy.",
            "Lichess artwork has per-piece-set licensing recorded by the upstream lila repository.",
            "Take Take Take pieces were rendered from its public web client bundle.",
            "Lichess mono/disguised and Chess.com blindfold are excluded because piece "
            "color or identity is intentionally hidden.",
        ],
    }


def publish_generated_corpus(data_dir: Path, generated_dir: Path) -> None:
    names = ("boards", "records.jsonl", "MANIFEST.json")
    previous_dir = generated_dir.parent / "previous"
    previous_dir.mkdir()
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for name in names:
            target = data_dir / name
            if target.exists():
                target.replace(previous_dir / name)
                moved_old.append(name)
        for name in names:
            (generated_dir / name).replace(data_dir / name)
            moved_new.append(name)
    except Exception:
        for name in reversed(moved_new):
            target = data_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        for name in reversed(moved_old):
            (previous_dir / name).replace(data_dir / name)
        raise


def generate_positions(count: int, *, seed: int) -> list[str]:
    rng = random.Random(seed)
    positions: list[str] = []
    seen: set[str] = set()
    while len(positions) < count:
        board = chess.Board()
        target_ply = rng.randrange(4, 72)
        for _ in range(target_ply):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            captures = [move for move in moves if board.is_capture(move)]
            move = rng.choice(captures if captures and rng.random() < 0.35 else moves)
            board.push(move)
        placement = board.board_fen()
        if placement in seen or len(board.piece_map()) < 10:
            continue
        seen.add(placement)
        positions.append(f"{placement} {'w' if board.turn else 'b'} - - 0 1")
    return positions


def labels_from_fen(fen: str, *, orientation: str) -> list[int]:
    labels: list[int] = []
    for rank in fen.split()[0].split("/"):
        for character in rank:
            if character.isdigit():
                labels.extend([0] * int(character))
            else:
                labels.append(CLASS_NAMES.index(character))
    if len(labels) != 64:
        raise ValueError(f"Expected 64 labels in {fen}")
    return labels if orientation == "white" else list(reversed(labels))


def render_board(
    output: Path,
    *,
    labels: list[int],
    pieces: dict[str, Image.Image],
    orientation: str,
    light: str,
    dark: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    board = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE))
    draw = ImageDraw.Draw(board)
    light_rgb = hex_color(light)
    dark_rgb = hex_color(dark)
    for index, label in enumerate(labels):
        row, column = divmod(index, 8)
        background = light_rgb if (row + column) % 2 == 0 else dark_rgb
        box = (
            column * SQUARE_SIZE,
            row * SQUARE_SIZE,
            (column + 1) * SQUARE_SIZE,
            (row + 1) * SQUARE_SIZE,
        )
        draw.rectangle(box, fill=background)
        if label:
            piece = pieces[CLASS_NAMES[label]]
            board.paste(piece, (box[0], box[1]), piece)

    draw_coordinates(draw, orientation=orientation, light=light_rgb, dark=dark_rgb, font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, optimize=True)


def draw_coordinates(
    draw: ImageDraw.ImageDraw,
    *,
    orientation: str,
    light: tuple[int, int, int],
    dark: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    files = "abcdefgh" if orientation == "white" else "hgfedcba"
    ranks = "87654321" if orientation == "white" else "12345678"
    for index in range(8):
        rank_color = dark if index % 2 == 0 else light
        file_color = dark if index % 2 == 1 else light
        draw.text((2, index * SQUARE_SIZE), ranks[index], fill=rank_color, font=font)
        file_name = files[index]
        width = draw.textbbox((0, 0), file_name, font=font)[2]
        draw.text(
            ((index + 1) * SQUARE_SIZE - width - 2, BOARD_SIZE - 14),
            file_name,
            fill=file_color,
            font=font,
        )


def piece_path(assets: Path, platform: str, symbol: str) -> Path:
    stem = PIECE_FILENAMES[symbol]
    if platform == "lichess":
        stem = f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}"
    return assets / f"{stem}.png"


def coordinate_font() -> ImageFont.FreeTypeFont:
    configured = os.getenv("CHESS_SCAN_PLATFORM_FONT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
    for path in candidates:
        if not path.is_file():
            continue
        actual_sha256 = sha256_file(path)
        if actual_sha256 != COORDINATE_FONT_SHA256:
            raise ValueError(
                f"Coordinate font hash changed for {path}: expected "
                f"{COORDINATE_FONT_SHA256}, got {actual_sha256}"
            )
        return ImageFont.truetype(str(path), 11)
    raise FileNotFoundError(
        "Set CHESS_SCAN_PLATFORM_FONT to the hash-verified coordinate font; expected "
        f"SHA-256 {COORDINATE_FONT_SHA256}"
    )


if __name__ == "__main__":
    main()
