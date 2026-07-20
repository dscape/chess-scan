#!/usr/bin/env python3
"""Render an external, FEN-labelled screenshot corpus from platform piece assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
from PIL import Image, ImageDraw, ImageFont

from chess_scan.board import CLASS_NAMES

BOARD_SIZE = 512
SQUARE_SIZE = BOARD_SIZE // 8
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
PLATFORM_THEMES = {
    "chess.com": (
        ("green", "#edeed1", "#779952"),
        ("blue", "#ececd7", "#4d6d92"),
        ("brown", "#f0d9b5", "#b58863"),
        ("purple", "#efefef", "#8877b7"),
        ("sky", "#efefef", "#c2d7e2"),
        ("tournament", "#ebece8", "#316549"),
        ("bubblegum", "#fff3f3", "#f9cdd3"),
        ("metal", "#c9c9c9", "#6e6e6e"),
    ),
    "lichess": (
        ("brown", "#f0d9b5", "#b58863"),
        ("blue", "#dee3e6", "#8ca2ad"),
        ("green", "#ffffdd", "#86a666"),
        ("purple", "#e5daf0", "#957ab0"),
        ("grey", "#d3d3d3", "#8a8a8a"),
        ("wood", "#d8b170", "#a06b3b"),
    ),
    "taketaketake": (
        ("purple", "#dad9e7", "#aaa0bd"),
        ("violet", "#e5e0ef", "#9588aa"),
        ("neutral", "#e8e8e8", "#aaa7a4"),
    ),
}


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    assets_dir = data_dir / "assets"
    styles = discover_styles(assets_dir)
    positions = generate_positions(args.positions, seed=args.seed)
    font = coordinate_font()
    records: list[dict[str, Any]] = []

    for platform, platform_styles in styles.items():
        themes = PLATFORM_THEMES[platform]
        for style_index, style in enumerate(platform_styles):
            for position_index, fen in enumerate(positions):
                orientation = "black" if (style_index + position_index) % 2 else "white"
                theme_name, light, dark = themes[(style_index + position_index) % len(themes)]
                labels = labels_from_fen(fen, orientation=orientation)
                output = (
                    data_dir
                    / "boards"
                    / platform
                    / style
                    / f"{position_index + 1:03d}-{orientation}.png"
                )
                render_board(
                    output,
                    labels=labels,
                    assets=assets_dir / platform / style,
                    platform=platform,
                    orientation=orientation,
                    light=light,
                    dark=dark,
                    font=font,
                )
                records.append(
                    {
                        "path": str(output.relative_to(data_dir)),
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

    records_path = data_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )
    source_files = sorted(path for path in (data_dir / "source").glob("*") if path.is_file())
    real_records_path = data_dir / "real" / "records.jsonl"
    real_squares_path = data_dir / "real" / "squares.jsonl"
    if not real_records_path.is_file() or not real_squares_path.is_file():
        raise FileNotFoundError(
            "The curated real/records.jsonl and real/squares.jsonl inventories are required"
        )
    manifest = {
        "version": "platforms-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "positions": args.positions,
        "test_positions": args.test_positions,
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
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "chess-scan-training" / "platforms-v1",
    )
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--test-positions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()
    if args.positions <= args.test_positions or args.test_positions < 1:
        parser.error("positions must be greater than test-positions, which must be positive")
    return args


def discover_styles(assets_dir: Path) -> dict[str, list[str]]:
    styles: dict[str, list[str]] = {}
    for platform in PLATFORM_THEMES:
        platform_dir = assets_dir / platform
        available = sorted(path.name for path in platform_dir.iterdir() if path.is_dir())
        if platform == "chess.com":
            available = [style for style in available if style != "blindfold"]
        elif platform == "lichess":
            available = [style for style in available if style not in {"disguised", "mono"}]
        valid = [
            style
            for style in available
            if all(
                piece_path(platform_dir / style, platform, symbol).is_file()
                for symbol in PIECE_FILENAMES
            )
        ]
        if not valid:
            raise ValueError(f"No complete piece styles found for {platform} in {platform_dir}")
        styles[platform] = valid
    return styles


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
    assets: Path,
    platform: str,
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
            symbol = CLASS_NAMES[label]
            piece = Image.open(piece_path(assets, platform, symbol)).convert("RGBA")
            piece = piece.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
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


def coordinate_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), 11)
    return ImageFont.load_default()


def hex_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
