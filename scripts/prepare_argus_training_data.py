#!/usr/bin/env python3
"""Extract and verify the labeled Argus classifier corpus outside the repository."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from chess_scan.argus_data import (
    EXPECTED_ARCHIVE_SHA256,
    default_data_dir,
    file_record,
    prepare_sampled_split,
    sha256_file,
    validate_source_data,
)

ARCHIVE_MEMBERS = (
    "data/piece_classifier_dataset/",
    "data/chess_positions/",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / "argus-backups/2026-03-29/argus_data.tar.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--skip-archive-hash",
        action="store_true",
        help="Skip the slow archive hash only when validating an already-extracted copy",
    )
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Argus archive not found: {archive}")
    archive_sha256 = None if args.skip_archive_hash else sha256_file(archive)
    if archive_sha256 is not None and archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            f"Argus archive hash mismatch: expected {EXPECTED_ARCHIVE_SHA256}, got {archive_sha256}"
        )

    if not _source_is_complete(output_dir):
        _extract_corpus(archive, output_dir)
    source_summary = validate_source_data(output_dir)
    prepared_dir = output_dir / "prepared"
    split_summary = {
        "train": prepare_sampled_split(
            output_dir / "data" / "chess_positions" / "train",
            prepared_dir,
            samples_per_piece_class=700,
        ),
        "test": prepare_sampled_split(
            output_dir / "data" / "chess_positions" / "test",
            prepared_dir,
            samples_per_piece_class=250,
        ),
    }
    manifest = build_manifest(
        output_dir,
        archive=archive,
        archive_sha256=archive_sha256 or EXPECTED_ARCHIVE_SHA256,
        source_summary=source_summary,
        split_summary=split_summary,
    )
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Prepared external Argus corpus at {output_dir}")


def build_manifest(
    output_dir: Path,
    *,
    archive: Path,
    archive_sha256: str,
    source_summary: dict,
    split_summary: dict,
) -> dict:
    replay_dir = output_dir / "data" / "piece_classifier_dataset"
    prepared_dir = output_dir / "prepared"
    recorded_files = [
        *(replay_dir / name for name in ("images.npy", "labels.npy", "metadata.json")),
        *sorted(prepared_dir.glob("*")),
    ]
    return {
        "version": "argus-2026-03-29-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_archive": archive.name,
        "source_archive_bytes": archive.stat().st_size,
        "source_archive_sha256": archive_sha256,
        "external_only": True,
        "source": source_summary,
        "prepared_splits": split_summary,
        "files": [file_record(path, relative_to=output_dir) for path in recorded_files],
    }


def _source_is_complete(output_dir: Path) -> bool:
    try:
        validate_source_data(output_dir)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _extract_corpus(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(ARCHIVE_MEMBERS):
                continue
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise FileNotFoundError(f"Could not read archive member: {member.name}")
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


if __name__ == "__main__":
    main()
