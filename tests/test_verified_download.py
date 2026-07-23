from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import BinaryIO

import pytest

from chess_scan.verified_download import install_verified_download


def test_installer_hashes_written_bytes_instead_of_trusting_transport(tmp_path: Path) -> None:
    destination = tmp_path / "source.pgn"

    def dishonest_download(target: BinaryIO) -> str:
        target.write(b"wrong bytes")
        return hashlib.sha256(b"expected bytes").hexdigest()

    with pytest.raises(ValueError, match="Pinned source hash changed"):
        install_verified_download(
            source="test-source",
            expected_sha256=hashlib.sha256(b"expected bytes").hexdigest(),
            destination=destination,
            download=dishonest_download,
        )

    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_concurrent_installs_use_isolated_temporary_files(tmp_path: Path) -> None:
    destination = tmp_path / "source.pgn"
    content = b"verified bytes"
    expected = hashlib.sha256(content).hexdigest()
    barrier = Barrier(2)

    def install() -> str:
        def download(target: BinaryIO) -> None:
            barrier.wait(timeout=2)
            target.write(content)

        return install_verified_download(
            source="test-source",
            expected_sha256=expected,
            destination=destination,
            download=download,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: install(), range(2)))

    assert results == [expected, expected]
    assert destination.read_bytes() == content
    assert list(tmp_path.glob("*.part")) == []
