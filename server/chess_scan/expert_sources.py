"""Restricted transport for checksum-pinned external expert PGNs."""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Timer
from typing import BinaryIO, cast
from urllib.parse import urlsplit

from chess_scan.verified_download import install_verified_download

_ALLOWED_SOURCE_HOSTS = frozenset({"lichess.org"})
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_SOURCE_TOTAL_TIMEOUT_SECONDS = 60.0
_SOURCE_DNS_TIMEOUT_SECONDS = 10.0
_SOURCE_CONNECT_TIMEOUT_SECONDS = 10.0


class _TargetIOError(Exception):
    def __init__(self, error: OSError) -> None:
        super().__init__(str(error))
        self.error = error


def download_verified_expert_source(
    url: str,
    expected_sha256: str,
    destination: Path,
) -> str:
    return install_verified_download(
        source=url,
        expected_sha256=expected_sha256,
        destination=destination,
        download=lambda target: _download_expert_source(url, target),
    )


def canonical_expert_source_url(url: str) -> str:
    hostname, port, request_target = _parse_expert_source_url(url)
    port_suffix = "" if port == 443 else f":{port}"
    return f"https://{hostname}{port_suffix}{request_target}"


def _parse_expert_source_url(url: str) -> tuple[str, int, str]:
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise ValueError("Expert source URL contains control characters")
    parsed = urlsplit(url)
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError("Expert source URL has an invalid port") from exc
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme != "https"
        or hostname not in _ALLOWED_SOURCE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port != 443
    ):
        raise ValueError("Expert sources must use canonical Lichess HTTPS URLs")
    try:
        request_target = (parsed.path or "/").encode("ascii").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Expert source URL path must be ASCII") from exc
    if (
        re.fullmatch(
            r"/study/[A-Za-z0-9]{8}(?:/[A-Za-z0-9]{8})?(?:\.pgn)?",
            request_target,
        )
        is None
    ):
        raise ValueError("Expert source URL has a noncanonical study path")

    return hostname, port, request_target


def _resolve_expert_source_url(
    url: str,
) -> tuple[str, int, str, list[tuple[int, int, int, str, tuple]]]:
    hostname, port, request_target = _parse_expert_source_url(url)
    addresses = socket.getaddrinfo(
        hostname,
        port,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    if not addresses:
        raise OSError("Expert source host did not resolve")
    if any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError("Expert source host resolved to a non-public address")
    return hostname, port, request_target, addresses


def _resolve_expert_source_before(
    url: str,
    *,
    deadline: float,
) -> tuple[str, int, str, list[tuple[int, int, int, str, tuple]]]:
    results: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def resolve() -> None:
        try:
            results.put((True, _resolve_expert_source_url(url)))
        except Exception as error:
            results.put((False, error))

    thread = Thread(target=resolve, name="expert-source-dns", daemon=True)
    thread.start()
    timeout = min(_SOURCE_DNS_TIMEOUT_SECONDS, _remaining_source_time(deadline))
    try:
        succeeded, result = results.get(timeout=timeout)
    except Empty as error:
        raise TimeoutError("Expert source DNS deadline exceeded") from error
    if not succeeded:
        if isinstance(result, Exception):
            raise result
        raise OSError("Expert source resolution failed")
    return cast(tuple[str, int, str, list[tuple[int, int, int, str, tuple]]], result)


def _download_expert_source(url: str, target: BinaryIO) -> None:
    deadline = time.monotonic() + _SOURCE_TOTAL_TIMEOUT_SECONDS
    hostname, _port, request_target, addresses = _resolve_expert_source_before(
        url,
        deadline=deadline,
    )
    last_error: OSError | ssl.SSLError | http.client.HTTPException | None = None
    for family, socktype, protocol, _canonical_name, socket_address in addresses:
        target.seek(0)
        target.truncate()
        raw_socket = socket.socket(family, socktype, protocol)
        raw_socket.settimeout(
            min(_SOURCE_CONNECT_TIMEOUT_SECONDS, _remaining_source_time(deadline))
        )
        tls_socket: ssl.SSLSocket | None = None
        active_socket: list[socket.socket] = [raw_socket]

        def close_at_deadline() -> None:
            try:
                active_socket[0].close()
            except OSError:
                pass

        watchdog = Timer(_remaining_source_time(deadline), close_at_deadline)
        watchdog.daemon = True
        watchdog.start()
        try:
            raw_socket.connect(socket_address)
            raw_socket.settimeout(
                min(_SOURCE_CONNECT_TIMEOUT_SECONDS, _remaining_source_time(deadline))
            )
            tls_socket = ssl.create_default_context().wrap_socket(
                raw_socket,
                server_hostname=hostname,
            )
            active_socket[0] = tls_socket
            request = (
                f"GET {request_target} HTTP/1.1\r\n"
                f"Host: {hostname}\r\n"
                "User-Agent: Chess-Scan-QA/1\r\n"
                "Accept: application/x-chess-pgn,text/plain;q=0.9\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            tls_socket.settimeout(
                min(_SOURCE_CONNECT_TIMEOUT_SECONDS, _remaining_source_time(deadline))
            )
            tls_socket.sendall(request)
            response = http.client.HTTPResponse(tls_socket, method="GET")
            response.begin()
            if 300 <= response.status < 400:
                raise ValueError("Expert source redirects are not allowed")
            if response.status != 200:
                raise ValueError(f"Expert source returned HTTP {response.status}")
            size = 0
            read = getattr(response, "read1", response.read)
            while True:
                tls_socket.settimeout(
                    min(_SOURCE_CONNECT_TIMEOUT_SECONDS, _remaining_source_time(deadline))
                )
                chunk = read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_SOURCE_BYTES:
                    raise ValueError("Expert source exceeds the download limit")
                try:
                    target.write(chunk)
                except OSError as exc:
                    raise _TargetIOError(exc) from exc
            try:
                target.flush()
            except OSError as exc:
                raise _TargetIOError(exc) from exc
            return
        except _TargetIOError as exc:
            raise exc.error from exc
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError("Expert source download deadline exceeded") from exc
            last_error = exc
        finally:
            watchdog.cancel()
            if tls_socket is not None:
                tls_socket.close()
            else:
                raw_socket.close()
    if last_error is not None:
        raise last_error
    raise OSError("Expert source connection failed")


def _remaining_source_time(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Expert source download deadline exceeded")
    return remaining
