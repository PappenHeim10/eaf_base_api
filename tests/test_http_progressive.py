"""Progressive HTTP: one media file, one stream, resumable by byte offset.

Everything here runs against the suite's real loopback server (see
`tests.loopback_server`) rather than a mocked fetch layer, because every rule
this transport has is a rule about the wire: which headers go out, what the
status means, whether `Content-Range` really starts where we asked, and what
the bytes on disk look like afterwards.

The pins fall into four groups:

* **transport** - identity encoding, one request per attempt, the caller's file
  name surviving a `Content-Disposition`, per-source headers staying per-source;
* **the resume contract** - the temporary file as the local truth, `If-Range`
  only with a strong ETag, validators compared by us and never delegated to the
  server, a `200` to a range request rewriting from zero rather than appending,
  and exactly one automatic restart before a `ResumeConflict`;
* **sizes** - a short body kept resumable as `IncompleteBody`, an oversized one
  refused and deleted as `OversizedBody`, an unknown total reported as 0;
* **safety** - schemes, userinfo and redirect targets, on the source URL and
  again on every hop.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from base_api.base import BaseCore
from base_api.models import MediaSource
from base_api.modules.config import DownloadConfigHLS, DownloadConfigHTTP, DownloadConfigRAW, RuntimeConfig
from base_api.modules.errors import (
    AccessDeniedError,
    HTTPStatusError,
    IncompleteBody,
    MediaSourceError,
    OversizedBody,
    RequestRetriesExhausted,
    ResourceGone,
    ResumeConflict,
    UnsupportedProtocolError,
)
from base_api.modules.static_functions import (
    build_progressive_state,
    load_progressive_state,
    write_progressive_state,
)
from tests.loopback_server import QuietHandler, serving

#: Two deterministic bodies of equal length. Equal length matters: a test that
#: proves a restart really happened has to rule out "the file is right because
#: the old prefix happened to match", and same-length different-content bodies
#: make a spliced file impossible to mistake for a correct one.
BLOB = random.Random(11).randbytes(4096)
OTHER_BLOB = random.Random(12).randbytes(4096)

STRONG_ETAG = '"v1"'
WEAK_ETAG = 'W/"v1"'
LAST_MODIFIED = "Wed, 01 Jan 2025 00:00:00 GMT"
REFERER = {"Referer": "https://source.example/"}


# --- the server ------------------------------------------------------------------


@dataclass
class MediaPlan:
    """What the loopback server does, per test.

    `script` is consumed one directive per request, so a test says "fail the
    first request, serve the second" by listing exactly that. An empty script
    means every request is served normally, which is the common case.
    """

    body: bytes = BLOB
    etag: str | None = STRONG_ETAG
    last_modified: str | None = LAST_MODIFIED
    #: False makes the server answer 200 with the whole resource even when a
    #: Range was asked for - the single most common broken-server behavior.
    honor_range: bool = True
    #: 206 for every request, Range or not. A server contradicting itself.
    always_partial: bool = False
    #: No Content-Length: the body is framed as chunked, so its total size is
    #: unknowable until the stream ends.
    chunked: bool = False
    #: What Content-Range / Content-Length claim the resource's size is, when
    #: that has to differ from the body actually served.
    total_override: int | None = None
    content_disposition: str | None = None
    script: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.total_override if self.total_override is not None else len(self.body)

    def next_directive(self) -> dict[str, Any]:
        return self.script.pop(0) if self.script else {}


def handler_for(plan: MediaPlan) -> type[QuietHandler]:
    """A handler bound to one plan. Records every request it answers."""

    class Handler(QuietHandler):
        def _write_chunked(self, payload: bytes) -> None:
            for start in range(0, len(payload), 1024):
                piece = payload[start:start + 1024]
                self.wfile.write(f"{len(piece):X}\r\n".encode("ascii"))
                self.wfile.write(piece)
                self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            plan.requests.append({
                "path": self.path,
                "range": self.headers.get("Range"),
                "if_range": self.headers.get("If-Range"),
                "accept_encoding": self.headers.get("Accept-Encoding"),
                "referer": self.headers.get("Referer"),
            })
            step = plan.next_directive()

            location = step.get("location")
            if location is not None:
                self.send_response(int(step.get("status", 302)))
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            status = step.get("status")
            if status is not None:
                self.send_response(int(status))
                if step.get("retry_after") is not None:
                    self.send_header("Retry-After", str(step["retry_after"]))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            start = 0
            partial = plan.always_partial
            range_header = self.headers.get("Range")
            if range_header and plan.honor_range:
                start = int(range_header.split("=", 1)[1].split("-", 1)[0])
                if start >= len(plan.body):
                    # Unsatisfiable: the client already holds at least as many
                    # bytes as there are.
                    self.send_response(416)
                    if not step.get("omit_content_range"):
                        self.send_header("Content-Range", f"bytes */{plan.total}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                partial = True

            payload = plan.body[start:]
            self.send_response(206 if partial else 200)
            self.send_header("Accept-Ranges", "bytes")
            if plan.etag:
                self.send_header("ETag", plan.etag)
            if plan.last_modified:
                self.send_header("Last-Modified", plan.last_modified)
            if plan.content_disposition:
                self.send_header("Content-Disposition", plan.content_disposition)
            if partial:
                content_range = step.get(
                    "content_range", f"bytes {start}-{len(plan.body) - 1}/{plan.total}"
                )
                if content_range is not None:
                    self.send_header("Content-Range", content_range)

            truncate_after = step.get("truncate_after")
            if plan.chunked:
                # A framed short body: the stream ends cleanly, just early.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self._write_chunked(
                    payload if truncate_after is None else payload[:truncate_after]
                )
                return

            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if truncate_after is None:
                self.wfile.write(payload)
                return
            # An unframed short body: the connection dies mid-transfer, which is
            # what an interrupted download actually looks like.
            self.wfile.write(payload[:truncate_after])
            self.close_connection = True

    return Handler


@pytest.fixture
def media_server():
    """Start a server for a plan; yields `(base_url, plan)`."""
    from contextlib import contextmanager

    @contextmanager
    def start(plan: MediaPlan):
        with serving(handler_for(plan)) as base_url:
            yield base_url, plan

    return start


# --- helpers ---------------------------------------------------------------------


def make_core(**overrides: Any) -> BaseCore:
    """A real core whose retry delays are short enough to run in a test."""
    settings = RuntimeConfig()
    settings.request_attempts = 3
    settings.request_retry_initial_delay = 0.01
    settings.request_retry_max_delay = 0.05
    settings.request_retry_jitter = 0.0
    settings.timeout = 10
    for key, value in overrides.items():
        setattr(settings, key, value)
    return BaseCore(settings)


def http_config(tmp_path: Path, url: str, **overrides: Any) -> DownloadConfigHTTP:
    defaults: dict[str, Any] = dict(
        quality="best",
        path=str(tmp_path / "out.mp4"),
        callback=lambda written, total: None,
        media_source=MediaSource(url=url, source_type="HTTP"),
        state_path=str(tmp_path / "state.json"),
        chunk_size=512,
        state_flush_bytes=512,
    )
    defaults.update(overrides)
    return DownloadConfigHTTP(**defaults)


def seed_partial(
    tmp_path: Path,
    url: str,
    *,
    written: int,
    total: int | None = len(BLOB),
    body: bytes = BLOB,
    etag: str | None = STRONG_ETAG,
    etag_weak: bool = False,
    last_modified: str | None = LAST_MODIFIED,
) -> tuple[Path, Path]:
    """Put a partial download on disk exactly as an interrupted run leaves it."""
    target = tmp_path / "out.mp4"
    temp = tmp_path / "out.mp4.tmp"
    state = tmp_path / "state.json"
    temp.write_bytes(body[:written])
    write_progressive_state(
        str(state),
        build_progressive_state(
            url=url,
            output_path=str(target),
            temp_path=str(temp),
            total_size=total,
            downloaded_bytes=written,
            etag=etag,
            etag_weak=etag_weak,
            last_modified=last_modified,
        ),
    )
    return target, state


# --- transport -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_plain_download_writes_the_exact_bytes_and_reports_byte_progress(
    media_server, tmp_path
):
    plan = MediaPlan()
    progress: list[tuple[int, int]] = []
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(
                http_config(
                    tmp_path,
                    f"{base_url}/video.mp4",
                    callback=lambda written, total: progress.append((written, total)),
                )
            )
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    # Bytes, not segments, and against the real total from the first callback on.
    assert progress[0] == (0, len(BLOB))
    assert progress[-1] == (len(BLOB), len(BLOB))
    assert all(total == len(BLOB) for _, total in progress)
    assert [written for written, _ in progress] == sorted(written for written, _ in progress)
    # Nothing left behind.
    assert not (tmp_path / "out.mp4.tmp").exists()
    assert not (tmp_path / "state.json").exists()
    assert len(plan.requests) == 1


@pytest.mark.asyncio
async def test_the_request_asks_for_identity_encoding_and_carries_no_range(
    media_server, tmp_path
):
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert plan.requests[0]["accept_encoding"] == "identity"
    assert plan.requests[0]["range"] is None
    assert plan.requests[0]["if_range"] is None


@pytest.mark.asyncio
async def test_the_output_name_is_the_callers_whatever_the_server_suggests(
    media_server, tmp_path
):
    """A Content-Disposition is remote input: it never names a file here."""
    plan = MediaPlan(content_disposition='attachment; filename="../evil.exe"')
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            await core.download(http_config(tmp_path, f"{base_url}/whatever-name.mp4"))
        finally:
            await core.close()

    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["out.mp4"]


@pytest.mark.asyncio
async def test_source_headers_ride_along_and_stay_with_their_own_source(
    media_server, tmp_path
):
    """The isolation rule the HLS transport already keeps, on this one too."""
    plan = MediaPlan()
    source_a = MediaSource(
        url="", source_type="HTTP", headers=dict(REFERER)
    )
    core = make_core()
    core.initialize_session()
    session_before = dict(core.session.headers)

    with media_server(plan) as (base_url, _):
        source_a.url = f"{base_url}/a.mp4"
        source_b = MediaSource(url=f"{base_url}/b.mp4", source_type="HTTP")
        try:
            for source, name in ((source_a, "a.mp4"), (source_b, "b.mp4")):
                result = await core.download(
                    http_config(
                        tmp_path,
                        source.url,
                        media_source=source,
                        path=str(tmp_path / name),
                        state_path=str(tmp_path / f"{name}.state"),
                    )
                )
                assert result is True
            session_after = dict(core.session.headers)
        finally:
            await core.close()

    by_path = {entry["path"]: entry for entry in plan.requests}
    assert by_path["/a.mp4"]["referer"] == REFERER["Referer"]
    assert by_path["/b.mp4"]["referer"] is None
    # The source's own dict is untouched, and so is the session.
    assert source_a.headers == REFERER
    assert session_after == session_before


# --- the resume contract ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resumed_download_asks_for_the_right_offset_and_appends(
    media_server, tmp_path
):
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=1000)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] == "bytes=1000-"
    # A strong entity tag may be a range precondition.
    assert plan.requests[0]["if_range"] == STRONG_ETAG
    assert len(plan.requests) == 1


@pytest.mark.asyncio
async def test_an_interrupted_download_leaves_a_resumable_state_the_next_run_finishes(
    media_server, tmp_path
):
    plan = MediaPlan(script=[{"truncate_after": 900}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        try:
            # One attempt only, so the interruption ends this call instead of
            # being repaired inside it.
            with pytest.raises(RequestRetriesExhausted):
                await core.download(http_config(tmp_path, url, max_attempts=1))
        finally:
            await core.close()

        temp = tmp_path / "out.mp4.tmp"
        assert temp.exists() and temp.stat().st_size == 900
        state = load_progressive_state(str(tmp_path / "state.json"))
        assert state is not None
        assert state["downloaded_bytes"] == 900
        assert state["etag"] == STRONG_ETAG
        assert state["total_size"] == len(BLOB)

        core2 = make_core()
        try:
            result = await core2.download(http_config(tmp_path, url))
        finally:
            await core2.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[1]["range"] == "bytes=900-"
    assert not (tmp_path / "state.json").exists()


@pytest.mark.asyncio
async def test_an_interruption_is_repaired_inside_one_call(media_server, tmp_path):
    plan = MediaPlan(script=[{"truncate_after": 700}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert [entry["range"] for entry in plan.requests] == [None, "bytes=700-"]


@pytest.mark.asyncio
async def test_a_server_that_ignores_range_rewrites_the_file_from_zero(
    media_server, tmp_path
):
    """The splice this rule exists to prevent: appending a whole second copy."""
    plan = MediaPlan(honor_range=False)
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=1000)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert plan.requests[0]["range"] == "bytes=1000-"
    # Exactly the resource, not 1000 bytes plus the resource.
    assert (tmp_path / "out.mp4").read_bytes() == BLOB


@pytest.mark.asyncio
async def test_a_206_that_starts_at_the_wrong_offset_restarts_at_byte_zero(
    media_server, tmp_path
):
    plan = MediaPlan(script=[{"content_range": f"bytes 3000-{len(BLOB) - 1}/{len(BLOB)}"}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=1000)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert [entry["range"] for entry in plan.requests] == ["bytes=1000-", None]


@pytest.mark.asyncio
async def test_a_206_without_a_parsable_content_range_restarts_at_byte_zero(
    media_server, tmp_path
):
    plan = MediaPlan(script=[{"content_range": "pages 1-2/9"}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=1000)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert [entry["range"] for entry in plan.requests] == ["bytes=1000-", None]


@pytest.mark.asyncio
async def test_a_changed_etag_restarts_at_byte_zero(media_server, tmp_path):
    """A server that answers 206 anyway must not be able to splice two videos."""
    plan = MediaPlan(body=OTHER_BLOB, etag='"v2"')
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        # The partial bytes came from the *old* resource.
        seed_partial(tmp_path, url, written=1000, body=BLOB, etag=STRONG_ETAG)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    # Entirely the new resource: no prefix of the old one survived.
    assert (tmp_path / "out.mp4").read_bytes() == OTHER_BLOB
    assert [entry["range"] for entry in plan.requests] == ["bytes=1000-", None]


@pytest.mark.asyncio
async def test_a_weak_etag_is_never_a_range_precondition_but_still_resumes(
    media_server, tmp_path
):
    plan = MediaPlan(etag=WEAK_ETAG)
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=1000, etag=STRONG_ETAG, etag_weak=True)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] == "bytes=1000-"
    # The one rule: a weak validator never goes out as If-Range.
    assert plan.requests[0]["if_range"] is None


@pytest.mark.asyncio
async def test_a_changed_last_modified_restarts_at_byte_zero(media_server, tmp_path):
    plan = MediaPlan(body=OTHER_BLOB, etag=None, last_modified="Thu, 02 Jan 2025 00:00:00 GMT")
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=1000, body=BLOB, etag=None)
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == OTHER_BLOB
    assert [entry["range"] for entry in plan.requests] == ["bytes=1000-", None]


@pytest.mark.asyncio
async def test_a_changed_total_size_restarts_at_byte_zero(media_server, tmp_path):
    # Same validators, different length: the resource was replaced by one the
    # old bytes cannot be a prefix of.
    plan = MediaPlan(body=OTHER_BLOB, etag=None, last_modified=None, total_override=9000)
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(
            tmp_path, url, written=1000, body=BLOB, etag=None, last_modified=None,
            total=len(BLOB),
        )
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == OTHER_BLOB
    assert [entry["range"] for entry in plan.requests] == ["bytes=1000-", None]


@pytest.mark.asyncio
async def test_416_stating_the_local_size_finalizes_the_file(media_server, tmp_path):
    """The whole file is already on disk; only the move is missing."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=len(BLOB))
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert len(plan.requests) == 1  # no re-download of a file we already have
    assert not (tmp_path / "state.json").exists()


@pytest.mark.asyncio
async def test_416_without_a_content_range_restarts_at_byte_zero(media_server, tmp_path):
    plan = MediaPlan(script=[{"omit_content_range": True}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        seed_partial(tmp_path, url, written=len(BLOB))
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert [entry["range"] for entry in plan.requests] == [f"bytes={len(BLOB)}-", None]


@pytest.mark.asyncio
async def test_only_one_automatic_restart_then_a_resume_conflict(media_server, tmp_path):
    """A server that keeps contradicting itself must not loop forever."""
    plan = MediaPlan(always_partial=True)
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(ResumeConflict):
                await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    # The first request, one restart, and then the error - not a third try.
    assert len(plan.requests) == 2


@pytest.mark.asyncio
async def test_the_state_file_carries_the_progressive_schema(media_server, tmp_path):
    plan = MediaPlan(script=[{"truncate_after": 900}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(RequestRetriesExhausted):
                await core.download(
                    http_config(tmp_path, f"{base_url}/video.mp4", max_attempts=1)
                )
        finally:
            await core.close()

    payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["kind"] == "http-progressive"
    assert payload["url"].endswith("/video.mp4")
    assert payload["temp_path"] == str(tmp_path / "out.mp4.tmp")
    assert payload["output_path"] == str(tmp_path / "out.mp4")
    assert payload["etag_weak"] is False
    assert payload["last_modified"] == LAST_MODIFIED


@pytest.mark.asyncio
async def test_an_hls_state_in_the_same_directory_is_never_read_as_a_progressive_one(
    media_server, tmp_path
):
    """Both transports share `.state/`; the `kind` is what keeps them apart."""
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "m3u8_url": "https://cdn.example/master.m3u8",
        "output_path": str(tmp_path / "out.mp4"),
        "segment_dir": str(tmp_path / "out.mp4.segments"),
        "segments": [{"url": "https://cdn.example/f.mp4", "length": 10, "offset": 0}],
        "missing": [0],
    }), encoding="utf-8")
    (tmp_path / "out.mp4.tmp").write_bytes(BLOB[:1000])

    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] is None  # nothing was resumed from it


@pytest.mark.asyncio
async def test_an_unknown_state_version_is_discarded(media_server, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "version": 99,
        "kind": "http-progressive",
        "url": "http://example.invalid/video.mp4",
        "downloaded_bytes": 1000,
    }), encoding="utf-8")
    (tmp_path / "out.mp4.tmp").write_bytes(BLOB[:1000])

    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] is None


@pytest.mark.asyncio
async def test_a_state_for_another_url_is_discarded(media_server, tmp_path):
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        seed_partial(tmp_path, "http://elsewhere.invalid/other.mp4", written=1000)
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] is None


@pytest.mark.asyncio
async def test_a_temporary_file_without_a_state_is_discarded(media_server, tmp_path):
    """Bytes nothing can vouch for are not a resume base."""
    (tmp_path / "out.mp4.tmp").write_bytes(OTHER_BLOB[:2000])

    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] is None


@pytest.mark.asyncio
async def test_the_file_wins_when_the_state_claims_more_bytes_than_it_holds(
    media_server, tmp_path
):
    """A crash between the file write and the state write, reconciled downwards."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        _, state_path = seed_partial(tmp_path, url, written=1000)
        # The state now claims more than the file actually holds.
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["downloaded_bytes"] = 3000
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] == "bytes=1000-"


@pytest.mark.asyncio
async def test_a_state_behind_the_file_gives_up_only_the_difference(
    media_server, tmp_path
):
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        _, state_path = seed_partial(tmp_path, url, written=2000)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["downloaded_bytes"] = 1500
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            result = await core.download(http_config(tmp_path, url))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] == "bytes=1500-"


# --- sizes -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_total_is_reported_as_zero_and_still_completes(
    media_server, tmp_path
):
    plan = MediaPlan(chunked=True)
    progress: list[tuple[int, int]] = []
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(
                http_config(
                    tmp_path,
                    f"{base_url}/video.mp4",
                    callback=lambda written, total: progress.append((written, total)),
                )
            )
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    # No invented denominator while it is unknown ...
    assert all(total == 0 for _, total in progress[:-1])
    # ... and the truthful one once the stream has ended.
    assert progress[-1] == (len(BLOB), len(BLOB))


@pytest.mark.asyncio
async def test_a_short_body_is_an_incomplete_body_and_stays_resumable(
    media_server, tmp_path
):
    """Framed correctly, just short: not a broken connection, a short resource."""
    plan = MediaPlan(chunked=True, script=[{"truncate_after": 1500}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/video.mp4"
        try:
            with pytest.raises(IncompleteBody) as raised:
                await core.download(
                    http_config(tmp_path, url, expected_size=len(BLOB))
                )
        finally:
            await core.close()

    assert raised.value.written == 1500
    assert raised.value.expected == len(BLOB)
    # The prefix and its state survive, so a later run can continue.
    assert (tmp_path / "out.mp4.tmp").read_bytes() == BLOB[:1500]
    state = load_progressive_state(str(tmp_path / "state.json"))
    assert state is not None and state["downloaded_bytes"] == 1500
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.asyncio
async def test_an_oversized_body_is_refused_and_its_file_removed(media_server, tmp_path):
    """More bytes than the stated total: the file holds data that is not the resource."""
    plan = MediaPlan(chunked=True)
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(OversizedBody) as raised:
                await core.download(
                    http_config(tmp_path, f"{base_url}/video.mp4", expected_size=1000)
                )
        finally:
            await core.close()

    assert raised.value.expected == 1000
    assert raised.value.written > 1000
    # Unlike a short body this is not resumable, so nothing is kept.
    assert not (tmp_path / "out.mp4.tmp").exists()
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "out.mp4").exists()


# --- status classification -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_transient_server_error_is_retried_and_then_succeeds(
    media_server, tmp_path
):
    plan = MediaPlan(script=[{"status": 503}, {"status": 500}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert len(plan.requests) == 3


@pytest.mark.asyncio
async def test_429_waits_for_the_retry_after_the_server_asked_for(
    media_server, tmp_path
):
    plan = MediaPlan(script=[{"status": 429, "retry_after": 2}])
    delays: list[float] = []
    core = make_core()

    async def record(self, delay, stop_event):
        delays.append(delay)
        return False

    with media_server(plan) as (base_url, _):
        with patch.object(BaseCore, "_sleep_unless_stopped", record):
            try:
                result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
            finally:
                await core.close()

    assert result is True
    assert delays == [2.0]  # the server's number, not our backoff curve


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (404, HTTPStatusError),
        (403, AccessDeniedError),
        (401, AccessDeniedError),
        (410, ResourceGone),
        (400, HTTPStatusError),
    ],
)
async def test_terminal_statuses_do_not_retry(media_server, tmp_path, status, error):
    plan = MediaPlan(script=[{"status": status}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(error):
                await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert len(plan.requests) == 1
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.asyncio
async def test_the_retry_budget_is_finite(media_server, tmp_path):
    plan = MediaPlan(script=[{"status": 503}] * 10)
    core = make_core(request_attempts=3)
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(RequestRetriesExhausted):
                await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert len(plan.requests) == 3


# --- safety ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.invalid/video.mp4",
        "https://user:secret@example.invalid/video.mp4",
        "https://user@example.invalid/video.mp4",
        "https:///video.mp4",
        "",
    ],
)
async def test_a_source_url_this_transport_may_not_fetch_is_refused(tmp_path, url):
    core = make_core()
    try:
        with pytest.raises(MediaSourceError):
            await core.download(http_config(tmp_path, url))
    finally:
        await core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    ["file:///etc/passwd", "https://user:secret@elsewhere.invalid/video.mp4"],
)
async def test_a_redirect_this_transport_may_not_follow_is_refused(
    media_server, tmp_path, location
):
    """A hop is a destination a remote party chose, so it is checked again."""
    plan = MediaPlan(script=[{"location": location}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(MediaSourceError):
                await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.asyncio
async def test_an_ordinary_redirect_is_followed(media_server, tmp_path):
    plan = MediaPlan(script=[{"location": "/real-video.mp4"}])
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert [entry["path"] for entry in plan.requests] == ["/video.mp4", "/real-video.mp4"]


@pytest.mark.asyncio
async def test_a_redirect_loop_is_bounded(media_server, tmp_path):
    plan = MediaPlan(script=[{"location": "/video.mp4"}] * 20)
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            with pytest.raises(HTTPStatusError):
                await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
        finally:
            await core.close()

    assert len(plan.requests) == BaseCore._PROGRESSIVE_MAX_REDIRECTS + 1


# --- cancellation ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stop_event_cancels_and_removes_the_partial_file_and_state(
    media_server, tmp_path
):
    plan = MediaPlan()
    stop_event = asyncio.Event()
    core = make_core()

    def stop_once_bytes_flow(written: int, total: int) -> None:
        if written:
            stop_event.set()

    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(
                http_config(
                    tmp_path,
                    f"{base_url}/video.mp4",
                    callback=stop_once_bytes_flow,
                    stop_event=stop_event,
                    chunk_size=64,
                )
            )
        finally:
            await core.close()

    assert result is False
    assert not (tmp_path / "out.mp4").exists()
    assert not (tmp_path / "out.mp4.tmp").exists()
    assert not (tmp_path / "state.json").exists()


@pytest.mark.asyncio
async def test_a_stop_event_set_upfront_never_reaches_the_network(
    media_server, tmp_path
):
    plan = MediaPlan()
    stop_event = asyncio.Event()
    stop_event.set()
    core = make_core()
    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(
                http_config(tmp_path, f"{base_url}/video.mp4", stop_event=stop_event)
            )
        finally:
            await core.close()

    assert result is False
    assert plan.requests == []


@pytest.mark.asyncio
async def test_a_cancelled_download_can_keep_its_progress_when_asked_to(
    media_server, tmp_path
):
    plan = MediaPlan()
    stop_event = asyncio.Event()
    core = make_core()

    with media_server(plan) as (base_url, _):
        try:
            result = await core.download(
                http_config(
                    tmp_path,
                    f"{base_url}/video.mp4",
                    callback=lambda written, total: stop_event.set() if written else None,
                    stop_event=stop_event,
                    chunk_size=64,
                    cleanup_on_stop=False,
                )
            )
        finally:
            await core.close()

    assert result is False
    assert (tmp_path / "out.mp4.tmp").exists()
    state = load_progressive_state(str(tmp_path / "state.json"))
    assert state is not None and state["downloaded_bytes"] > 0


# --- dispatch and finalization ---------------------------------------------------


@pytest.mark.asyncio
async def test_download_dispatches_on_the_configuration_type(media_server, tmp_path):
    plan = MediaPlan()
    core = make_core()
    seen: list[Any] = []

    async def spy(configuration):
        seen.append(configuration)
        return True

    with media_server(plan) as (base_url, _):
        core.progressive_download = spy
        try:
            config = http_config(tmp_path, f"{base_url}/video.mp4")
            assert await core.download(config) is True
        finally:
            await core.close()

    assert seen == [config]


@pytest.mark.asyncio
async def test_a_configuration_without_a_transport_is_an_explicit_error(tmp_path):
    core = make_core()
    try:
        with pytest.raises(TypeError):
            await core.download(DownloadConfigRAW(quality="best", path=str(tmp_path / "x.mp4")))
    finally:
        await core.close()


@pytest.mark.asyncio
async def test_the_progressive_transport_refuses_a_non_http_source(tmp_path):
    core = make_core()
    try:
        with pytest.raises(UnsupportedProtocolError):
            await core.download(
                http_config(
                    tmp_path,
                    "https://cdn.example/master.m3u8",
                    media_source=MediaSource(
                        url="https://cdn.example/master.m3u8", source_type="HLS"
                    ),
                )
            )
    finally:
        await core.close()


@pytest.mark.asyncio
async def test_an_hls_configuration_still_refuses_an_http_source(tmp_path):
    """The pre-existing contract, unchanged by the new dispatch."""
    core = make_core()
    try:
        with pytest.raises(UnsupportedProtocolError):
            await core.download(
                DownloadConfigHLS(
                    quality="best",
                    path=str(tmp_path / "out.mp4"),
                    callback=lambda done, total: None,
                    media_source=MediaSource(
                        url="https://cdn.example/video.mp4", source_type="HTTP"
                    ),
                )
            )
    finally:
        await core.close()


@pytest.mark.asyncio
async def test_the_windows_rename_retry_covers_the_progressive_finalization(
    media_server, tmp_path
):
    """The finished file must not be lost to a momentary sharing violation."""
    plan = MediaPlan()
    calls: list[int] = []
    real_replace = __import__("os").replace

    def flaky_replace(source, target):
        calls.append(1)
        if len(calls) == 1:
            error = PermissionError(32, "locked")
            error.winerror = 32
            raise error
        return real_replace(source, target)

    core = make_core()
    with media_server(plan) as (base_url, _):
        # No state file here: `os.replace` is shared, and the state writer uses
        # it too - this test is about the final move only.
        with patch("base_api.base.os.replace", side_effect=flaky_replace):
            try:
                result = await core.download(
                    http_config(tmp_path, f"{base_url}/video.mp4", state_path=None)
                )
            finally:
                await core.close()

    assert result is True
    assert len(calls) == 2
    assert (tmp_path / "out.mp4").read_bytes() == BLOB


@pytest.mark.asyncio
async def test_a_finished_mp4_is_never_remuxed(media_server, tmp_path):
    """A progressive MP4 is already the file asked for; PyAV never sees it."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        with patch.object(BaseCore, "_convert_ts_to_mp4") as remux:
            try:
                result = await core.download(http_config(tmp_path, f"{base_url}/video.mp4"))
            finally:
                await core.close()

    assert result is True
    remux.assert_not_called()
