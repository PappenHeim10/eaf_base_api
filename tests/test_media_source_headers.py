"""Per-source request headers: the transport contract travels with the source.

A provider used to make its media downloadable by mutating the shared session
(`session.headers.update(...)`), which meant the requirement was invisible on
the `MediaSource` and leaked onto every other source downloaded over the same
core. These tests pin the replacement:

* `MediaSource.headers` is plain data - empty by default, copied defensively.
* Every request belonging to a source - master manifest, media playlist,
  variant fallback, each segment including retries - carries that source's
  headers, applied per request.
* Sources without headers behave exactly as before, on the same core, with no
  inheritance from a headered source and no mutation of session defaults.

Precedence, pinned in test_precedence_*: source headers override session
headers for their request; the session itself is never written to.
"""

import asyncio
from pathlib import Path

import pytest

from base_api.base import BaseCore
from base_api.models import MediaSource
from base_api.modules.config import DownloadConfigHLS, RuntimeConfig

MASTER = (
    "#EXTM3U\n"
    "#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360\n"
    "media_360.m3u8\n"
)
MEDIA_PLAYLIST = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:3\n"
    "#EXT-X-TARGETDURATION:4\n"
    "#EXTINF:4.0,\n"
    "seg0.ts\n"
    "#EXTINF:4.0,\n"
    "seg1.ts\n"
    "#EXT-X-ENDLIST\n"
)

REFERER_A = {"Referer": "https://a.example/"}


def _recording_core():
    """A real BaseCore whose network layer records (url, headers) per request."""
    core = BaseCore(RuntimeConfig())
    text_calls: list[tuple[str, dict | None]] = []
    byte_calls: list[tuple[str, dict | None]] = []

    async def fake_fetch_text(url=None, **kwargs):
        await asyncio.sleep(0)  # let concurrent downloads interleave
        text_calls.append((url, kwargs.get("headers")))
        return MASTER if "master" in url else MEDIA_PLAYLIST

    async def fake_fetch_bytes(url, **kwargs):
        await asyncio.sleep(0)
        byte_calls.append((url, kwargs.get("headers")))
        return b"SEGMENTBYTES"

    core.fetch_text = fake_fetch_text
    core.fetch_bytes = fake_fetch_bytes
    return core, text_calls, byte_calls


# --- the model ----------------------------------------------------------------


def test_a_source_without_headers_still_constructs_the_old_way():
    source = MediaSource(url="https://cdn.example/x.m3u8", source_type="HLS")
    assert source.headers == {}


def test_default_headers_are_not_shared_between_sources():
    first = MediaSource(url="https://cdn.example/1.m3u8", source_type="HLS")
    second = MediaSource(url="https://cdn.example/2.m3u8", source_type="HLS")
    first.headers["Referer"] = "https://a.example/"
    assert second.headers == {}


def test_explicit_headers_are_preserved():
    source = MediaSource(
        url="https://cdn.example/x.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )
    assert source.headers == REFERER_A


def test_a_caller_dict_is_copied_not_aliased():
    template = {"Referer": "https://a.example/"}
    first = MediaSource(url="https://cdn.example/1.m3u8", source_type="HLS", headers=template)
    second = MediaSource(url="https://cdn.example/2.m3u8", source_type="HLS", headers=template)

    template["Referer"] = "https://changed.example/"
    first.headers["X-Extra"] = "1"

    assert first.headers == {"Referer": "https://a.example/", "X-Extra": "1"}
    assert second.headers == {"Referer": "https://a.example/"}


# --- precedence at the request boundary ----------------------------------------


def test_precedence_source_headers_override_session_headers_per_request():
    core = BaseCore(RuntimeConfig())
    core.initialize_session()
    # The session stores its keys lowercased; a source writes "Referer". The
    # override must win case-insensitively and leave a single entry, not put a
    # second Referer line on the wire.
    core.session.headers["Referer"] = "https://legacy.example/"

    merged = core._merged_headers({"Referer": "https://a.example/"})

    referer_entries = {k: v for k, v in merged.items() if k.lower() == "referer"}
    assert referer_entries == {"Referer": "https://a.example/"}


def test_precedence_no_override_keeps_session_headers_and_nothing_persists():
    core = BaseCore(RuntimeConfig())
    core.initialize_session()
    core.session.headers["Referer"] = "https://legacy.example/"
    before = dict(core.session.headers)

    core._merged_headers({"Referer": "https://a.example/"})
    after_override = dict(core.session.headers)
    merged_plain = core._merged_headers(None)

    assert after_override == before  # the override never landed on the session
    plain_referers = {k: v for k, v in merged_plain.items() if k.lower() == "referer"}
    assert list(plain_referers.values()) == ["https://legacy.example/"]


# --- propagation: playlists ----------------------------------------------------


@pytest.mark.asyncio
async def test_get_segments_sends_source_headers_on_master_and_playlist():
    core, text_calls, _ = _recording_core()
    source = MediaSource(
        url="https://cdn.a/master.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )

    segments = await core.get_segments(source=source, quality="best")

    assert segments == ["https://cdn.a/seg0.ts", "https://cdn.a/seg1.ts"]
    assert [url for url, _ in text_calls] == [
        "https://cdn.a/master.m3u8",
        "https://cdn.a/media_360.m3u8",
    ]
    assert all(headers == REFERER_A for _, headers in text_calls)
    # get_segments must not have adopted or mutated the source's own dict.
    assert source.headers == REFERER_A


NESTED_MASTER = (
    "#EXTM3U\n"
    "#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360\n"
    "nested_360.m3u8\n"
)


@pytest.mark.asyncio
async def test_get_segments_sends_source_headers_on_the_variant_fallback():
    # The playlist behind the chosen variant turns out to be another master:
    # the engine follows its first sub-playlist, and that third fetch belongs
    # to the source too.
    core = BaseCore(RuntimeConfig())
    calls: list[tuple[str, dict | None]] = []

    async def fake_fetch_text(url=None, **kwargs):
        calls.append((url, kwargs.get("headers")))
        if url.endswith("master.m3u8"):
            return MASTER
        if url.endswith("media_360.m3u8"):
            return NESTED_MASTER
        return MEDIA_PLAYLIST

    core.fetch_text = fake_fetch_text
    source = MediaSource(
        url="https://cdn.a/master.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )

    segments = await core.get_segments(source=source, quality="best")

    assert segments == ["https://cdn.a/seg0.ts", "https://cdn.a/seg1.ts"]
    assert [url for url, _ in calls] == [
        "https://cdn.a/master.m3u8",
        "https://cdn.a/media_360.m3u8",
        "https://cdn.a/nested_360.m3u8",
    ]
    assert all(headers == REFERER_A for _, headers in calls)


@pytest.mark.asyncio
async def test_a_source_without_headers_requests_exactly_as_before():
    core, text_calls, _ = _recording_core()
    source = MediaSource(url="https://cdn.b/master.m3u8", source_type="HLS")

    await core.get_segments(source=source, quality="best")

    assert text_calls and all(headers is None for _, headers in text_calls)


# --- propagation: segments, through the real download ---------------------------


@pytest.mark.asyncio
async def test_download_sends_source_headers_on_every_segment(tmp_path: Path):
    core, text_calls, byte_calls = _recording_core()
    source = MediaSource(
        url="https://cdn.a/master.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )
    out = tmp_path / "out.ts"

    result = await core.download(
        DownloadConfigHLS(
            quality="best",
            path=str(out),
            callback=lambda done, total: None,
            media_source=source,
            remux=False,
        )
    )

    assert result is True
    assert out.read_bytes() == b"SEGMENTBYTES" * 2
    assert all(headers == REFERER_A for _, headers in text_calls)
    assert [url for url, _ in byte_calls] == [
        "https://cdn.a/seg0.ts",
        "https://cdn.a/seg1.ts",
    ]
    assert all(headers == REFERER_A for _, headers in byte_calls)


@pytest.mark.asyncio
async def test_segment_retries_keep_the_source_headers(tmp_path: Path):
    core, _, _ = _recording_core()
    attempts: list[tuple[str, dict | None]] = []

    async def flaky_fetch_bytes(url, **kwargs):
        attempts.append((url, kwargs.get("headers")))
        if url.endswith("seg0.ts") and len([u for u, _ in attempts if u == url]) == 1:
            raise ConnectionError("first attempt fails")
        return b"SEGMENTBYTES"

    core.fetch_bytes = flaky_fetch_bytes
    source = MediaSource(
        url="https://cdn.a/master.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )

    result = await core.download(
        DownloadConfigHLS(
            quality="best",
            path=str(tmp_path / "out.ts"),
            callback=lambda done, total: None,
            media_source=source,
            remux=False,
        )
    )

    assert result is True
    seg0_attempts = [headers for url, headers in attempts if url.endswith("seg0.ts")]
    assert len(seg0_attempts) == 2  # the failure and the retry
    assert all(headers == REFERER_A for headers in seg0_attempts)


# --- isolation ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_headered_source_does_not_leak_into_a_plain_one(tmp_path: Path):
    """The central A2 regression: same core, source A with a Referer, source B
    without - B must not inherit it, and the session must end unchanged."""
    core, text_calls, byte_calls = _recording_core()
    core.initialize_session()
    session_before = dict(core.session.headers)

    source_a = MediaSource(
        url="https://cdn.a/master.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )
    source_b = MediaSource(url="https://cdn.b/master.m3u8", source_type="HLS")

    for source, name in ((source_a, "a.ts"), (source_b, "b.ts")):
        result = await core.download(
            DownloadConfigHLS(
                quality="best",
                path=str(tmp_path / name),
                callback=lambda done, total: None,
                media_source=source,
                remux=False,
            )
        )
        assert result is True

    requests = text_calls + byte_calls
    a_requests = [headers for url, headers in requests if "cdn.a" in url]
    b_requests = [headers for url, headers in requests if "cdn.b" in url]
    assert a_requests and all(headers == REFERER_A for headers in a_requests)
    assert b_requests and all(not headers for headers in b_requests)
    assert dict(core.session.headers) == session_before


@pytest.mark.asyncio
async def test_two_concurrent_downloads_keep_their_own_headers(tmp_path: Path):
    core, text_calls, byte_calls = _recording_core()

    referer_b = {"Referer": "https://b.example/"}
    source_a = MediaSource(
        url="https://cdn.a/master.m3u8", source_type="HLS", headers=dict(REFERER_A)
    )
    source_b = MediaSource(
        url="https://cdn.b/master.m3u8", source_type="HLS", headers=dict(referer_b)
    )

    results = await asyncio.gather(
        core.download(
            DownloadConfigHLS(
                quality="best", path=str(tmp_path / "a.ts"),
                callback=lambda done, total: None, media_source=source_a, remux=False,
            )
        ),
        core.download(
            DownloadConfigHLS(
                quality="best", path=str(tmp_path / "b.ts"),
                callback=lambda done, total: None, media_source=source_b, remux=False,
            )
        ),
    )

    assert results == [True, True]
    for url, headers in text_calls + byte_calls:
        expected = REFERER_A if "cdn.a" in url else referer_b
        assert headers == expected, f"{url} carried {headers}"
