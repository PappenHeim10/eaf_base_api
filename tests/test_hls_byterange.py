"""EXT-X-BYTERANGE / fragmented-MP4 HLS: many logical segments in one resource.

Before this capability the engine flattened a PeerTube-style playlist into N
identical full-file URLs: every range was dropped, the whole fragmented MP4
would have been fetched once per fragment and the copies concatenated. These
tests pin the replacement end to end:

* parsing: EXT-X-MAP (with and without BYTERANGE), explicit ranges, implicit
  offsets that continue the previous sub-range, and the plain-string return
  for ordinary playlists staying exactly as it was;
* transport: `Range: bytes=<offset>-<offset+length-1>` per request, merged
  with the source's own headers request-locally, engine range winning over a
  caller-supplied Range, wrong-sized payloads rejected;
* the engine: exact byte reconstruction, retries repeating the identical
  range, concurrent ranges staying isolated, resume distinguishing segments
  that share one URL, and stale range-less states being discarded;
* the wire: a real loopback HTTP server, no mocked fetch layer.
"""

import asyncio
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from base_api.base import BaseCore
from base_api.models import HLSSegment, MediaSource
from base_api.modules.config import DownloadConfigHLS, RuntimeConfig
from base_api.modules.errors import PlaylistExtractionError
from base_api.modules.static_functions import load_segment_state, segment_file_path

REFERER = {"Referer": "https://source.example/"}

MASTER = (
    "#EXTM3U\n"
    "#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=640x360\n"
    "media.m3u8\n"
)

# Mirrors the observed PeerTube structure: version 7, VOD, one EXT-X-MAP with
# a range, every fragment a sub-range of the same fragmented MP4, and the last
# range using an implicit offset.
PEERTUBE_PLAYLIST = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:7\n"
    "#EXT-X-TARGETDURATION:4\n"
    "#EXT-X-PLAYLIST-TYPE:VOD\n"
    '#EXT-X-MAP:URI="file-fragmented.mp4",BYTERANGE="1393@0"\n'
    "#EXTINF:4.000,\n"
    "#EXT-X-BYTERANGE:1257668@1393\n"
    "file-fragmented.mp4\n"
    "#EXTINF:4.000,\n"
    "#EXT-X-BYTERANGE:1235731@1259061\n"
    "file-fragmented.mp4\n"
    "#EXTINF:2.520,\n"
    "#EXT-X-BYTERANGE:634541\n"
    "file-fragmented.mp4\n"
    "#EXT-X-ENDLIST\n"
)

#: The deterministic backing resource all reconstruction tests slice from.
BLOB = random.Random(7).randbytes(4096)

# Deliberately non-contiguous ranges: a full-file download, a wrong offset or a
# reordering cannot reproduce this expected byte sequence by accident.
FIXTURE_RANGES = [(0, 64), (200, 100), (1000, 60), (2000, 500)]  # (offset, length)
FIXTURE_PLAYLIST = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:7\n"
    "#EXT-X-TARGETDURATION:4\n"
    "#EXT-X-PLAYLIST-TYPE:VOD\n"
    '#EXT-X-MAP:URI="video-fragmented.mp4",BYTERANGE="64@0"\n'
    "#EXTINF:4.0,\n"
    "#EXT-X-BYTERANGE:100@200\n"
    "video-fragmented.mp4\n"
    "#EXTINF:4.0,\n"
    "#EXT-X-BYTERANGE:60@1000\n"
    "video-fragmented.mp4\n"
    "#EXTINF:4.0,\n"
    "#EXT-X-BYTERANGE:500@2000\n"
    "video-fragmented.mp4\n"
    "#EXT-X-ENDLIST\n"
)
FIXTURE_EXPECTED = b"".join(BLOB[o:o + l] for o, l in FIXTURE_RANGES)


def playlist_core(media_playlist: str):
    """A real core whose text layer serves the fixture playlists."""
    core = BaseCore(RuntimeConfig())
    text_calls: list[tuple[str, dict | None]] = []

    async def fake_fetch_text(url=None, **kwargs):
        text_calls.append((url, kwargs.get("headers")))
        return MASTER if url.endswith("master.m3u8") else media_playlist

    core.fetch_text = fake_fetch_text
    return core, text_calls


def slicing_core(media_playlist: str = FIXTURE_PLAYLIST):
    """A real core over a fake transport that honors Range requests exactly."""
    core, text_calls = playlist_core(media_playlist)
    byte_calls: list[tuple[str, dict | None]] = []

    async def fake_fetch_bytes(url, **kwargs):
        await asyncio.sleep(0)
        headers = kwargs.get("headers") or {}
        byte_calls.append((url, dict(headers)))
        range_value = headers.get("Range")
        if range_value is None:
            return BLOB
        start, end = range_value.removeprefix("bytes=").split("-")
        return BLOB[int(start):int(end) + 1]

    core.fetch_bytes = fake_fetch_bytes
    return core, text_calls, byte_calls


def fixture_config(tmp_path: Path, **overrides) -> DownloadConfigHLS:
    defaults = dict(
        quality="best",
        path=str(tmp_path / "out.mp4"),
        callback=lambda done, total: None,
        media_source=MediaSource(
            url="https://cdn.example/master.m3u8", source_type="HLS", headers=dict(REFERER)
        ),
        remux=False,
    )
    defaults.update(overrides)
    return DownloadConfigHLS(**defaults)


# --- parsing --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peertube_playlist_parses_into_exact_ranged_segments():
    core, _ = playlist_core(PEERTUBE_PLAYLIST)
    source = MediaSource(url="https://pt.example/hls/master.m3u8", source_type="HLS")

    segments = await core.get_segments(source=source, quality="best")

    url = "https://pt.example/hls/file-fragmented.mp4"
    assert segments == [
        HLSSegment(url=url, length=1393, offset=0),          # EXT-X-MAP
        HLSSegment(url=url, length=1257668, offset=1393),
        HLSSegment(url=url, length=1235731, offset=1259061),
        # implicit offset: continues right after the previous sub-range
        HLSSegment(url=url, length=634541, offset=2494792),
    ]


@pytest.mark.asyncio
async def test_map_byterange_without_offset_starts_at_zero():
    playlist = (
        "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:4\n"
        '#EXT-X-MAP:URI="f.mp4",BYTERANGE="128"\n'
        "#EXTINF:4.0,\n#EXT-X-BYTERANGE:100@128\nf.mp4\n#EXT-X-ENDLIST\n"
    )
    core, _ = playlist_core(playlist)
    source = MediaSource(url="https://a.example/master.m3u8", source_type="HLS")

    segments = await core.get_segments(source=source, quality="best")

    assert segments[0] == HLSSegment(url="https://a.example/f.mp4", length=128, offset=0)


@pytest.mark.asyncio
async def test_implicit_offset_without_a_predecessor_is_a_playlist_error():
    playlist = (
        "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:4\n"
        "#EXTINF:4.0,\n#EXT-X-BYTERANGE:100\nf.mp4\n#EXT-X-ENDLIST\n"
    )
    core, _ = playlist_core(playlist)
    source = MediaSource(url="https://a.example/master.m3u8", source_type="HLS")

    with pytest.raises(PlaylistExtractionError):
        await core.get_segments(source=source, quality="best")


@pytest.mark.asyncio
async def test_a_mixed_playlist_keeps_unranged_entries_rangeless():
    playlist = (
        "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:4\n"
        "#EXTINF:4.0,\n#EXT-X-BYTERANGE:100@0\nf.mp4\n"
        "#EXTINF:4.0,\nplain.ts\n#EXT-X-ENDLIST\n"
    )
    core, _ = playlist_core(playlist)
    source = MediaSource(url="https://a.example/master.m3u8", source_type="HLS")

    segments = await core.get_segments(source=source, quality="best")

    assert segments == [
        HLSSegment(url="https://a.example/f.mp4", length=100, offset=0),
        HLSSegment(url="https://a.example/plain.ts"),
    ]


@pytest.mark.asyncio
async def test_an_ordinary_playlist_still_returns_plain_url_strings():
    """The pre-byterange public contract: unranged playlists are string lists."""
    playlist = (
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n"
        "#EXTINF:4.0,\nseg0.ts\n#EXTINF:4.0,\nseg1.ts\n#EXT-X-ENDLIST\n"
    )
    core, _ = playlist_core(playlist)
    source = MediaSource(url="https://a.example/master.m3u8", source_type="HLS")

    segments = await core.get_segments(source=source, quality="best")

    assert segments == ["https://a.example/seg0.ts", "https://a.example/seg1.ts"]
    assert all(isinstance(entry, str) for entry in segments)


@pytest.mark.asyncio
async def test_ranged_segment_lists_are_not_served_from_the_segment_cache():
    core, text_calls = playlist_core(FIXTURE_PLAYLIST)
    source = MediaSource(url="https://cdn.example/master.m3u8", source_type="HLS")

    first = await core.get_segments(source=source, quality="best")
    calls_after_first = len(text_calls)
    second = await core.get_segments(source=source, quality="best")

    assert first == second
    # The second resolution fetched again instead of trusting a cache that can
    # only store plain URL strings.
    assert len(text_calls) == calls_after_first * 2


# --- Range header construction ----------------------------------------------------


@pytest.mark.parametrize(
    ("length", "offset", "expected"),
    [
        (100, 50, "bytes=50-149"),
        (1, 0, "bytes=0-0"),
        (1257668, 1393, "bytes=1393-1259060"),  # the observed PeerTube fragment
    ],
)
def test_range_header_uses_inclusive_end_semantics(length, offset, expected):
    assert HLSSegment(url="u", length=length, offset=offset).range_header == expected


def test_a_plain_segment_has_no_range_header():
    segment = HLSSegment(url="u")
    assert segment.has_range is False
    assert segment.range_header is None


# --- download_segment transport behavior ------------------------------------------


@pytest.mark.asyncio
async def test_range_request_carries_source_headers_and_range():
    core = BaseCore(RuntimeConfig())
    seen = {}

    async def fake_fetch_bytes(url, **kwargs):
        seen.update(kwargs.get("headers") or {})
        return BLOB[50:150]

    core.fetch_bytes = fake_fetch_bytes
    source_headers = dict(REFERER)

    url, content, ok = await core.download_segment(
        "https://cdn.example/f.mp4", 10, None, headers=source_headers, byte_range=(50, 100)
    )

    assert ok and content == BLOB[50:150]
    assert seen == {"Referer": "https://source.example/", "Range": "bytes=50-149"}
    # Request-local: the source's own dict gained nothing.
    assert source_headers == REFERER


@pytest.mark.asyncio
async def test_the_playlist_range_wins_over_a_range_in_source_headers():
    core = BaseCore(RuntimeConfig())
    seen = {}

    async def fake_fetch_bytes(url, **kwargs):
        seen.update(kwargs.get("headers") or {})
        return BLOB[0:10]

    core.fetch_bytes = fake_fetch_bytes

    await core.download_segment(
        "https://cdn.example/f.mp4", 10, None,
        headers={"Referer": "https://source.example/", "range": "bytes=999-1999"},
        byte_range=(0, 10),
    )

    ranges = {k: v for k, v in seen.items() if k.lower() == "range"}
    assert ranges == {"Range": "bytes=0-9"}


@pytest.mark.asyncio
async def test_a_wrong_sized_payload_is_a_failure_not_data():
    """A server ignoring Range answers with the full resource; accepting that
    once per fragment would corrupt the output."""
    core = BaseCore(RuntimeConfig())

    async def fake_fetch_bytes(url, **kwargs):
        return BLOB  # full file instead of the requested 100 bytes

    core.fetch_bytes = fake_fetch_bytes

    url, content, ok = await core.download_segment(
        "https://cdn.example/f.mp4", 10, None, byte_range=(0, 100)
    )

    assert ok is False
    assert content == b""


# --- the engine: reconstruction, retries, concurrency, resume ----------------------


@pytest.mark.asyncio
async def test_exact_reconstruction_of_init_plus_ordered_ranges(tmp_path):
    core, _, byte_calls = slicing_core()

    result = await core.download(fixture_config(tmp_path))

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == FIXTURE_EXPECTED
    # Exactly one request per fragment, in the playlist's ranges - and none
    # without a Range, so the full resource was never downloaded.
    assert [h.get("Range") for _, h in byte_calls] == [
        "bytes=0-63", "bytes=200-299", "bytes=1000-1059", "bytes=2000-2499"
    ]
    assert all(h.get("Referer") == REFERER["Referer"] for _, h in byte_calls)


@pytest.mark.asyncio
async def test_a_transient_range_failure_retries_with_the_identical_request(tmp_path):
    core, _, _ = slicing_core()
    attempts: list[dict] = []
    failed_once = False

    async def flaky_fetch_bytes(url, **kwargs):
        nonlocal failed_once
        headers = dict(kwargs.get("headers") or {})
        attempts.append(headers)
        if headers.get("Range") == "bytes=1000-1059" and not failed_once:
            failed_once = True
            raise ConnectionError("transient")
        start, end = headers["Range"].removeprefix("bytes=").split("-")
        return BLOB[int(start):int(end) + 1]

    core.fetch_bytes = flaky_fetch_bytes

    result = await core.download(fixture_config(tmp_path))

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == FIXTURE_EXPECTED
    retried = [h for h in attempts if h.get("Range") == "bytes=1000-1059"]
    assert len(retried) == 2  # the failure and the retry
    assert retried[0] == retried[1]  # identical URL headers: Range and Referer


@pytest.mark.asyncio
async def test_concurrent_ranges_on_one_url_stay_isolated(tmp_path):
    core, _, byte_calls = slicing_core()
    in_flight = 0
    max_in_flight = 0

    real_fetch_bytes = core.fetch_bytes

    async def tracking_fetch_bytes(url, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.01)  # hold the slot so requests overlap
            return await real_fetch_bytes(url, **kwargs)
        finally:
            in_flight -= 1

    core.fetch_bytes = tracking_fetch_bytes

    result = await core.download(fixture_config(tmp_path))

    assert result is True
    assert max_in_flight > 1, "segments did not actually overlap"
    # Every concurrent request carried exactly its own range, nothing bled over.
    assert sorted(h.get("Range") for _, h in byte_calls) == [
        "bytes=0-63", "bytes=1000-1059", "bytes=200-299", "bytes=2000-2499"
    ]
    assert (tmp_path / "out.mp4").read_bytes() == FIXTURE_EXPECTED


@pytest.mark.asyncio
async def test_resume_skips_done_ranges_but_not_other_ranges_of_the_same_url(tmp_path):
    state_path = tmp_path / "state.json"

    # First run: one specific range fails permanently -> failed + state on disk.
    core, _, byte_calls = slicing_core()
    real_fetch_bytes = core.fetch_bytes

    async def broken_fetch_bytes(url, **kwargs):
        headers = kwargs.get("headers") or {}
        if headers.get("Range") == "bytes=1000-1059":
            raise ConnectionError("still broken")
        return await real_fetch_bytes(url, **kwargs)

    core.fetch_bytes = broken_fetch_bytes

    result = await core.download(fixture_config(tmp_path, segment_state_path=str(state_path)))
    assert result is False
    state = load_segment_state(str(state_path))
    assert state["version"] == 2
    assert state["missing"] == [2]
    assert state["segments"][2] == {
        "url": "https://cdn.example/video-fragmented.mp4", "length": 60, "offset": 1000
    }
    # Three ranges of the shared URL are already on disk as individual files.
    segment_dir = state["segment_dir"]
    done_files = [i for i in range(4) if Path(segment_file_path(segment_dir, i, state["segment_index_width"])).exists()]
    assert done_files == [0, 1, 3]

    # Second run, healed transport: only the missing range may be requested.
    core2, _, byte_calls2 = slicing_core()
    result = await core2.download(fixture_config(tmp_path, segment_state_path=str(state_path)))

    assert result is True
    assert [h.get("Range") for _, h in byte_calls2] == ["bytes=1000-1059"]
    assert (tmp_path / "out.mp4").read_bytes() == FIXTURE_EXPECTED
    assert not state_path.exists()  # completed downloads clean their state


@pytest.mark.asyncio
async def test_a_rangeless_legacy_state_with_repeated_urls_is_discarded(tmp_path):
    """A version-1 state cannot express ranges; one that repeats URLs was
    written for a byte-range playlist by an engine without range support.
    Resuming it would fetch the full resource once per fragment."""
    import json

    state_path = tmp_path / "state.json"
    url = "https://cdn.example/video-fragmented.mp4"
    state_path.write_text(json.dumps({
        "version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": None,
        "m3u8_url": "https://cdn.example/master.m3u8",
        "quality": "best",
        "output_path": str(tmp_path / "out.mp4"),
        "segment_dir": str(tmp_path / "out.mp4.segments"),
        "segment_index_width": 6,
        "start_segment": 0,
        "total": 4,
        "missing": [1, 2, 3],
        "segments": [url, url, url, url],
    }), encoding="utf-8")

    core, _, byte_calls = slicing_core()
    result = await core.download(fixture_config(tmp_path, segment_state_path=str(state_path)))

    assert result is True
    # The stale state was thrown away: the playlist was re-resolved and all
    # four ranges downloaded correctly.
    assert sorted(h.get("Range") for _, h in byte_calls) == [
        "bytes=0-63", "bytes=1000-1059", "bytes=200-299", "bytes=2000-2499"
    ]
    assert (tmp_path / "out.mp4").read_bytes() == FIXTURE_EXPECTED


# --- the wire: a real loopback HTTP server, nothing mocked --------------------------


class _RangeHandler(BaseHTTPRequestHandler):
    recorded: list[tuple[str, str | None, str | None]] = []
    honor_range = True

    def do_GET(self):
        type(self).recorded.append(
            (self.path, self.headers.get("Range"), self.headers.get("Referer"))
        )
        if self.path.endswith("master.m3u8"):
            body = MASTER.encode()
        elif self.path.endswith("media.m3u8"):
            body = FIXTURE_PLAYLIST.replace(
                "video-fragmented.mp4", "/video-fragmented.mp4"
            ).encode()
        else:
            range_value = self.headers.get("Range")
            if range_value and type(self).honor_range:
                start, end = range_value.removeprefix("bytes=").split("-")
                start, end = int(start), int(end)
                body = BLOB[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(BLOB)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = BLOB  # a server that ignores Range
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def range_server():
    _RangeHandler.recorded = []
    _RangeHandler.honor_range = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.mark.asyncio
async def test_loopback_server_receives_ranges_and_the_file_reconstructs(range_server, tmp_path):
    core = BaseCore(RuntimeConfig())
    config = fixture_config(
        tmp_path,
        media_source=MediaSource(
            url=f"{range_server}/master.m3u8", source_type="HLS", headers=dict(REFERER)
        ),
    )

    try:
        result = await core.download(config)
    finally:
        await core.close()

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == FIXTURE_EXPECTED

    data_requests = [r for r in _RangeHandler.recorded if r[0].endswith(".mp4")]
    playlist_requests = [r for r in _RangeHandler.recorded if r[0].endswith(".m3u8")]
    assert sorted(r[1] for r in data_requests) == [
        "bytes=0-63", "bytes=1000-1059", "bytes=200-299", "bytes=2000-2499"
    ]
    # Source headers rode along on every request; no request fetched the full file.
    assert all(r[2] == REFERER["Referer"] for r in data_requests + playlist_requests)
    assert all(r[1] is not None for r in data_requests)


@pytest.mark.asyncio
async def test_loopback_server_ignoring_range_fails_the_download(range_server, tmp_path):
    _RangeHandler.honor_range = False
    core = BaseCore(RuntimeConfig())
    config = fixture_config(
        tmp_path,
        media_source=MediaSource(url=f"{range_server}/master.m3u8", source_type="HLS"),
    )

    try:
        result = await core.download(config)
    finally:
        await core.close()

    assert result is False
    assert not (tmp_path / "out.mp4").exists()
