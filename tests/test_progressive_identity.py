"""Resuming across a URL that changed underneath the download.

A signed media URL is not an identity. It expires within hours, and asking the
provider again for the same track yields a different URL for byte-identical
content. Keyed on the URL, the transport threw away a perfectly good partial
file every single time - and had to persist the signed URL, signature and all,
in the user's download folder in order to compare against it.

`MediaSource.identity` is the provider saying "these are the same bytes". These
tests pin both halves of that: that it makes a resume survive a URL change, and
that it changes nothing whatsoever for the providers that supply none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from base_api.models import MediaSource
from base_api.modules.static_functions import build_progressive_state, write_progressive_state
from tests.test_http_progressive import (
    BLOB,
    LAST_MODIFIED,
    STRONG_ETAG,
    MediaPlan,
    http_config,
    make_core,
    media_server,  # noqa: F401 - fixture
)


IDENTITY = "provider:abc123:140"
SIGNED_QUERY = "expire=1788668116&ip=2001-db8-SENTINEL&sig=SIG-SENTINEL-VALUE"


def seed(
    tmp_path: Path,
    url: str,
    *,
    identity: str | None,
    written: int = 1024,
    total: int | None = len(BLOB),
) -> Path:
    """An interrupted download, exactly as the transport leaves one behind."""
    state = tmp_path / "state.json"
    (tmp_path / "out.mp4.tmp").write_bytes(BLOB[:written])
    write_progressive_state(
        str(state),
        build_progressive_state(
            url=url,
            identity=identity,
            output_path=str(tmp_path / "out.mp4"),
            temp_path=str(tmp_path / "out.mp4.tmp"),
            total_size=total,
            downloaded_bytes=written,
            etag=STRONG_ETAG,
            etag_weak=False,
            last_modified=LAST_MODIFIED,
        ),
    )
    return state


def source(url: str, identity: str | None) -> MediaSource:
    return MediaSource(url=url, source_type="HTTP", identity=identity)


@pytest.mark.asyncio
async def test_an_identity_lets_a_resume_survive_a_changed_url(media_server, tmp_path):
    """The case the field exists for: same track, refreshed signature."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        seed(tmp_path, f"{base_url}/media.mp4?expire=1&sig=OLD", identity=IDENTITY)
        result = await core.download(
            http_config(
                tmp_path,
                f"{base_url}/media.mp4?expire=2&sig=NEW",
                media_source=source(f"{base_url}/media.mp4?expire=2&sig=NEW", IDENTITY),
            )
        )

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    # One request, and it asked to continue rather than to start over.
    assert plan.requests[0]["range"] == "bytes=1024-"


@pytest.mark.asyncio
async def test_without_an_identity_a_changed_url_still_restarts(media_server, tmp_path):
    """The behaviour every existing provider has, unchanged."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        seed(tmp_path, f"{base_url}/media.mp4?v=old", identity=None)
        result = await core.download(
            http_config(
                tmp_path,
                f"{base_url}/media.mp4?v=new",
                media_source=source(f"{base_url}/media.mp4?v=new", None),
            )
        )

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB
    assert plan.requests[0]["range"] is None, "a changed URL must start at byte zero"


@pytest.mark.asyncio
async def test_a_different_identity_restarts_even_when_the_url_is_the_same(
    media_server, tmp_path
):
    """The provider says these are different tracks; the URL matching is noise."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/media.mp4"
        seed(tmp_path, url, identity="provider:abc123:139")
        result = await core.download(
            http_config(tmp_path, url, media_source=source(url, "provider:abc123:140"))
        )

    assert result is True
    assert plan.requests[0]["range"] is None


@pytest.mark.asyncio
async def test_a_matching_identity_does_not_override_a_changed_total(media_server, tmp_path):
    """Identity answers "same track", never "same bytes".

    The validators still decide. A provider that reuses an identity for a
    re-encoded track must not get a file spliced from two different encodes.
    """
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/media.mp4?{SIGNED_QUERY}"
        seed(tmp_path, url, identity=IDENTITY, total=len(BLOB) + 999)
        result = await core.download(
            http_config(tmp_path, url, media_source=source(url, IDENTITY))
        )

    assert result is True
    assert (tmp_path / "out.mp4").read_bytes() == BLOB


@pytest.mark.asyncio
async def test_a_state_written_before_identities_existed_is_not_mistaken_for_a_match(
    media_server, tmp_path
):
    """An older state has no identity; a source that now has one must not resume."""
    plan = MediaPlan()
    core = make_core()
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/media.mp4"
        state = tmp_path / "state.json"
        (tmp_path / "out.mp4.tmp").write_bytes(BLOB[:1024])
        legacy = json.loads(
            json.dumps(
                {
                    "version": 1,
                    "kind": "http-progressive",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "url": url,
                    "output_path": str(tmp_path / "out.mp4"),
                    "temp_path": str(tmp_path / "out.mp4.tmp"),
                    "total_size": len(BLOB),
                    "downloaded_bytes": 1024,
                    "etag": STRONG_ETAG,
                    "etag_weak": False,
                    "last_modified": LAST_MODIFIED,
                }
            )
        )
        state.write_text(json.dumps(legacy), encoding="utf-8")

        result = await core.download(
            http_config(tmp_path, url, media_source=source(url, IDENTITY))
        )

    assert result is True
    assert plan.requests[0]["range"] is None


@pytest.mark.asyncio
async def test_the_state_file_never_holds_a_signed_url_once_an_identity_exists(
    media_server, tmp_path
):
    """The state lives in the user's download folder. It must not hold a signature."""
    plan = MediaPlan(script=[{"truncate_after": 1024}])
    core = make_core(request_attempts=1)
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/media.mp4?{SIGNED_QUERY}"
        with pytest.raises(Exception):
            await core.download(
                http_config(tmp_path, url, media_source=source(url, IDENTITY))
            )

        written = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert written["identity"] == IDENTITY
    assert "SIG-SENTINEL-VALUE" not in json.dumps(written)
    assert "2001-db8-SENTINEL" not in json.dumps(written)
    # Still enough left to tell which resource a stale file belongs to.
    assert "/media.mp4" in written["url"]


@pytest.mark.asyncio
async def test_without_an_identity_the_state_keeps_the_url_it_always_kept(
    media_server, tmp_path
):
    """Backwards compatibility: the URL is the comparison, so it stays whole."""
    plan = MediaPlan(script=[{"truncate_after": 1024}])
    core = make_core(request_attempts=1)
    with media_server(plan) as (base_url, _):
        url = f"{base_url}/media.mp4?v=7"
        with pytest.raises(Exception):
            await core.download(
                http_config(tmp_path, url, media_source=source(url, None))
            )

        written = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert written["url"] == url
    assert written["identity"] is None
