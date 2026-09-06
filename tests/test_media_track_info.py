"""What `MediaTrackInfo` promises, and what it deliberately refuses to promise.

The value of this model is entirely in its discipline: it stores what a provider
said and nothing else. These tests pin that discipline, because the pressure to
break it - to default a missing field to something convenient, to normalise a
codec string on the way in - arrives one small commit at a time.
"""

from dataclasses import fields

import pytest

from base_api.models import MediaSource, MediaTrackInfo


TECHNICAL_FIELDS = tuple(f.name for f in fields(MediaTrackInfo))


def test_a_fresh_track_states_nothing():
    """Every field unset means "the provider said nothing" - not zero, not false."""
    track = MediaTrackInfo()
    for name in TECHNICAL_FIELDS:
        assert getattr(track, name) is None, f"{name} invented a default"


def test_absent_metadata_is_none_and_not_a_falsy_stand_in():
    """`None` and `False` are different statements and must stay different.

    `is_default_audio=False` is a provider saying "this is not the default
    track". `None` is a provider that has no concept of default tracks. A
    caller picking among renditions acts differently on the two, so a model that
    collapsed them would silently pick the wrong audio track.
    """
    stated = MediaTrackInfo(is_default_audio=False, is_dynamic_range_compressed=False)
    unstated = MediaTrackInfo()

    assert stated.is_default_audio is False
    assert unstated.is_default_audio is None
    assert stated.is_default_audio != unstated.is_default_audio

    assert unstated.bitrate_bps is None and unstated.bitrate_bps != 0
    assert unstated.container is None and unstated.container != ""
    assert unstated.fps is None and unstated.fps != 0.0


def test_codec_strings_are_stored_verbatim():
    """The model never normalises a codec to a family - that is a caller's job."""
    track = MediaTrackInfo(video_codec="avc1.640028", audio_codec="mp4a.40.2")

    assert track.video_codec == "avc1.640028"
    assert track.audio_codec == "mp4a.40.2"
    assert track.video_codec != "avc1"
    assert track.video_codec != "h264"


@pytest.mark.parametrize(
    "codec",
    ["avc1.640028", "vp09.00.40.08", "av01.0.08M.08", "mp4a.40.5", "opus"],
)
def test_no_codec_value_is_rewritten_on_the_way_in(codec):
    assert MediaTrackInfo(video_codec=codec).video_codec == codec


def test_a_source_always_has_a_track():
    """No caller should need a `None` guard before reading a technical field."""
    source = MediaSource(url="https://example.test/a.mp4", source_type="HTTP")

    assert source.track is not None
    assert isinstance(source.track, MediaTrackInfo)
    assert source.track.video_codec is None


def test_two_sources_do_not_alias_one_caller_track():
    """The guarantee `headers` already gives, extended to the track info.

    An adapter that builds several sources from one template object must not
    end up with four sources that share - and overwrite - one another's data.
    """
    template = MediaTrackInfo(role="video", fps=30.0)

    first = MediaSource(url="https://example.test/1.mp4", source_type="HTTP", track=template)
    second = MediaSource(url="https://example.test/2.mp4", source_type="HTTP", track=template)

    first.track.fps = 60.0

    assert second.track.fps == 30.0
    assert template.fps == 30.0


def test_a_caller_mutating_its_own_track_afterwards_changes_nothing():
    template = MediaTrackInfo(container="mp4")
    source = MediaSource(url="https://example.test/a.mp4", source_type="HTTP", track=template)

    template.container = "webm"

    assert source.track.container == "mp4"


def test_identity_defaults_to_none_so_the_url_stays_the_identity():
    """A provider with no stable track name changes nothing about resuming."""
    assert MediaSource(url="https://example.test/a.mp4", source_type="HTTP").identity is None


def test_identity_is_opaque_and_stored_as_given():
    source = MediaSource(
        url="https://example.test/a.mp4",
        source_type="HTTP",
        identity="provider:abc123:140",
    )
    assert source.identity == "provider:abc123:140"


def test_existing_sources_are_unaffected_by_the_new_fields():
    """Every field a caller relied on before still means what it meant."""
    source = MediaSource(
        url="https://example.test/a.mp4",
        source_type="HTTP",
        headers={"Referer": "https://example.test/"},
        expected_size=1234,
        quality_value=1080,
        quality_label="1080p",
    )

    assert source.url == "https://example.test/a.mp4"
    assert source.source_type == "HTTP"
    assert source.headers == {"Referer": "https://example.test/"}
    assert source.expected_size == 1234
    assert source.quality_value == 1080
    assert source.quality_label == "1080p"
