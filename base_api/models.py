from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional


@dataclass
class MediaTrackInfo:
    """What one media source technically *contains*, as the provider stated it.

    Deliberately separate from `MediaSource`: that one is a transport contract -
    where the bytes are, what headers they need, how large they are - while this
    describes the media inside them. The two have different lifetimes and
    different readers, and a provider that states nothing technical produces one
    empty object here rather than a dozen `None`s scattered across the transport.

    Every field is optional and every one means the same thing when unset:
    **the provider stated nothing**. Never "zero", never "false". A `False` in
    `is_default_audio` is a provider saying "this is not the default track"; a
    `None` is a provider that has no concept of default tracks at all, and a
    caller choosing between renditions must be able to tell those apart.

    Nothing here is derived, normalised or guessed. Codec strings are kept
    verbatim - `"avc1.640028"`, not `"avc1"` and certainly not `"h264"` - for
    the same reason `MediaSource.quality_label` is: the moment a library starts
    interpreting a provider's vocabulary it owns a taxonomy that has to track
    every provider forever. Mapping a codec string to a family is a decision for
    whoever is choosing between tracks, and it belongs there.
    """

    #: What the source carries: "combined", "video" or "audio". The one field a
    #: caller needs before it can pair a silent video track with a picture-less
    #: audio one; everything else only refines that choice.
    role: Optional[str] = None
    #: The container the bytes are in - "mp4", "webm", "mkv". The provider's own
    #: word for it, or the file extension when that is all the provider gives.
    container: Optional[str] = None
    #: Verbatim codec strings. See the note above on why they stay unparsed.
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    #: Frames per second. A float because providers state fractional rates -
    #: 29.97 and 23.976 are ordinary values, and rounding them to int at the
    #: model boundary would lose the distinction from 30 and 24.
    fps: Optional[float] = None
    #: Bits per second, normalised to that unit whatever the provider used.
    #: Bits rather than the kbit/s some APIs report, so a reader never has to
    #: guess which of the two a number is.
    bitrate_bps: Optional[int] = None
    #: Pixel dimensions, for display and diagnostics. **Never for ranking.**
    #: A portrait video is 1080x1920 and its provider still calls it "1080p", so
    #: a caller that sorts by height ranks it above a genuine 1440p. The tier a
    #: user asked for lives in `MediaSource.quality_value`.
    width: Optional[int] = None
    height: Optional[int] = None
    #: BCP-47-ish tag as the provider wrote it, e.g. "en-US". Not parsed.
    language: Optional[str] = None
    #: Whether this is the original or provider-default audio track. On a video
    #: with dubbed renditions this is the only field that tells the intended
    #: track from nine machine translations of it.
    is_default_audio: Optional[bool] = None
    #: Whether this is a Dynamic Range Compression (DRC) rendition - a
    #: loudness-normalised variant of the same audio, published alongside the
    #: original. Named for the audio process it is, not for a reduction:
    #: "dynamic range" also names the video property HDR/SDR describes, and a
    #: field called `is_reduced_dynamic_range` sitting next to a future
    #: `dynamic_range` would read as "not HDR" to anyone skimming. Providers
    #: mark it variously - a `-drc` suffix on a format id, an `isDrc` flag - and
    #: normalising those to this one boolean is the adapter's job.
    is_dynamic_range_compressed: Optional[bool] = None
    #: Duration of this track, when the provider states one *per track*. Most do
    #: not: they state a duration for the whole video, which is not the same
    #: thing and is not what a caller checking two tracks for drift needs.
    #: Advisory only - the honest measurement comes from the downloaded
    #: containers, after both tracks are on disk.
    duration_ms: Optional[int] = None


@dataclass
class MediaSource:
    url: str
    source_type: str  # e.g., "HLS", "DASH", "HTTP"
    quality: Optional[str] = None
    #: HTTP request headers the media requests for this source must carry, e.g.
    #: a Referer required by hotlink protection. Transport metadata only - never
    #: cookies, tokens or other credentials. The download engine applies them
    #: per request; they never mutate any session.
    headers: Dict[str, str] = field(default_factory=dict)
    #: Total size in bytes the provider states for this source, when it states
    #: one. It is provider metadata, not a measurement: the transport still
    #: takes the wire's own Content-Length / Content-Range as authoritative for
    #: the body it is receiving. A progressive download uses it to size the
    #: progress bar before the first response header arrives, and as the
    #: completeness bound when the server states no length of its own.
    expected_size: Optional[int] = None
    #: The provider's own numeric quality tier for this source - PeerTube's
    #: `resolution.id`, an HLS variant height, whatever the provider ranks by.
    #: Comparable across the sources of one media, meaningless across providers.
    #: Ordering ("best"/"worst"/"half") is defined on this value, never on the
    #: label below and never on a dimension re-derived from the URL.
    quality_value: Optional[int] = None
    #: The provider's original quality label, verbatim - "1080p", "720p60",
    #: "Original". Never parsed into a number: a PeerTube portrait video is
    #: `resolution.id = 1920` with `label = "1080p"`, so the label and the
    #: numeric tier deliberately disagree and both are kept as they arrived.
    quality_label: Optional[str] = None
    #: What this source technically contains. Always present, never `None`, so a
    #: caller reads `source.track.video_codec` without a guard - an empty
    #: `MediaTrackInfo` says "the provider stated nothing" just as clearly as a
    #: missing object would, and says it without a branch at every use.
    track: MediaTrackInfo = field(default_factory=MediaTrackInfo)
    #: A stable, provider-chosen name for *the thing these bytes are*, opaque to
    #: this library. Two sources carrying the same identity are the same track,
    #: even when their URLs differ.
    #:
    #: It exists because a signed CDN URL is not an identity: it carries an
    #: expiry, and re-resolving the same track yields a different URL for the
    #: same bytes. Keyed on the URL, a resume is discarded every time - and the
    #: signed URL, with whatever its signature covers, has to be persisted to
    #: compare against. Keyed on an identity, neither is true.
    #:
    #: The engine never parses it and never invents one. A provider with no
    #: stable name for a track leaves it `None`, and the URL goes on being the
    #: identity exactly as before.
    identity: Optional[str] = None

    def __post_init__(self) -> None:
        # Own copies: two sources built from one caller's objects must not alias
        # each other, and a caller mutating its dict or its track afterwards
        # must not silently rewrite the contract of an already-created source.
        self.headers = dict(self.headers)
        self.track = replace(self.track)

@dataclass
class HLSSegment:
    """One entry of a resolved HLS media playlist, in download order.

    Plain HLS segments carry only a URL. Byte-range playlists (EXT-X-BYTERANGE /
    fragmented MP4, where many logical segments live inside one resource) also
    carry the sub-range: `length` bytes starting at byte `offset`. The range is
    playlist-level information and deliberately not part of `MediaSource` -
    the source describes the transport contract, the segment describes which
    bytes of which resource one fragment is.
    """

    url: str
    length: Optional[int] = None
    offset: Optional[int] = None

    @property
    def has_range(self) -> bool:
        return self.length is not None

    @property
    def range_header(self) -> Optional[str]:
        """The HTTP Range for this segment: bytes=<offset>-<offset+length-1>."""
        if self.length is None:
            return None
        start = self.offset or 0
        return f"bytes={start}-{start + self.length - 1}"


@dataclass
class Media:
    provider: str
    original_url: str
    title: str
    provider_id: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    thumbnail: Optional[str] = None
    duration: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    sources: List[MediaSource] = field(default_factory=list)
