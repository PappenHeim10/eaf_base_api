from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    def __post_init__(self) -> None:
        # Own copy: two sources built from one caller dict must not alias each
        # other, and a caller mutating its dict afterwards must not silently
        # rewrite the transport contract of an already-created source.
        self.headers = dict(self.headers)

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
