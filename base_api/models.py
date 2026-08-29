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
