from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MediaSource:
    url: str
    source_type: str  # e.g., "HLS", "DASH", "HTTP"
    quality: Optional[str] = None

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
