from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable
from enum import StrEnum
import random

# Download Report is the report the function returns
@dataclass
class DownloadState:
    version: int
    created_at: Any
    updated_at: Any
    m3u8_url: str | None
    quality: str | int
    output_path: Path | str
    segment_dir: Path | str | None
    segment_index_width: int
    start_segment: int
    total: int
    missing: list[int]
    # Version 1: plain URL strings. Version 2 (byte-range playlists): dicts of
    # url plus, for ranged entries, length and offset.
    segments: list[Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


#: The `kind` every progressive-HTTP resume state carries. The HLS state and
#: this one share the `.state/` directory, so the discriminator has to be in
#: the payload: a reader that does not find exactly this value must treat the
#: file as somebody else's and start fresh rather than misread its fields.
PROGRESSIVE_STATE_KIND = "http-progressive"

#: Schema version of `ProgressiveDownloadState`. An unknown version is
#: discarded, never guessed at - a resume is only safe when every field means
#: what this engine thinks it means.
PROGRESSIVE_STATE_VERSION = 1


@dataclass
class ProgressiveDownloadState:
    """Resume state of one progressive HTTP download.

    `downloaded_bytes` is what the state *believes*; the temporary file's real
    size is the local truth and wins whenever the two disagree. The validators
    are kept so the next run can decide for itself whether the remote resource
    is still the one those bytes came from - the transport never relies on the
    server evaluating `If-Range` correctly.

    `identity` and `url` together answer "are these bytes still the resource we
    are fetching now?". Whichever of the two is present decides; `identity`
    wins when a provider supplied one. See the field notes below for why a URL
    alone cannot answer that question for a signed source.
    """

    version: int
    kind: str
    created_at: Any
    updated_at: Any
    #: The source URL when nothing else identifies the resource, and a redacted
    #: form of it - host and path, no query values - when `identity` does. It is
    #: a diagnostic either way: enough to see which resource a stale state file
    #: belongs to, never enough to leak a signature into the user's download
    #: folder.
    url: str
    #: The provider's stable name for this track, when it has one. Present here
    #: so a resume survives the URL changing underneath it: a signed URL expires
    #: within hours and re-resolving the same track yields a different one, so a
    #: state keyed on the URL discards a perfectly good partial file every time.
    #: Absent for every provider that supplies none, in which case `url` is the
    #: identity exactly as it always was.
    identity: str | None
    output_path: Path | str
    temp_path: Path | str
    total_size: int | None
    downloaded_bytes: int
    etag: str | None
    #: Whether `etag` arrived as a `W/`-prefixed weak validator. A weak ETag is
    #: never put into `If-Range`; it is only ever compared by this engine.
    etag_weak: bool
    last_modified: str | None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# Download state is used for the literal file that tracks it
@dataclass
class DownloadReport:
    status: str
    total: int
    downloaded: int
    missing: list[int]
    missing_urls: list[str]
    segment_dir: Path | str | None
    segment_state_path: Path | str | None
    start_segment: int
    quality: str | int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ResultOrder(StrEnum):
    """Controls when ``Helper`` exposes completed item results."""

    COMPLETION = "completion"
    ORIGINAL = "original"


class ErrorMode(StrEnum):
    """Terminal action after retries are exhausted."""

    RAISE = "raise"
    YIELD = "yield"
    SKIP = "skip"


class ErrorAction(StrEnum):
    """Decision optionally returned by a user-provided error handler."""

    RETRY = "retry"
    RAISE = "raise"
    YIELD = "yield"
    SKIP = "skip"


class ScrapeStage(StrEnum):
    """Identifies whether a yielded failure belongs to a page or an item."""

    PAGE = "page"
    ITEM = "item"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    Bounded exponential retry configuration for one Helper stage.

    ``max_attempts`` includes the first call.  The default performs one attempt,
    which avoids duplicating retries already performed by ``BaseCore.fetch``.
    ``jitter`` adds a uniformly random number of seconds to each retry delay.
    """

    max_attempts: int = 1
    base_delay: float = 0.0
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.0
    retry_for: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("RetryPolicy delays and jitter cannot be negative")
        if self.multiplier < 1:
            raise ValueError("RetryPolicy.multiplier must be at least 1")
        if not self.retry_for or not all(
            isinstance(item, type) and issubclass(item, Exception)
            for item in self.retry_for
        ):
            raise TypeError("RetryPolicy.retry_for must contain Exception classes")

    def permits(self, error: Exception) -> bool:
        """Return whether this exception type is eligible for automatic retry."""
        return isinstance(error, self.retry_for)

    def delay_after(self, attempt: int) -> float:
        """Return the delay after the numbered failed attempt."""
        exponential = self.base_delay * (self.multiplier ** max(attempt - 1, 0))
        return min(exponential, self.max_delay) + random.uniform(0.0, self.jitter)


@dataclass(frozen=True, slots=True)
class ScrapeErrorContext:
    """Complete context passed to a page or item error handler."""

    stage: ScrapeStage
    url: str
    error: Exception
    attempt: int
    max_attempts: int
    page_index: int
    item_index: int | None


type ErrorHandler = Callable[
    [ScrapeErrorContext], ErrorAction | Awaitable[ErrorAction]
]
