from __future__ import annotations
import re
import os
import time
import random
import hashlib
import string
import shutil
import asyncio
import inspect
import logging
import traceback
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from urllib.parse import urljoin, urlsplit
from dataclasses import MISSING, dataclass, field, fields
from curl_cffi import CurlOpt # Used for DNS over HTTPS
from curl_cffi.requests.errors import RequestsError
from curl_cffi.requests import AsyncSession, Response
from cachetools import TTLCache
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter, retry_if_exception, RetryError
from typing import (
    Union, Callable, Tuple, AsyncGenerator, ClassVar, Generic, TypeVar,
    cast, List, Dict, Any, Awaitable, Self, TYPE_CHECKING, Protocol,
)


# 1. Standardize on relative imports
from base_api.modules.errors import *
from base_api.modules.type_hints import (
    DownloadReport, ResultOrder, ErrorMode, ErrorAction, ScrapeStage, 
    RetryPolicy, ScrapeErrorContext, ErrorHandler
)
from base_api.modules.static_functions import (
    load_segment_state, parse_retry_after, log_precondition_failed,
    write_segment_state, build_segment_state, segment_file_path,
    parse_challenge, other_challenge, least_factors, available_qualities,
    choose_variant, collect_variants, get_segment_index_width,
    build_progressive_state, format_url_for_log, load_progressive_state, normalize_etag,
    parse_content_range, parse_unsatisfied_content_range, write_progressive_state,
)
from base_api.modules.config import (
    config, RuntimeConfig, DownloadConfigHLS, DownloadConfigHTTP,
    DownloadConfigRAW, IteratorConfig,
)
from base_api.models import HLSSegment
from base_api.modules.progress_bars import Callback
from base_api.modules.logger import configure_app_logging

# 2. Handle optional dependencies cleanly
try:
    import m3u8
except ImportError:
    m3u8 = None

# 3. Handle specific runtime imports
if TYPE_CHECKING:
    from av.audio.codeccontext import AudioCodecContext
    import m3u8

# The following imports are optional, because they depend on per API and I want to be as memory efficient as possible

try:
    import m3u8
    # Needed for all videos that use HLS streaming. Some do not and use mp4 containers / files instead
except (ModuleNotFoundError, ImportError):
    m3u8 = None  # type: ignore


REGEX_CHALLENGE = re.compile(r'var p=(\d+); var s=(\d+);.*?(\d+):1;', re.DOTALL)


class CachePolicy(StrEnum):
    """Control whether a text request reads from or writes to the cache."""

    USE = "use"
    BYPASS = "bypass"
    REFRESH = "refresh"


def _contains_http_status(error: BaseException, status_code: int) -> bool:
    """Return whether an exception or one of its wrappers has an HTTP status."""
    pending: deque[BaseException] = deque((error,))
    seen: set[int] = set()
    while pending:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if (
            isinstance(current, HTTPStatusError)
            and current.status_code == status_code
        ):
            return True

        for attribute in ("original_error", "last_error", "__cause__", "__context__"):
            nested = getattr(current, attribute, None)
            if isinstance(nested, BaseException):
                pending.append(nested)

        nested_errors = getattr(current, "errors", ())
        if isinstance(nested_errors, (tuple, list)):
            pending.extend(
                nested for nested in nested_errors if isinstance(nested, BaseException)
            )
    return False


def _normalize_packet_timestamps(
    packet: Any,
    offsets: dict[int, int],
    last_dts: dict[int, int],
    last_durations: dict[int, int],
) -> int:
    """Keep remuxed packet timestamps continuous across HLS discontinuities."""
    if packet.dts is None:
        return 0

    stream_index = packet.stream.index
    raw_dts = packet.dts
    offset = offsets.get(stream_index, 0)
    normalized_dts = raw_dts + offset
    previous_dts = last_dts.get(stream_index)
    correction = 0

    if previous_dts is not None:
        step = max(1, last_durations.get(stream_index, packet.duration or 1))
        expected_dts = previous_dts + step
        try:
            ten_seconds = max(1, int(10 / float(packet.time_base)))
        except (TypeError, ValueError, ZeroDivisionError):
            ten_seconds = step * 240

        discontinuity_threshold = max(step * 10, ten_seconds)
        delta = normalized_dts - expected_dts
        if normalized_dts <= previous_dts or abs(delta) > discontinuity_threshold:
            correction = -delta
            offset += correction
            offsets[stream_index] = offset
            normalized_dts = raw_dts + offset

    packet.dts = normalized_dts
    if packet.pts is not None:
        packet.pts += offset
    last_dts[stream_index] = normalized_dts
    last_durations[stream_index] = max(1, packet.duration or 1)
    return correction


@dataclass(frozen=True, slots=True)
class RequestCacheKey:
    """Identity of a cacheable HTTP request without retaining credentials."""

    method: str
    url: str
    allow_redirects: bool
    params_fingerprint: str
    body_fingerprint: str
    headers_fingerprint: str
    cookies_fingerprint: str


@dataclass(frozen=True, slots=True)
class SegmentCacheKey:
    master_url: str
    quality: str


class CacheBackend(Protocol):
    """Storage contract consumed by :class:`BaseCore`."""

    def get_response(self, key: RequestCacheKey) -> str | None: ...
    def set_response(self, key: RequestCacheKey, content: str) -> None: ...
    def delete_response(self, key: RequestCacheKey) -> None: ...
    def invalidate_url(self, url: str) -> None: ...
    def get_segments(self, key: SegmentCacheKey) -> list[str] | None: ...
    def set_segments(self, key: SegmentCacheKey, segments: Sequence[str]) -> None: ...


def _text_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _segments_size(value: tuple[str, ...]) -> int:
    return sum(len(segment.encode("utf-8")) for segment in value)


def _freeze_cache_value(value: Any) -> Any:
    """Convert common request values into a deterministic, hashable structure."""
    if isinstance(value, Mapping):
        items = (
            (_freeze_cache_value(key), _freeze_cache_value(item))
            for key, item in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_cache_value(item) for item in value), key=repr))
    if isinstance(value, bytearray):
        return bytes(value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    return repr(value)


def _cache_fingerprint(value: Any) -> str:
    frozen = _freeze_cache_value(value)
    return hashlib.sha256(repr(frozen).encode("utf-8")).hexdigest()


class Cache(CacheBackend):
    """Thread-safe, bounded TTL caches for text responses and HLS segments."""

    def __init__(self, configuration: "RuntimeConfig") -> None:
        self._responses: TTLCache[RequestCacheKey, str] = TTLCache(
            maxsize=max(1, configuration.response_cache_size_bytes),
            ttl=configuration.response_cache_ttl,
            getsizeof=_text_size,
        )
        self._segments: TTLCache[SegmentCacheKey, tuple[str, ...]] = TTLCache(
            maxsize=max(1, configuration.segment_cache_size_bytes),
            ttl=configuration.segment_cache_ttl,
            getsizeof=_segments_size,
        )
        self._responses_enabled = configuration.response_cache_size_bytes > 0
        self._segments_enabled = configuration.segment_cache_size_bytes > 0
        self.lock = threading.RLock()

    def get_response(self, key: RequestCacheKey) -> str | None:
        if not self._responses_enabled:
            return None
        with self.lock:
            return self._responses.get(key)

    def set_response(self, key: RequestCacheKey, content: str) -> None:
        if not self._responses_enabled or _text_size(content) > self._responses.maxsize:
            return
        with self.lock:
            self._responses[key] = content

    def delete_response(self, key: RequestCacheKey) -> None:
        with self.lock:
            self._responses.pop(key, None)

    def invalidate_url(self, url: str) -> None:
        with self.lock:
            for key in tuple(self._responses):
                if key.url == url:
                    self._responses.pop(key, None)

    def get_segments(self, key: SegmentCacheKey) -> list[str] | None:
        if not self._segments_enabled:
            return None
        with self.lock:
            segments = self._segments.get(key)
            return list(segments) if segments is not None else None

    def set_segments(self, key: SegmentCacheKey, segments: Sequence[str]) -> None:
        frozen_segments = tuple(segments)
        if (
            not self._segments_enabled
            or _segments_size(frozen_segments) > self._segments.maxsize
        ):
            return
        with self.lock:
            self._segments[key] = frozen_segments

    def clear(self) -> None:
        with self.lock:
            self._responses.clear()
            self._segments.clear()


_MEDIA_SOURCES_KEY = "eaf_base_api.load_sources"


class _UnloadedValue:
    """Private marker that distinguishes an unresolved field from a real ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNLOADED>"

    def __copy__(self) -> _UnloadedValue:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _UnloadedValue:
        return self

    def __reduce__(self) -> tuple[Callable[[], _UnloadedValue], tuple[()]]:
        return (_unloaded_value, ())


_UNLOADED = _UnloadedValue()


def _unloaded_value() -> _UnloadedValue:
    """Restore the process-wide sentinel while unpickling or deep-copying."""
    return _UNLOADED


def media_field(
    *sources: str, # Tuple that defines the sources in which the field can appear e.g., (html, api)
    default: Any = MISSING, # The default state for the field is missing cuz yeah it hasn't been loaded yet
    default_factory: Callable[[], Any] | Any = MISSING,
    repr: bool = False,
    compare: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """
    Declare a dataclass field that can be populated by one or more media loaders.

    Source order is significant: the first source has the highest precedence when
    several successfully loaded sources contain the same field.  If neither a
    ``default`` nor a ``default_factory`` is supplied, the field starts with an
    internal *unloaded* sentinel.  Consequently, a loader can return ``None`` and
    direct attribute access will correctly treat that value as loaded.

    ``repr`` and ``compare`` default to false because generated dataclass methods
    should not accidentally access an unresolved field.  Use ``BaseMedia.to_dict``
    when serialising a partially loaded model.

    Example::

        title: str | None = media_field("api", "html")
        stream_url: str | None = media_field("html")
    """
    if not sources:
        raise ValueError("media_field() requires at least one loader source")

    normalized_sources: list[str] = [] # Converts it into a list
    for source in sources:
        if not isinstance(source, str) or not source or not source.isidentifier(): # just checks if source is a valid string
            raise ValueError(
                "media loader sources must be non-empty Python identifiers; "
                f"received {source!r}"
            )
        if source in normalized_sources: # You shouldn't provide ("html", "html") because why?
            raise ValueError(f"media field source {source!r} was declared twice")
        normalized_sources.append(source)

    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("media_field() cannot receive both default and default_factory")

    field_metadata = dict(metadata or {})
    if _MEDIA_SOURCES_KEY in field_metadata: # You can't put the key for the field metadata as a field metadata because this interferes with the logic here, this should never happen though
        raise ValueError(f"metadata key {_MEDIA_SOURCES_KEY!r} is reserved")
    field_metadata[_MEDIA_SOURCES_KEY] = tuple(normalized_sources) # Creates the field metadata with the source key + tuple of normalized sources e.g., (html, api)

    field_arguments: dict[str, Any] = {
        "repr": repr,
        "compare": compare,
        "metadata": field_metadata,
    }

    # Repr and compare (__repr__ / __eq__) are False by default because if the element is not yet loaded
    # and you try to log it, this would cause an error
    # metadata holds the prepared dictionary of field / sources along with the media source key

    if default_factory is not MISSING:
        field_arguments["default_factory"] = default_factory
        # If we just give [] or {} as a default value in an object this will raise an issue cuz Python
        # for security reasons doesn't allow this because the memory would be shared between the different dataclass
        # instances. That's why we give a list, dict function which then creates the actual [] or {} when the dataclass
        # is created (yeah I know this seems weird, but needed)
    elif default is not MISSING:
        field_arguments["default"] = default
        # If a default value was provided the field will automatically get this value and it won't be tried to load
    else:
        field_arguments["default"] = _UNLOADED
        # We pick _UNLOADED here because the problem is that Python defaults to "None" for stuff that is
        # really not Available or is just Empty. The problem is, that I can't distinguish then if something
        # just isn't available because the HTML didn't expose it for example or if it's just empty.
        # With _UNLOADED we can definitely know for sure that the element is NOT yet available and NEEDS to be loaded

    return field(**field_arguments) # Creates the dataclass with the keyword values


class LoadState(StrEnum):
    """Observable lifecycle of one named ``BaseMedia`` source."""
    NOT_LOADED = "not_loaded" # (html or api or whatever was not yet loaded(
    LOADING = "loading" # It is currently loading e.g., being fetched
    LOADED = "loaded" # It finished loading (this is good), don't need to fetch it again :)
    FAILED = "failed" # It failed loading :(


@dataclass(frozen=True, slots=True)
class _MediaSchema:
    """Cached, validated view of a media dataclass's loadable fields."""
    field_names: frozenset[str] # Set of field names e.g., ('title', 'description')
    field_sources: dict[str, tuple[str, ...]] # Source map e.g.,: {'title': ('api', 'html')
    source_fields: dict[str, frozenset[str]] # like field_sources but in reverse
    source_order: tuple[str, ...] # Source order, usually ('api', 'html') cuz API is faster to load


_MEDIA_SCHEMA_CACHE: dict[type[Any], _MediaSchema] = {}
# Preserves a Cache of MediaScheme, because looking this up each time without would take time, this makes it faster


def _media_schema(model_type: type[Any]) -> _MediaSchema:
    """Build source/field indexes once per concrete dataclass type."""
    cached = _MEDIA_SCHEMA_CACHE.get(model_type)
    if cached is not None:
        return cached # Loads from cache

    # Model Type is needed to tell the base layout of the class.
    # If we do this per instance this would run as often as the instance appears which is unnecessary
    # Because why would I need to build the media scheme 5000 times when I can just do it once, tell the future
    # references which type it is and fetch it from cache, because guess what I won't randomly change
    # the scheme mid runtime xD
    dataclass_fields = tuple(fields(model_type)) # Variable annotations + metadata
    field_sources: dict[str, tuple[str, ...]] = {} # Creates a mutable dict out of the field sources
    source_fields_mutable: dict[str, set[str]] = {} # Creates a mutable dictionary out of the source fields
    source_order: list[str] = [] # See above
    # The fields need to be mutable here, because each new defined object in my classes need to be added
    # to the dictionary. So this is only temporary and down below this is going back to a frozenset

    for dataclass_field in dataclass_fields:
        # Get the tuple in the _MEDIA_SOURCES_KEY
        sources = dataclass_field.metadata.get(_MEDIA_SOURCES_KEY)
        # sources = e.g., ("api", "html")
        if not sources:
            continue # Regular dataclass fields are ignored e.g., 'url' and 'core' for most classes

        field_sources[dataclass_field.name] = tuple(sources)
        # Add e.g, title to field_sources, becomes {"title": ("api", "html")}
        for source in sources:
            # .setdefault("api", set()) Does e.g., 'api' exist in this dictionary. If not
            # Create it and return an empty set and add the dataclass field name
            # So it becomes {"api": {"title"}}
            source_fields_mutable.setdefault(source, set()).add(dataclass_field.name)
            if source not in source_order:
                source_order.append(source)

    schema = _MediaSchema(
        field_names=frozenset(item.name for item in dataclass_fields), # {"title", "duration", "raw_html"
        field_sources=field_sources, # {"title": ("api", "html")
        source_fields={
            source: frozenset(field_names) # "api": frozenset({"title", "duration"})
            for source, field_names in source_fields_mutable.items()
        },
        source_order=tuple(source_order), # ("api", "html")
    )
    _MEDIA_SCHEMA_CACHE[model_type] = schema # Loads it into the cache
    return schema # Return the final scheme


@dataclass(slots=True, kw_only=True, repr=False)
class BaseMedia:
    """
    Base class for dataclass models whose fields are loaded from remote sources.

    Subclasses declare loadable attributes with :func:`media_field` and map each
    source name to an async method through ``loader_methods``.  A loader returns a
    mapping; it never mutates the model itself.  ``BaseMedia`` validates the full
    mapping and commits it atomically, so a failed parser cannot leave half-loaded
    model state behind.

    Different callers requesting the same source share one task. Cancelling a
    waiter cancels that shared operation so network work never escapes its caller;
    every waiter then observes cancellation and the source becomes retryable.
    Loading different sources may happen concurrently, but field precedence stays
    deterministic because it follows the order declared by ``media_field``.
    """

    url: str
    core: object

    loader_methods: ClassVar[Mapping[str, str]] = {}

    _load_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )
    _source_states: dict[str, LoadState] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_results: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_errors: dict[str, BaseException] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_tasks: dict[str, asyncio.Task[None]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    if not TYPE_CHECKING:
        def __getattribute__(self, name: str) -> Any:
            """
            Reject only the private unloaded sentinel, never a legitimate ``None``.

            Attribute access remains synchronous and therefore cannot initiate network
            I/O.  The exception tells callers exactly which field and sources to pass
            to ``load_fields`` or ``load_sources``.
            """
            value = object.__getattribute__(self, name)
            if value is not _UNLOADED:
                return value

            # Because we override the __getattribute__ method, we need to call object.__getattribute__.
            # If I'd use self.<something> we would get a recursion error as the function would call itself infinitely

            model_type = type(self)
            """
            Explanation why model_type exists:
            So, basically when we build the schema (down below) this takes some time. However, this function actually
            caches the fully resolved dataclass. So if we are going over a thousand Video objects usually we would need
            to reconstruct the thousand video objects, well, a thousand times.
            
            By giving the model_type with self, we can tell the cache: Yo, this is a Video object. See if you have
            already processed this and if yes it will use this instead of processing it each time again.
            Might only save a few milliseconds but I ain't Ubisoft XDDD
            """
            schema = _media_schema(model_type)
            sources = schema.field_sources.get(name, ()) # E.g., ("api", "html") for name=title (PornHub API as an example)
            url = object.__getattribute__(self, "url") # The actual URL to fetch e.g., https://example.com/video/id?=
            all_source_errors = object.__getattribute__(self, "_source_errors")
            relevant_errors = {
                source: all_source_errors[source]
                for source in sources
                if source in all_source_errors
            } # Basically just checks which sources had an error e.g., if html failed fetching this will return html
            raise DataNotLoadedError(
                model_type.__name__, name, url, sources, relevant_errors
                # Raises a custom exception that tells you which attribute failed with its associated source and the error
                # that happened along with the actual URL (so that I can replicate it when you report it)
            )

    def __repr__(self) -> str:
        """Represent identity and load state without touching unresolved fields."""
        loaded = ", ".join(sorted(self.loaded_sources)) or "none"
        return f"{type(self).__name__}(url={self.url!r}, loaded_sources={loaded})"

    @property
    def loaded_sources(self) -> frozenset[str]:
        """Return an immutable snapshot of sources that loaded successfully."""
        states = object.__getattribute__(self, "_source_states")
        return frozenset(
            source for source, state in states.items() if state is LoadState.LOADED
        ) # Returns the sources that have been successfully fetched

    @property
    def source_errors(self) -> Mapping[str, BaseException]:
        """Return a copy of the most recent failure for each source."""
        return dict(object.__getattribute__(self, "_source_errors"))

    def source_state(self, source: str) -> LoadState:
        """Inspect one declared source without starting a load."""
        schema = _media_schema(type(self))
        if source not in schema.source_fields:
            raise LoaderConfigurationError(
                f"{type(self).__name__} has no media fields assigned to source {source!r}"
            )
        return object.__getattribute__(self, "_source_states").get(
            source, LoadState.NOT_LOADED
        ) # Basically tells you the state of the source

    def is_field_loaded(self, field_name: str) -> bool:
        """Return whether a field contains a real value, including real ``None``."""
        self._validate_field_name(field_name)
        return object.__getattribute__(self, field_name) is not _UNLOADED
        # Basically checks if a field has been loaded

    def unloaded_fields(self) -> frozenset[str]:
        """Return all declared media fields that still contain the sentinel."""
        schema = _media_schema(type(self))
        return frozenset(
            field_name
            for field_name in schema.field_sources
            if object.__getattribute__(self, field_name) is _UNLOADED
        ) # Returns a set with all fields that have not yet been loaded

    def to_dict(
        self,
        *,
        include_unloaded: bool = False,
        include_core: bool = False,
    ) -> dict[str, Any]:
        """
        Serialise public dataclass fields without triggering lazy-field errors.

        Unresolved fields are omitted by default.  When ``include_unloaded`` is
        true they are represented as ``None``; this keeps the private sentinel out
        of application data.  ``core`` is excluded by default because clients and
        sessions are normally not serialisable.
        """
        result: dict[str, Any] = {}
        for dataclass_field in fields(type(self)):
            name = dataclass_field.name
            if name.startswith("_") or (name == "core" and not include_core):
                continue
            value = object.__getattribute__(self, name)
            if value is _UNLOADED:
                if include_unloaded:
                    result[name] = None
                continue
            result[name] = value
        return result

    async def load_sources(
        self,
        *sources: str,
        retry_failed: bool = True,
    ) -> Self:
        """
        Load named sources concurrently and return this model.

        Successful sources remain committed if a sibling source fails.  One
        failure is raised directly; several failures are wrapped in
        ``MediaLoadErrors`` rather than an ``ExceptionGroup``.
        """
        normalized_sources = tuple(dict.fromkeys(sources))
        if not normalized_sources:
            return self

        schema = _media_schema(type(self))
        for source in normalized_sources:
            if source not in schema.source_fields:
                raise LoaderConfigurationError(
                    f"{type(self).__name__} has no media fields assigned to "
                    f"source {source!r}"
                )
            self._loader_for_source(source)

        results = await asyncio.gather(
            *(
                self._ensure_source(source, retry_failed=retry_failed)
                for source in normalized_sources
            ),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                failures.append(result)

        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise MediaLoadErrors(tuple(failures))
        return self

    async def load_fields(
        self,
        *field_names: str,
        retry_failed: bool = True,
    ) -> Self:
        """
        Load the smallest useful set of sources for the requested fields.

        The source selection is a deterministic greedy cover.  For example, if
        ``html`` can populate both requested fields while ``api`` can populate
        only one, only ``html`` is loaded.  Ties follow the precedence order in
        the field declarations.
        """
        requested = tuple(dict.fromkeys(field_names))
        if not requested:
            return self

        schema = _media_schema(type(self))
        pending: set[str] = set()
        for field_name in requested:
            self._validate_field_name(field_name)
            if object.__getattribute__(self, field_name) is not _UNLOADED:
                continue
            if field_name not in schema.field_sources:
                raise FieldNotLoadableError(type(self).__name__, field_name)
            pending.add(field_name)

        selected_sources: list[str] = []
        while pending:
            best_source: str | None = None
            best_coverage: set[str] = set()
            best_preference_cost: int | None = None
            for source in schema.source_order:
                coverage = pending.intersection(schema.source_fields[source])
                preference_cost = sum(
                    schema.field_sources[field_name].index(source)
                    for field_name in coverage
                )
                if (
                    len(coverage) > len(best_coverage)
                    or (
                        len(coverage) == len(best_coverage)
                        and coverage
                        and (
                            best_preference_cost is None
                            or preference_cost < best_preference_cost
                        )
                    )
                ):
                    best_source = source
                    best_coverage = coverage
                    best_preference_cost = preference_cost

            if best_source is None:
                # Schema construction guarantees this cannot happen unless model
                # metadata was modified at runtime after it had been cached.
                unresolved = sorted(pending)
                raise LoaderConfigurationError(
                    f"No loader source can resolve fields {unresolved!r} on "
                    f"{type(self).__name__}"
                )
            selected_sources.append(best_source)
            pending.difference_update(best_coverage)

        return await self.load_sources(
            *selected_sources, retry_failed=retry_failed
        )

    async def get_field(self, field_name: str, *, retry_failed: bool = True) -> Any:
        """Load one field if necessary and return its value."""
        self._validate_field_name(field_name)
        if object.__getattribute__(self, field_name) is _UNLOADED:
            await self.load_fields(field_name, retry_failed=retry_failed)
        return object.__getattribute__(self, field_name)

    def _validate_field_name(self, field_name: str) -> None:
        schema = _media_schema(type(self))
        if field_name not in schema.field_names:
            raise UnknownMediaFieldError(type(self).__name__, field_name)

    def _loader_for_source(self, source: str) -> Callable[[], Awaitable[Mapping[str, Any]]]:
        method_name = type(self).loader_methods.get(source)
        if method_name is None:
            raise LoaderConfigurationError(
                f"{type(self).__name__}.loader_methods does not map source {source!r}"
            )
        try:
            loader = object.__getattribute__(self, method_name)
        except AttributeError as error:
            raise LoaderConfigurationError(
                f"{type(self).__name__}.loader_methods maps {source!r} to missing "
                f"method {method_name!r}"
            ) from error
        if not callable(loader):
            raise LoaderConfigurationError(
                f"{type(self).__name__}.{method_name} is not callable"
            )
        return cast(Callable[[], Awaitable[Mapping[str, Any]]], loader)

    async def _ensure_source(self, source: str, *, retry_failed: bool) -> None:
        """Return after one shared source task succeeds, or re-raise its error."""
        lock = object.__getattribute__(self, "_load_lock")
        async with lock:
            states = object.__getattribute__(self, "_source_states")
            tasks = object.__getattribute__(self, "_source_tasks")
            errors = object.__getattribute__(self, "_source_errors")
            state = states.get(source, LoadState.NOT_LOADED)

            if state is LoadState.LOADED:
                return
            if state is LoadState.FAILED and not retry_failed:
                raise errors[source]

            task = tasks.get(source)
            if task is None:
                states[source] = LoadState.LOADING
                task = asyncio.create_task(
                    self._execute_source_loader(source),
                    name=f"{type(self).__name__}:{source}:{self.url}",
                )
                tasks[source] = task

        # Direct awaiting intentionally propagates cancellation into the shared
        # source task. This keeps source I/O inside the lifetime of its callers.
        await task

    async def _execute_source_loader(self, source: str) -> None:
        """Execute, validate, and atomically commit one source loader."""
        model_name = type(self).__name__
        try:
            loader = self._loader_for_source(source)
            awaitable = loader()
            if not inspect.isawaitable(awaitable):
                raise LoaderContractError(
                    model_name,
                    source,
                    self.url,
                    "the configured loader must be async and return an awaitable",
                )
            raw_result = await awaitable
            result = self._validate_loader_result(source, raw_result)

            lock = object.__getattribute__(self, "_load_lock")
            async with lock:
                object.__getattribute__(self, "_source_results")[source] = result
                object.__getattribute__(self, "_source_states")[source] = LoadState.LOADED
                object.__getattribute__(self, "_source_errors").pop(source, None)
                self._apply_source_precedence(source)
                object.__getattribute__(self, "_source_tasks").pop(source, None)

        except asyncio.CancelledError:
            lock = object.__getattribute__(self, "_load_lock")
            async with lock:
                object.__getattribute__(self, "_source_states")[source] = LoadState.NOT_LOADED
                object.__getattribute__(self, "_source_tasks").pop(source, None)
            raise
        except Exception as error:
            recorded_error: BaseException
            if isinstance(error, (LoaderContractError, LoaderConfigurationError)):
                recorded_error = error
            else:
                recorded_error = MediaLoadError(
                    model_name, source, self.url, error
                )

            lock = object.__getattribute__(self, "_load_lock")
            async with lock:
                object.__getattribute__(self, "_source_states")[source] = LoadState.FAILED
                object.__getattribute__(self, "_source_errors")[source] = recorded_error
                object.__getattribute__(self, "_source_tasks").pop(source, None)

            if recorded_error is error:
                raise
            raise recorded_error from error

    def _validate_loader_result(
        self, source: str, raw_result: Any
    ) -> dict[str, Any]:
        """Enforce the all-fields, no-surprises loader mapping contract."""
        model_name = type(self).__name__
        if not isinstance(raw_result, Mapping):
            raise LoaderContractError(
                model_name,
                source,
                self.url,
                f"expected a mapping, received {type(raw_result).__name__}",
            )

        result = dict(raw_result)
        if not all(isinstance(name, str) for name in result):
            raise LoaderContractError(
                model_name, source, self.url, "all result keys must be strings"
            )
        if any(value is _UNLOADED for value in result.values()):
            raise LoaderContractError(
                model_name,
                source,
                self.url,
                "a loader may not return BaseMedia's private unloaded sentinel",
            )

        expected_fields = _media_schema(type(self)).source_fields[source]
        actual_fields = set(result)
        missing = sorted(expected_fields.difference(actual_fields))
        unexpected = sorted(actual_fields.difference(expected_fields))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing fields {missing!r}; return None when absent")
            if unexpected:
                details.append(f"unexpected fields {unexpected!r}")
            raise LoaderContractError(
                model_name, source, self.url, "; ".join(details)
            )
        return result

    def _apply_source_precedence(self, completed_source: str) -> None:
        """
        Recompute affected fields from loaded source snapshots.

        This method is called while ``_load_lock`` is held.  Looking through each
        field's sources in declaration order makes the final value independent of
        network completion order.
        """
        schema = _media_schema(type(self))
        states = object.__getattribute__(self, "_source_states")
        results = object.__getattribute__(self, "_source_results")
        for field_name in schema.source_fields[completed_source]:
            for candidate_source in schema.field_sources[field_name]:
                if states.get(candidate_source) is LoadState.LOADED:
                    object.__setattr__(
                        self,
                        field_name,
                        results[candidate_source][field_name],
                    )
                    break





MediaT = TypeVar("MediaT", bound=BaseMedia)
OperationT = TypeVar("OperationT")


@dataclass(frozen=True, slots=True)
class ScrapeResult(Generic[MediaT]):
    """
    Immutable result yielded for an item success or a configured stage failure.

    A successful result has ``item`` and no ``error``.  A yielded failure has an
    ``error`` and no ``item``.  Page successes are internal and are not yielded.
    """

    stage: ScrapeStage
    url: str
    page_index: int
    item_index: int | None
    attempts: int
    item: MediaT | None = None
    error: ScrapeOperationError | None = None

    def __post_init__(self) -> None:
        if (self.item is None) == (self.error is None):
            raise ValueError("ScrapeResult must contain exactly one of item or error")
        if self.stage is ScrapeStage.PAGE and self.item is not None:
            raise ValueError("a page-stage ScrapeResult cannot contain an item")

    @property
    def succeeded(self) -> bool:
        """Return true only for a successfully constructed and loaded item."""
        return self.error is None

    def unwrap(self) -> MediaT:
        """Return the item or raise the typed terminal scrape error."""
        if self.error is not None:
            raise self.error
        return cast(MediaT, self.item)


class ScrapeStream(Generic[MediaT]):
    """
    Async iterator/context manager owning a Helper scheduler.

    Exhausting the iterator cleans it up naturally.  When a caller may ``break``
    early, use ``async with`` so ``__aexit__`` immediately cancels outstanding page
    and item tasks instead of waiting for async-generator garbage collection.
    """

    def __init__(self, generator: AsyncGenerator[ScrapeResult[MediaT], None]) -> None:
        self._generator = generator
        self._closed = False

    def __aiter__(self) -> ScrapeStream[MediaT]:
        return self

    async def __anext__(self) -> ScrapeResult[MediaT]:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise

    async def __aenter__(self) -> ScrapeStream[MediaT]:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel scheduler work and close the underlying async generator once."""
        if self._closed:
            return
        self._closed = True
        await self._generator.aclose()


@dataclass(frozen=True, slots=True)
class _PageJob:
    index: int # The Index of the Page
    url: str # The URL of the Page


@dataclass(frozen=True, slots=True)
class _ItemJob:
    page_index: int # The Index of the Page (where the item was fetched from)
    item_index: int # The Item Index (needed to keep total order)
    url: str # The URL of the Item e.g., Video, Short URL
    data: dict[str, Any] # The actual item data defined by the extractor. Given into the constructor class


@dataclass(frozen=True, slots=True)
class _AttemptOutcome(Generic[OperationT]):
    value: OperationT | None
    error: ScrapeOperationError | None
    action: ErrorAction | None
    attempts: int


@dataclass(frozen=True, slots=True)
class _PageOutcome(Generic[MediaT]):
    job: _PageJob # Contains the Page Job class
    items: tuple[_ItemJob, ...] # Contains the Items, so the Videos, Shorts whatever (as ItemJob class)
    result: ScrapeResult[MediaT] | None # The actual Scrape Result


@dataclass(frozen=True, slots=True)
class _ItemOutcome(Generic[MediaT]):
    job: _ItemJob # The Item Job class
    result: ScrapeResult[MediaT] | None # The Scrape Result


class _OrderedResultBuffer(Generic[MediaT]):
    """Isolate original-order bookkeeping from the concurrency scheduler."""

    def __init__(self, page_count: int) -> None:
        self._page_count = page_count
        self._page_sizes: dict[int, int] = {}
        self._page_results: dict[int, ScrapeResult[MediaT] | None] = {}
        self._item_results: dict[
            tuple[int, int], ScrapeResult[MediaT] | None
        ] = {}
        self._next_page = 0
        self._next_item = 0
        self._page_result_emitted = False

    def add_page(self, outcome: _PageOutcome[MediaT]) -> None:
        self._page_sizes[outcome.job.index] = len(outcome.items)
        self._page_results[outcome.job.index] = outcome.result

    def add_item(self, outcome: _ItemOutcome[MediaT]) -> None:
        self._item_results[(outcome.job.page_index, outcome.job.item_index)] = (
            outcome.result
        )

    def drain(self) -> list[ScrapeResult[MediaT]]:
        """Return every now-contiguous result in page/extractor order."""
        ready: list[ScrapeResult[MediaT]] = []
        while self._next_page < self._page_count:
            if self._next_page not in self._page_sizes:
                break

            if not self._page_result_emitted:
                page_result = self._page_results.pop(self._next_page)
                self._page_result_emitted = True
                if page_result is not None:
                    ready.append(page_result)

            page_size = self._page_sizes[self._next_page]
            blocked = False
            while self._next_item < page_size:
                key = (self._next_page, self._next_item)
                if key not in self._item_results:
                    blocked = True
                    break
                item_result = self._item_results.pop(key)
                self._next_item += 1
                if item_result is not None:
                    ready.append(item_result)

            if blocked:
                break

            self._page_sizes.pop(self._next_page)
            self._next_page += 1
            self._next_item = 0
            self._page_result_emitted = False
        return ready


class Helper(Generic[MediaT]):
    """
    Concurrent two-stage scraper using bounded, dynamically managed task sets.

    Page tasks fetch and extract item dictionaries.  Item tasks construct a
    ``BaseMedia`` subclass and optionally load selected fields or sources.  There
    are no permanent workers, queues, sentinels, queue joins, or ``TaskGroup``.
    Completion is the explicit condition that page input, pending items, and both
    task sets are empty.

    ``ResultOrder.COMPLETION`` is the default and yields items as soon as their
    tasks finish. ``ResultOrder.ORIGINAL`` buffers only completed outcomes needed
    to restore target-page order and extractor order.
    """

    def __init__(
        self,
        core: BaseCore,
        constructor: Callable[..., MediaT],
        *,
        logger: logging.Logger | None = None,
        log_name: str = "helper.iterator",
        log_file: str | None = None,
        log_level: int = logging.INFO,
        http_ip: str | None = None,
        http_port: int | str | None = None,
    ) -> None:
        self.core = core # The Networking Backend
        self.constructor = constructor # The class that takes the data as input (dataclass) defined by each API
        self.logger = logger or configure_app_logging(
            log_name,
            log_file=log_file,
            level=log_level,
            http_ip=http_ip,
            http_port=http_port,
        )

    def iterator(
        self,
        target_page_urls: Sequence[str], # This is just a list of target URLs to scrape from
        item_extractor: Callable[[Any], Iterable[Mapping[str, Any]]], # The extractor that uses selectolax to parse data from the page
        *,
        iterator_config: IteratorConfig
    ) -> ScrapeStream[MediaT]:
        """
        Create a lazily started scrape stream.

        Extractors are synchronous callables returning an iterable of mappings.
        By default the complete extractor iteration runs in a worker thread so
        HTML parsing cannot block the event loop. Every mapping must contain a
        non-empty string under ``item_url_key`` and must be accepted as keyword
        arguments by ``constructor`` in addition to ``core``.

        ``load_sources`` runs before ``load_fields`` for each new instance. Both
        are optional; with neither configured, Helper only constructs models.
        Expected failures follow each stage's bounded retry policy and terminal
        error mode. Handler decisions can override the terminal mode but cannot
        exceed ``RetryPolicy.max_attempts``.
        """
        urls = tuple(target_page_urls)
        iterator_config = iterator_config.resolve(self.core.configuration)

        max_page_concurrency = iterator_config.max_page_concurrency
        max_item_concurrency = iterator_config.max_item_concurrency
        max_pending_items = iterator_config.max_pending_items
        extract_in_thread = iterator_config.extract_in_thread
        order = iterator_config.order
        page_error_mode = iterator_config.page_error_mode
        item_error_mode = iterator_config.item_error_mode
        page_retry = iterator_config.page_retry
        item_retry = iterator_config.item_retry
        page_error_handler = iterator_config.page_error_handler
        item_error_handler = iterator_config.item_error_handler
        load_fields = iterator_config.load_specific_fields
        load_sources = iterator_config.load_specific_sources
        page_request_method = iterator_config._page_request_method
        item_url_key = iterator_config._item_url_key

        # Validation of inputs
        if any(not isinstance(url, str) or not url for url in urls):
            raise ValueError("target_page_urls must contain non-empty strings")
        if not callable(item_extractor):
            raise TypeError("item_extractor must be callable")
        if max_page_concurrency < 1 or max_item_concurrency < 1:
            raise ValueError("page and item concurrency must both be at least 1")
        if max_pending_items is None:
            max_pending_items = max_item_concurrency * 4
        if max_pending_items < 1:
            raise ValueError("max_pending_items must be at least 1")
        if not isinstance(item_url_key, str) or not item_url_key:
            raise ValueError("item_url_key must be a non-empty string")

        normalized_order = ResultOrder(order)
        normalized_page_mode = ErrorMode(page_error_mode)
        normalized_item_mode = ErrorMode(item_error_mode)
        normalized_fields = tuple(dict.fromkeys(load_fields))
        normalized_sources = tuple(dict.fromkeys(load_sources))

        generator = self._iterate(
            urls=urls,
            item_extractor=item_extractor,
            max_page_concurrency=max_page_concurrency,
            max_item_concurrency=max_item_concurrency,
            max_pending_items=max_pending_items,
            page_request_method=page_request_method,
            item_url_key=item_url_key,
            extract_in_thread=extract_in_thread,
            load_fields=normalized_fields,
            load_sources=normalized_sources,
            order=normalized_order,
            page_error_mode=normalized_page_mode,
            item_error_mode=normalized_item_mode,
            page_retry=page_retry or RetryPolicy(),
            item_retry=item_retry or RetryPolicy(),
            page_error_handler=page_error_handler,
            item_error_handler=item_error_handler,
        )
        return ScrapeStream(generator)

    async def _iterate(
        self,
        *,
        urls: tuple[str, ...],
        item_extractor: Callable[[Any], Iterable[Mapping[str, Any]]],
        max_page_concurrency: int,
        max_item_concurrency: int,
        max_pending_items: int,
        page_request_method: str,
        item_url_key: str,
        extract_in_thread: bool,
        load_fields: tuple[str, ...],
        load_sources: tuple[str, ...],
        order: ResultOrder,
        page_error_mode: ErrorMode,
        item_error_mode: ErrorMode,
        page_retry: RetryPolicy,
        item_retry: RetryPolicy,
        page_error_handler: ErrorHandler | None,
        item_error_handler: ErrorHandler | None,
    ) -> AsyncGenerator[ScrapeResult[MediaT], None]:
        page_cursor = 0
        pending_items: deque[_ItemJob] = deque()
        page_tasks: dict[asyncio.Task[_PageOutcome[MediaT]], _PageJob] = {}
        item_tasks: dict[asyncio.Task[_ItemOutcome[MediaT]], _ItemJob] = {}
        ordered = _OrderedResultBuffer[MediaT](len(urls))

        try:
            while (
                page_cursor < len(urls) # As long as not all URLs have been processed
                or page_tasks # As long as page tasks still exist
                or pending_items # As long as pending items still exist (not yet fired up as a task)
                or item_tasks # As long as item tasks still exit
            ):
                # A bounded backlog provides backpressure: when item processing is
                # slower than page extraction, no additional pages are started.
                while (
                    page_cursor < len(urls) # As long as not all URLs have been processed
                    and len(page_tasks) < max_page_concurrency # As long as the page task count is smaller than the total page concurrency
                    and len(pending_items) < max_pending_items # __.__
                ):
                    page_job = _PageJob(page_cursor, urls[page_cursor]) # Creates a job for fetching a page
                    page_task = asyncio.create_task(
                        self._process_page(
                            page_job,
                            item_extractor=item_extractor,
                            request_method=page_request_method,
                            item_url_key=item_url_key,
                            extract_in_thread=extract_in_thread,
                            retry_policy=page_retry,
                            error_mode=page_error_mode,
                            error_handler=page_error_handler,
                        ),
                        name=f"scrape-page-{page_job.index}",
                    )
                    page_tasks[page_task] = page_job
                    page_cursor += 1

                while pending_items and len(item_tasks) < max_item_concurrency:
                    item_job = pending_items.popleft()
                    item_task = asyncio.create_task(
                        self._process_item(
                            item_job,
                            load_fields=load_fields,
                            load_sources=load_sources,
                            retry_policy=item_retry,
                            error_mode=item_error_mode,
                            error_handler=item_error_handler,
                        ),
                        name=(
                            f"scrape-item-{item_job.page_index}-"
                            f"{item_job.item_index}"
                        ),
                    )
                    item_tasks[item_task] = item_job

                active_tasks: set[asyncio.Task[Any]] = set(page_tasks)
                active_tasks.update(item_tasks)
                if not active_tasks:
                    # The loop condition says work remains, so reaching this branch
                    # would indicate a scheduler invariant bug rather than a remote
                    # scrape failure.
                    raise RuntimeError("Helper scheduler has pending work but no active tasks")

                done, _ = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                ready_results: list[ScrapeResult[MediaT]] = []

                for generic_task in done:
                    if generic_task in page_tasks:
                        page_task = cast(
                            asyncio.Task[_PageOutcome[MediaT]], generic_task
                        )
                        page_tasks.pop(page_task)
                        page_outcome = page_task.result()

                        if order is ResultOrder.ORIGINAL:
                            ordered.add_page(page_outcome)
                        elif page_outcome.result is not None:
                            ready_results.append(page_outcome.result)

                        pending_items.extend(page_outcome.items)
                    else:
                        item_task = cast(
                            asyncio.Task[_ItemOutcome[MediaT]], generic_task
                        )
                        item_tasks.pop(item_task)
                        item_outcome = item_task.result()

                        if order is ResultOrder.ORIGINAL:
                            ordered.add_item(item_outcome)
                        elif item_outcome.result is not None:
                            ready_results.append(item_outcome.result)

                if order is ResultOrder.ORIGINAL:
                    ready_results.extend(ordered.drain())

                for result in ready_results:
                    yield result
        finally:
            # This is the single owner of all scheduler tasks. It runs on normal
            # exhaustion, typed failure, caller cancellation, or ScrapeStream.close.
            remaining_tasks: list[asyncio.Task[Any]] = [*page_tasks, *item_tasks]
            for task in remaining_tasks:
                task.cancel()
            if remaining_tasks:
                await asyncio.gather(*remaining_tasks, return_exceptions=True)

    async def _process_page(
        self,
        job: _PageJob,
        *,
        item_extractor: Callable[[Any], Iterable[Mapping[str, Any]]],
        request_method: str,
        item_url_key: str,
        extract_in_thread: bool,
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
    ) -> _PageOutcome[MediaT]:
        async def operation() -> tuple[_ItemJob, ...]:
            self.logger.debug("Fetching page %s: %s", job.index, job.url)
            content = await self.core.fetch_text(job.url, method=request_method)

            def extract_all() -> tuple[Mapping[str, Any], ...]:
                extracted = item_extractor(content)
                if inspect.isawaitable(extracted):
                    raise TypeError(
                        "item_extractor must be synchronous; Helper can move it "
                        "to a worker thread"
                    )
                return tuple(extracted)

            if extract_in_thread:
                extracted_items = await asyncio.to_thread(extract_all)
            else:
                extracted_items = extract_all()

            jobs: list[_ItemJob] = []
            for item_index, raw_item in enumerate(extracted_items):
                if not isinstance(raw_item, Mapping):
                    raise TypeError(
                        "item_extractor entries must be mappings; received "
                        f"{type(raw_item).__name__} at index {item_index}"
                    )
                data = dict(raw_item)
                item_url = data.get(item_url_key)
                if not isinstance(item_url, str) or not item_url:
                    raise ValueError(
                        f"extractor item {item_index} must contain a non-empty "
                        f"string under key {item_url_key!r}"
                    )
                jobs.append(_ItemJob(job.index, item_index, item_url, data))
            return tuple(jobs)

        attempt = await self._run_operation(
            operation,
            stage=ScrapeStage.PAGE,
            url=job.url,
            page_index=job.index,
            item_index=None,
            retry_policy=retry_policy,
            error_mode=error_mode,
            error_handler=error_handler,
            error_factory=lambda error, number: PageFetchError(
                job.url, error, number, job.index
            ),
        )
        if attempt.error is None:
            return _PageOutcome(job, cast(tuple[_ItemJob, ...], attempt.value), None)

        result: ScrapeResult[MediaT] | None = None
        if attempt.action is ErrorAction.YIELD:
            result = ScrapeResult(
                stage=ScrapeStage.PAGE,
                url=job.url,
                page_index=job.index,
                item_index=None,
                attempts=attempt.attempts,
                error=attempt.error,
            )
        return _PageOutcome(job, (), result)

    async def _process_item(
        self,
        job: _ItemJob,
        *,
        load_fields: tuple[str, ...],
        load_sources: tuple[str, ...],
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
    ) -> _ItemOutcome[MediaT]:
        async def operation() -> MediaT:
            instance = self.constructor(core=self.core, **job.data)
            if not isinstance(instance, BaseMedia):
                raise TypeError(
                    "Helper constructors must return a BaseMedia instance; "
                    f"received {type(instance).__name__}"
                )
            if load_sources:
                await instance.load_sources(*load_sources)
            if load_fields:
                await instance.load_fields(*load_fields)
            return instance

        attempt = await self._run_operation(
            operation,
            stage=ScrapeStage.ITEM,
            url=job.url,
            page_index=job.page_index,
            item_index=job.item_index,
            retry_policy=retry_policy,
            error_mode=error_mode,
            error_handler=error_handler,
            error_factory=lambda error, number: ItemFetchError(
                job.url,
                error,
                number,
                job.page_index,
                job.item_index,
            ),
        )
        if attempt.error is None:
            success_result = ScrapeResult(
                stage=ScrapeStage.ITEM,
                url=job.url,
                page_index=job.page_index,
                item_index=job.item_index,
                attempts=attempt.attempts,
                item=cast(MediaT, attempt.value),
            )
            return _ItemOutcome(job, success_result)

        failure_result: ScrapeResult[MediaT] | None = None
        if attempt.action is ErrorAction.YIELD:
            failure_result = ScrapeResult(
                stage=ScrapeStage.ITEM,
                url=job.url,
                page_index=job.page_index,
                item_index=job.item_index,
                attempts=attempt.attempts,
                error=attempt.error,
            )
        return _ItemOutcome(job, failure_result)

    async def _run_operation(
        self,
        operation: Callable[[], Awaitable[OperationT]],
        *,
        stage: ScrapeStage,
        url: str,
        page_index: int,
        item_index: int | None,
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
        error_factory: Callable[[Exception, int], ScrapeOperationError],
    ) -> _AttemptOutcome[OperationT]:
        """Run one bounded retry loop and convert its terminal disposition."""
        for attempt in range(1, retry_policy.max_attempts + 1):
            try:
                value = await operation()
                return _AttemptOutcome(value, None, None, attempt)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Some callers yield or skip terminal failures instead of raising
                # them. Log while the exception is actively handled so its real
                # traceback, including the originating source line, is never lost.
                self.logger.exception(
                    "Exception while processing %s %s on attempt %s/%s",
                    stage.value,
                    url,
                    attempt,
                    retry_policy.max_attempts,
                )
                context = ScrapeErrorContext(
                    stage=stage,
                    url=url,
                    error=error,
                    attempt=attempt,
                    max_attempts=retry_policy.max_attempts,
                    page_index=page_index,
                    item_index=item_index,
                )
                action = await self._error_action(
                    context,
                    retry_policy=retry_policy,
                    error_mode=error_mode,
                    error_handler=error_handler,
                )

                # BaseCore.request already treats 404 as non-retryable. Keep the
                # page-level retry loop from undoing that decision, including when
                # the HTTP error has been wrapped by another library exception.
                if (
                    stage is ScrapeStage.PAGE
                    and action is ErrorAction.RETRY
                    and _contains_http_status(error, 404)
                ):
                    action = ErrorAction(error_mode.value)

                if action is ErrorAction.RETRY and attempt < retry_policy.max_attempts:
                    delay = retry_policy.delay_after(attempt)
                    self.logger.warning(
                        "Retrying %s %s after attempt %s/%s in %.3fs: %s",
                        stage.value,
                        url,
                        attempt,
                        retry_policy.max_attempts,
                        delay,
                        error,
                    )
                    if delay:
                        await asyncio.sleep(delay)
                    continue

                # A handler cannot create an unbounded retry loop. RETRY on the
                # final permitted attempt falls back to the configured mode.
                if action is ErrorAction.RETRY:
                    action = ErrorAction(error_mode.value)

                wrapped_error = error_factory(error, attempt)
                # YIELD stores this exception for a later ScrapeResult.unwrap().
                # Explicitly retain the cause so that raising it later renders the
                # original traceback rather than only the unwrap() line.
                wrapped_error.__cause__ = error
                wrapped_error.__suppress_context__ = True
                if action is ErrorAction.RAISE:
                    raise wrapped_error from error
                return _AttemptOutcome(None, wrapped_error, action, attempt)

        raise RuntimeError("retry loop exhausted without returning an outcome")

    async def _error_action(
        self,
        context: ScrapeErrorContext,
        *,
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
    ) -> ErrorAction:
        """Resolve automatic policy or validate a custom handler decision."""
        if error_handler is None:
            if (
                context.attempt < retry_policy.max_attempts
                and retry_policy.permits(context.error)
            ):
                return ErrorAction.RETRY
            return ErrorAction(error_mode.value)

        try:
            decision = error_handler(context)
            if inspect.isawaitable(decision):
                decision = await decision
            if not isinstance(decision, ErrorAction):
                raise TypeError(
                    "error handlers must return an ErrorAction value, received "
                    f"{decision!r}"
                )
            return decision
        except asyncio.CancelledError:
            raise
        except Exception as handler_error:
            raise ErrorHandlerError(
                context.stage.value, context.url, handler_error
            ) from handler_error


class ProgressiveAction(StrEnum):
    """What one attempt of a progressive download tells its orchestrator to do.

    Terminal failures are raised, not returned - the caller has to be able to
    tell "this download is over" from "this attempt is over". These four are
    the outcomes that leave the download itself still alive.
    """

    #: The body was streamed to its end. The file still has to pass the size
    #: check before it is moved into place.
    COMPLETED = "completed"
    #: A 416 whose stated total matches the temporary file exactly: the bytes
    #: are already all of them, and only the finalization is missing.
    ALREADY_COMPLETE = "already_complete"
    #: The remote resource cannot be reconciled with the local partial file.
    #: Recovered by discarding both and starting at byte zero, exactly once.
    RESTART = "restart"
    #: Transient: retry the same offset after a backoff.
    RETRY = "retry"
    #: The stop event fired.
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ProgressiveOutcome:
    """One attempt's result, including everything a resume needs to survive it.

    The validators travel back with the outcome so that a connection that dies
    mid-body still teaches the resume state which resource those bytes came
    from - without that, the next run has bytes it cannot vouch for.
    """

    action: ProgressiveAction
    written: int = 0
    total: int | None = None
    etag: str | None = None
    etag_weak: bool = False
    last_modified: str | None = None
    retry_after: float | None = None
    reason: str = ""

    @property
    def validators(self) -> Dict[str, Any]:
        return {
            "etag": self.etag,
            "etag_weak": self.etag_weak,
            "last_modified": self.last_modified,
        }


class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(
        self,
        configuration: "RuntimeConfig" = config,
        *,
        cache: CacheBackend | None = None,
    ) -> None:
        self.lock = asyncio.Lock()
        self._delay_lock = asyncio.Lock()
        self._cache_flight_lock = asyncio.Lock()
        self._inflight_text_requests: dict[RequestCacheKey, asyncio.Future[str]] = {}
        self.latest_key: str | None = None
        self.latest_key_time: float = 0.0
        self.last_request_time: float | None = None
        self.total_requests: int = 0  # Tracks how many requests have been made
        self.session: AsyncSession | None = None
        self.configuration = configuration
        self.cache = cache if cache is not None else Cache(self.configuration)
        self.logger = configure_app_logging("BASE API - [BaseCore]", log_file=None, level=logging.ERROR)
        self.default_headers = {
            "Accept-Language": self.configuration.locale,
        }

    async def __aenter__(self) -> Self:
        if self.session is None:
            self.initialize_session()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the owned HTTP session and allow the core to be reused later."""
        session, self.session = self.session, None
        if session is not None:
            await session.close()

    def enable_logging(self, log_file: str | None = None, level: int = logging.DEBUG, log_ip:
    str | None = None, log_port: int | str | None = None) -> None:
        """Enables logging dynamically for this module."""
        self.logger = configure_app_logging("BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip,
                                   http_port=log_port)

    def initialize_session(self) -> None:
        """Initialize the owned HTTP session once for the current lifecycle."""
        if self.session is not None:
            return

        verify = self.configuration.verify_ssl

        curl_options: Dict[CurlOpt, Union[bytes, int]] = {}
        if self.configuration.dns_over_https:
            curl_options[CurlOpt.DOH_URL] = str(self.configuration.dns_over_https).encode("utf-8")

        proxy = None
        if self.configuration.proxy:
            proxy = self.configuration.proxy

        if self.configuration.max_bandwidth_mb is not None and self.configuration.max_bandwidth_mb > 0:
            global_limit_bytes = int(self.configuration.max_bandwidth_mb * 1024 * 1024)
            total_concurrent_connections = (self.configuration.max_workers_download *
                                            self.configuration.videos_concurrency)
            per_connection_limit = max(1, int(global_limit_bytes / total_concurrent_connections))
            curl_options[CurlOpt.MAX_RECV_SPEED_LARGE] = per_connection_limit

        js3 = self.configuration.custom_ja3
        impersonation = self.configuration.impersonation
        http_version = self.configuration.http_version
        proxy_auth_str = self.configuration.proxy_auth
        trust_env = self.configuration.trust_env

        p_auth: Tuple[str, str] | None = None
        if proxy_auth_str and ":" in proxy_auth_str:
            u, p = proxy_auth_str.split(":", 1)
            p_auth = (u, p)

        self.session = cast(Any, AsyncSession)(
            interface=self.configuration.interface,
            proxy=proxy,
            timeout=self.configuration.timeout,
            verify=verify,
            ja3=js3,
            http_version=http_version,
            impersonate=impersonation,
            curl_options=curl_options,
            proxy_auth=p_auth,
            trust_env=trust_env,
            cookies=self.configuration.cookies,
        )
        # Ensure our defaults are on the session
        assert self.session is not None
        self.session.headers.update(self.default_headers)

    async def enforce_delay(self) -> None:
        """Enforce this core's configured delay between requests, when enabled."""
        delay = self.configuration.request_delay
        if delay and delay > 0:
            async with self._delay_lock:
                now = time.monotonic()
                if self.last_request_time is None:
                    self.last_request_time = now
                    return
                time_since_last_request = now - self.last_request_time
                self.logger.debug(
                    "Time since last request: %.2f seconds.", time_since_last_request
                )
                if time_since_last_request < delay:
                    sleep_time = delay - time_since_last_request
                    self.logger.debug("Enforcing delay of %.2f seconds.", sleep_time)
                    await asyncio.sleep(sleep_time)
                self.last_request_time = time.monotonic()

    def _merged_headers(self, override: Dict[str, str] | None) -> Dict[str, Any]:
        """
        Create request headers from current session headers + optional overrides.
        Overrides win, session headers are the base. The session itself is never
        modified here - an override exists for exactly one request.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        headers: Dict[str, Any] = cast(Dict[str, Any], cast(Any, dict(session.headers)))
        if override:
            for key, value in override.items():
                # HTTP header names are case-insensitive, but the session stores
                # its keys lowercased while callers write "Referer". A plain
                # dict.update would keep both spellings and put two lines on the
                # wire; replace case-insensitively so the override really wins.
                lower = key.lower()
                for existing in [k for k in headers if k.lower() == lower]:
                    del headers[existing]
                headers[key] = value
        return headers

    def _merged_cookies(self, override: Dict[str, str] | None) -> Dict[str, Any]:
        """Same as above, but for cookies"""
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        cookies: Dict[str, Any] = cast(Dict[str, Any], cast(Any, session.cookies.get_dict()))
        if override:
            cookies.update(override)
        return cookies

    async def request(
        self,
        url: str,
        *,
        timeout: float | None = None,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        retry_non_idempotent: bool = False,
    ) -> Response:
        """
        Execute an HTTP request and return a successful response.

        Network failures, HTTP 408/425/429, and 5xx responses are retried for
        idempotent methods. Retrying a non-idempotent method requires an explicit
        opt-in because the server may already have applied the request.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        request_method = method.upper()
        req_timeout = timeout if timeout is not None else self.configuration.timeout
        max_attempts = max(1, int(self.configuration.request_attempts))
        method_is_retryable = request_method in {
            "GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"
        } or retry_non_idempotent

        def should_retry(error: BaseException) -> bool:
            if not method_is_retryable:
                return False
            # A missing resource is a terminal response. Keep this explicit so a
            # future broadening of retryable HTTP errors cannot include 404.
            if _contains_http_status(error, 404):
                return False
            if isinstance(error, (RequestsError, NetworkRequestError)):
                return True
            return isinstance(error, HTTPStatusError) and (
                error.status_code in {408, 425, 429}
                or 500 <= error.status_code < 600
            )

        exponential_wait = wait_exponential_jitter(
            initial=self.configuration.request_retry_initial_delay,
            max=self.configuration.request_retry_max_delay,
            exp_base=self.configuration.request_multiplier,
            jitter=self.configuration.request_retry_jitter,
        )

        def retry_wait(retry_state: Any) -> float:
            error = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(error, RateLimitError) and error.retry_after is not None:
                return max(0.0, error.retry_after)
            return cast(float, exponential_wait(retry_state))

        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=retry_wait,
            retry=retry_if_exception(should_retry),
            reraise=False,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    try:
                        await self.enforce_delay()
                        req_headers = self._merged_headers(headers)
                        req_cookies = self._merged_cookies(cookies)
                        
                        current_time = asyncio.get_running_loop().time()
                        latest_key = self.latest_key
                        if "KEY" not in session.cookies and latest_key is not None:
                            if current_time - self.latest_key_time < 10:
                                session.cookies.set("KEY", latest_key, domain=".pornhub.com", path="/")

                        self.total_requests += 1
                        response = await cast(Any, session).request(
                            method=cast(Any, request_method),
                            url=url,
                            timeout=req_timeout,
                            allow_redirects=allow_redirects,
                            data=data,
                            json=json_data,
                            params=params,
                            headers=req_headers,
                            cookies=req_cookies,
                        )

                        status = response.status_code

                        content_type = response.headers.get("content-type", "").lower()
                        is_html = "text/html" in content_type if content_type else True

                        if is_html:
                            enc = getattr(response, "encoding", None) or "utf-8"
                            resp_text = cast(bytes, response.content).decode(enc, errors="replace")

                            if 'onload="go()"' in resp_text:
                                local_latest = getattr(self, "latest_key", None)
                                async with self.lock:
                                    if getattr(self, "latest_key", None) != local_latest:
                                        self.logger.info("Another task already resolved the challenge! Retrying request with the new cookie.")
                                        if self.latest_key:
                                            session.cookies.set("KEY", self.latest_key, domain=".pornhub.com", path="/")

                                        await asyncio.sleep(1.5)
                                        raise NetworkRequestError("Retrying request with the new cookie.")

                                    self.logger.info("Challenge page detected! Solving...")
                                    get_challenge = re.compile(r'go\(\).*?{(.*?)n=l.*?KEY.*?s\+":(\d+):', re.DOTALL)
                                    challenge_data = re.search(get_challenge, resp_text)

                                    if challenge_data:
                                        try:
                                            challenge_str, token_str = challenge_data.groups()
                                            code = parse_challenge(challenge_str)
                                            code = other_challenge(code)
                                            code = '\n'.join(code.split(';'))

                                            safe_chars = set(string.ascii_letters + string.digits + " \t\n=+-*/().:><&|~^")
                                            if not all(c in safe_chars for c in code):
                                                self.logger.error("Security Abort: Illegal chars in challenge, CODE: %s", code)
                                                raise SecurityAbort

                                            safe_globals: Dict[str, Any] = {"__builtins__": {}}
                                            safe_locals = {"p": 0, "s": 0}
                                            exec(code, safe_globals, safe_locals)

                                            p = safe_locals.get('p', 0)
                                            s = safe_locals.get('s', 0)
                                            n = least_factors(p)
                                            cookie_value = f'{n}*{p // n}:{s}:{token_str}:1'

                                            self.latest_key = cookie_value
                                            self.latest_key_time = asyncio.get_running_loop().time()
                                            session.cookies.set("KEY", cookie_value, domain=".pornhub.com", path="/")
                                            self.logger.info("RESOLVED CHALLENGE! Injected cookie: %s", cookie_value)

                                            try:
                                                self.cache.invalidate_url(url)
                                            except (KeyError, Exception):
                                                pass

                                            await asyncio.sleep(1.5)
                                            raise NetworkRequestError("Retrying request after solving challenge.")
                                        except (NetworkRequestError, SecurityAbort):
                                            raise
                                        except Exception as challenge_error:
                                            raise ChallengeMathError from challenge_error

                                    else:
                                        self.logger.error("Detected challenge page, but the regex failed to extract data.")
                                        await asyncio.sleep(1.5)
                                        raise ChallengeRegexError("Detected Challenge, but regex couldn't extract, report this!")


                        if 200 <= status < 300:
                            self.logger.debug("Successfully fetched URL: %s", url)
                            return response

                        if status in {401, 403}:
                            raise AccessDeniedError("Request blocked by server!")

                        if status == 412:
                            log_precondition_failed(logger=self.logger, attempt=attempt.retry_state.attempt_number, response=response)

                        if status == 410:
                            raise ResourceGone(f"Resource gone (HTTP 410) for URL: {url}")

                        if status == 429:
                            retry_after = parse_retry_after(
                                logger=self.logger, response=response
                            )
                            if retry_after is not None:
                                self.logger.warning(
                                    "Rate limited (429). Server requested %ss pause.",
                                    retry_after,
                                )
                            raise RateLimitError(
                                "429 Rate Limited", retry_after=retry_after, url=url
                            )

                        if 500 <= status < 600:
                            self.logger.warning("Server error %s on %s. Retrying...", status, url)
                            raise HTTPStatusError(f"Server error {status}", status_code=status, url=url)

                        self.logger.info("HTTP %s for %s.", status, url)
                        raise HTTPStatusError(
                            f"HTTP {status} for {url}", status_code=status, url=url
                        )

                    except RequestsError as e:
                        err_str = str(e).lower()
                        self.logger.error("Request error for URL %s: %s", url, e, exc_info=True)
                        if "certificate verify failed" in err_str:
                            raise ProxySSLError("Proxy has an invalid SSL certificate, set 'verify = False' in config") from e
                        elif "cookie conflict" in err_str:
                            raise UnknownError(f"Cookie conflict during request to {url}: {e}") from e
                        elif "proxy" in err_str:
                            raise InvalidProxy("Proxy error when trying a request, aborting!") from e
                        elif "timeout" in err_str or "read" in err_str:
                            self.logger.error("Timeout for URL %s: %s", url, e, exc_info=True)
                        raise
                    except (BaseScraperError, ResourceGone, ProxySSLError, InvalidProxy, UnknownError):
                        raise

                    except Exception as e:
                        self.logger.error("Unexpected error for %s: %s\n%s", url, e, traceback.format_exc())
                        raise UnknownError(f"Unexpected error for URL {url}: {e}") from e

        except RetryError as re_err:
            last_error = re_err.last_attempt.exception()
            if not isinstance(last_error, Exception):
                last_error = NetworkRequestError("Request retry budget was exhausted")
            self.logger.error(
                "Request to %s failed after %s attempts.", url, max_attempts
            )
            raise RequestRetriesExhausted(url, max_attempts, last_error) from last_error

        raise RuntimeError("request retry controller exited without an outcome")

    def _request_cache_key(
        self,
        *,
        url: str,
        method: str,
        allow_redirects: bool,
        params: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
        json_data: Mapping[str, Any] | None,
        headers: Dict[str, str] | None,
        cookies: Dict[str, str] | None,
    ) -> RequestCacheKey:
        merged_headers = {
            str(key).lower(): value
            for key, value in self._merged_headers(headers).items()
        }
        merged_cookies = self._merged_cookies(cookies)
        return RequestCacheKey(
            method=method.upper(),
            url=url,
            allow_redirects=allow_redirects,
            params_fingerprint=_cache_fingerprint(params),
            body_fingerprint=_cache_fingerprint((data, json_data)),
            headers_fingerprint=_cache_fingerprint(merged_headers),
            cookies_fingerprint=_cache_fingerprint(merged_cookies),
        )

    @staticmethod
    def _decode_response(response: Response, url: str, logger: logging.Logger) -> str:
        raw_content = cast(bytes, response.content)
        encoding = getattr(response, "encoding", None) or "utf-8"
        try:
            return raw_content.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            logger.warning(
                "Content could not be decoded as %s (%s), decoding latin1 instead!",
                encoding,
                url,
            )
            return raw_content.decode("latin1", errors="replace")

    async def fetch_text(
        self,
        url: str,
        *,
        cache_policy: CachePolicy = CachePolicy.USE,
        timeout: float | None = None,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        retry_non_idempotent: bool = False,
    ) -> str:
        """Fetch and decode text, optionally using the bounded response cache."""
        request_method = method.upper()
        cacheable = request_method == "GET" and cache_policy is not CachePolicy.BYPASS
        key = None
        if cacheable:
            key = self._request_cache_key(
                url=url,
                method=request_method,
                allow_redirects=allow_redirects,
                params=params,
                data=data,
                json_data=json_data,
                headers=headers,
                cookies=cookies,
            )

        if key is not None and cache_policy is CachePolicy.USE:
            cached = self.cache.get_response(key)
            if cached is not None:
                self.logger.info("Fetched content for %s from cache.", url)
                return cached

        leader = True
        pending: asyncio.Future[str] | None = None
        if key is not None:
            async with self._cache_flight_lock:
                if cache_policy is CachePolicy.USE:
                    cached = self.cache.get_response(key)
                    if cached is not None:
                        return cached
                pending = self._inflight_text_requests.get(key)
                if pending is None:
                    pending = asyncio.get_running_loop().create_future()
                    self._inflight_text_requests[key] = pending
                else:
                    leader = False

        if not leader:
            assert pending is not None
            return await asyncio.shield(pending)

        try:
            response = await self.request(
                url,
                timeout=timeout,
                cookies=cookies,
                allow_redirects=allow_redirects,
                data=data,
                method=request_method,
                headers=headers,
                json_data=json_data,
                params=params,
                retry_non_idempotent=retry_non_idempotent,
            )
            content = self._decode_response(response, url, self.logger)
            if key is not None:
                self.cache.set_response(key, content)
                assert pending is not None
                if not pending.done():
                    pending.set_result(content)
            return content
        except BaseException as error:
            if pending is not None and not pending.done():
                pending.set_exception(error)
                # Mark the exception as observed even when there were no followers.
                pending.exception()
            raise
        finally:
            if key is not None:
                async with self._cache_flight_lock:
                    if self._inflight_text_requests.get(key) is pending:
                        self._inflight_text_requests.pop(key, None)

    async def fetch_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        retry_non_idempotent: bool = False,
    ) -> bytes:
        """Fetch a response body as bytes without involving the text cache."""
        response = await self.request(
            url,
            timeout=timeout,
            cookies=cookies,
            allow_redirects=allow_redirects,
            data=data,
            method=method,
            headers=headers,
            json_data=json_data,
            params=params,
            retry_non_idempotent=retry_non_idempotent,
        )
        return cast(bytes, response.content)

    async def get_m3u8_by_quality(
            self,
            m3u8_url: str,
            quality: str | int,
            headers: Dict[str, str] | None = None,
    ) -> str:
        """
        Return the media-playlist URL for the requested quality.

        Supported preferences:
            best
            half
            worst

        Supported explicit qualities:
            144
            240
            360
            480
            540
            720
            1080
            1440
            2160

        Numeric strings such as "1080" and "1080p" are accepted too.
        """

        if m3u8 is None:
            raise ModuleNotFoundError(
                "HLS support requires the 'm3u8' package."
            )

        # Resolve callable / awaitable sources.
        if (
                inspect.iscoroutinefunction(m3u8_url)
                or (
                callable(m3u8_url)
                and not isinstance(m3u8_url, str)
        )
        ):
            m3u8_url = m3u8_url()

        if inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if not isinstance(m3u8_url, str):
            raise TypeError(
                "m3u8_url must resolve to a string."
            )

        # Load master playlist.
        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)
            base_url = None

            self.logger.debug(
                "Resolved inline/custom m3u8 master content."
            )

        else:
            content = await self.fetch_text(
                url=m3u8_url,
                headers=headers,
            )

            master = m3u8.loads(content)
            base_url = m3u8_url

            self.logger.debug(
                "Resolved m3u8 master: %s",
                m3u8_url,
            )

        if not master.is_variant:
            raise PlaylistExtractionError(
                f"Playlist is not a master playlist: {m3u8_url}"
            )

        variants = collect_variants(master)

        if not variants:
            raise PlaylistExtractionError(
                f"No usable video variants found: {m3u8_url}"
            )

        chosen = choose_variant(variants, quality)

        uri = chosen["uri"]

        # Master was fetched from a URL.
        if base_url is not None:
            return urljoin(
                base_url,
                uri,
            )

        # Inline playlist containing an absolute variant URL.
        if uri.startswith(("http://", "https://")):
            return uri

        raise PlaylistExtractionError(
            "Inline HLS master contains relative variant URLs, "
            "so they cannot be resolved without a base URL."
        )

    async def list_available_qualities(
            self,
            m3u8_url: str,
    ) -> list[int]:
        """
        Inspect an HLS master playlist and return canonical,
        sorted, unique qualities.
        """

        if m3u8 is None:
            raise ModuleNotFoundError(
                "HLS support requires the 'm3u8' package."
            )

        if (
                inspect.iscoroutinefunction(m3u8_url)
                or (
                callable(m3u8_url)
                and not isinstance(m3u8_url, str)
        )
        ):
            m3u8_url = m3u8_url()

        if inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if not isinstance(m3u8_url, str):
            raise TypeError(
                "m3u8_url must resolve to a string."
            )

        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)

        elif m3u8_url.startswith(("http://", "https://")):
            content = await self.fetch_text(
                url=m3u8_url
            )

            master = m3u8.loads(content)

        else:
            master = m3u8.loads(m3u8_url)

        if not master.is_variant:
            return []

        return available_qualities(collect_variants(master))

    @staticmethod
    def _parse_byterange(value: Any) -> tuple[int, int | None]:
        """Parse an HLS BYTERANGE value: "<length>[@<offset>]".

        Returns (length, offset); offset is None when the playlist omitted it,
        which means "continues the previous sub-range" for media segments and
        "start of the resource" for an EXT-X-MAP.
        """
        text = str(value).strip()
        length_part, sep, offset_part = text.partition("@")
        try:
            length = int(length_part)
            offset = int(offset_part) if sep else None
        except ValueError as error:
            raise PlaylistExtractionError(f"Invalid BYTERANGE value: {value!r}") from error
        if length <= 0 or (offset is not None and offset < 0):
            raise PlaylistExtractionError(f"Invalid BYTERANGE value: {value!r}")
        return length, offset

    async def get_segments(self, source: Any, quality: Union[str, int]) -> List[Any]:
        assert m3u8 is not None
        
        if getattr(source, "source_type", None) != "HLS":
            from base_api.modules.errors import UnsupportedProtocolError
            raise UnsupportedProtocolError(f"Unsupported source type: {getattr(source, 'source_type', 'None')}")
            
        m3u8_url_master = getattr(source, "url", "")
        # The source's own transport contract. Sent with every request that
        # belongs to this source; a source without one behaves exactly as before.
        raw_headers = getattr(source, "headers", None)
        source_headers: Dict[str, str] | None = dict(raw_headers) if raw_headers else None

        segment_cache_key = SegmentCacheKey(m3u8_url_master, str(quality))
        _segments = self.cache.get_segments(segment_cache_key)
        if _segments is not None:
            self.logger.info("Received: %s from cache!", len(_segments))
            return _segments

        # Resolve the quality-specific playlist URL (may still be a master in some edge cases)
        playlist_url = await self.get_m3u8_by_quality(
            m3u8_url=m3u8_url_master, quality=quality, headers=source_headers
        )
        self.logger.debug("Trying to fetch segments from m3u8 -> %s", playlist_url)

        # M3U8s are volatile → don't cache
        content = await self.fetch_text(
            url=playlist_url, cache_policy=CachePolicy.BYPASS, headers=source_headers
        )
        parsed = m3u8.loads(content)

        # If we accidentally got a master, pick the first media playlist (existing behavior),
        # and IMPORTANT: update base_url for urljoin to the *new* playlist URL.
        base_url = playlist_url
        if parsed.is_variant:
            self.logger.warning("Media playlist expected; got variant. Resolving to first sub-playlist...")
            media_rel = parsed.playlists[0].uri
            media_url = urljoin(playlist_url, media_rel)
            self.logger.info("Resolved to new URL: %s", media_url)
            content = await self.fetch_text(
                url=media_url, cache_policy=CachePolicy.BYPASS, headers=source_headers
            )
            parsed = m3u8.loads(content)
            base_url = media_url

        specs: List[HLSSegment] = []

        # Robust init segment handling (EXT-X-MAP)
        # Older m3u8 lib: .segment_map; newer: .init_section
        init_url = None
        init_byterange = None
        segments_map = getattr(parsed, "segment_map", None)
        if segments_map:
            assert isinstance(segments_map, list)
            try:
                init_url = urljoin(base_url, segments_map[0].uri)
                init_byterange = getattr(segments_map[0], "byterange", None)
            except Exception as exc:
                self.logger.info("Couldn't get init url, this is probably not an issue: %s", exc)
                pass
        if init_url is None:
            init_section = getattr(parsed, "init_section", None)
            if init_section and getattr(init_section, "uri", None):
                init_url = urljoin(base_url, init_section.uri)
                init_byterange = getattr(init_section, "byterange", None)

        if init_url:
            if init_byterange:
                # RFC 8216 4.3.2.5: an EXT-X-MAP BYTERANGE without @offset
                # starts at the beginning of the resource.
                length, offset = self._parse_byterange(init_byterange)
                specs.append(HLSSegment(url=init_url, length=length, offset=offset or 0))
            else:
                specs.append(HLSSegment(url=init_url))
            self.logger.debug("Found init segment: %s", init_url)

        # Build absolute URLs for all media segments, carrying byte ranges.
        # RFC 8216 4.3.2.2: EXT-X-BYTERANGE without @offset continues directly
        # after the previous media segment's sub-range of the same resource;
        # without such a predecessor the playlist is invalid.
        next_offset: Dict[str, int] = {}
        for seg in parsed.segments:
            seg_url = urljoin(base_url, seg.uri)
            byterange = getattr(seg, "byterange", None)
            if byterange:
                length, offset = self._parse_byterange(byterange)
                if offset is None:
                    if seg_url not in next_offset:
                        raise PlaylistExtractionError(
                            f"EXT-X-BYTERANGE without offset has no preceding sub-range of {seg_url}"
                        )
                    offset = next_offset[seg_url]
                specs.append(HLSSegment(url=seg_url, length=length, offset=offset))
                next_offset[seg_url] = offset + length
            else:
                specs.append(HLSSegment(url=seg_url))

        ranged = any(spec.has_range for spec in specs)
        self.logger.debug(
            "Fetched %s segments from m3u8 URL (including init if present, byte-ranged=%s)",
            len(specs), ranged,
        )
        if ranged:
            # Ranged playlists are not cached: the segment cache stores plain
            # URL strings, and a range dropped on the way through it would turn
            # every fragment back into a full-file download.
            return cast(List[Any], specs)

        segments = [spec.url for spec in specs]
        self.logger.info("Saving segments to cache....")
        self.cache.set_segments(segment_cache_key, segments)
        return cast(List[Any], segments)


    def _safe_remove(self, path: str | None) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug("Failed to remove file %s: %s", path, e)

    def _safe_rmtree(self, path: str | None) -> None:
        if not path:
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug("Failed to remove directory %s: %s", path, e)

    async def download_segment(self, url: str, timeout: int, stop_event:
                                asyncio.Event | None = None,
                                headers: Dict[str, str] | None = None,
                                byte_range: tuple[int, int] | None = None) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment, optionally a byte range of it.

        `byte_range` is (offset, length) from the playlist and becomes the
        request's `Range: bytes=<offset>-<offset+length-1>`. The header is
        request-local: it is merged with the source's own headers into a fresh
        dict, never written back into them, and never onto the session. The
        playlist-computed range wins over any Range a caller put into the
        source headers - the playlist is the contract for what this segment is.
        Returns (url, content, success).
        """
        try:
            if stop_event is not None and stop_event.is_set():
                return url, b"", False # Stopping the download here

            request_headers: Dict[str, str] | None = headers
            if byte_range is not None:
                offset, length = byte_range
                request_headers = {
                    key: value for key, value in (headers or {}).items()
                    if key.lower() != "range"
                }
                request_headers["Range"] = f"bytes={offset}-{offset + length - 1}"

            content = await self.fetch_bytes(url, timeout=timeout, headers=request_headers)

            if byte_range is not None and len(content) != byte_range[1]:
                # A server that ignores Range answers 200 with the whole
                # resource; concatenating that once per fragment would corrupt
                # the output. Wrong-sized payloads are failures, not data.
                self.logger.warning(
                    "Range request for %s returned %s bytes instead of %s (%s)",
                    url, len(content), byte_range[1], request_headers.get("Range") if request_headers else None,
                )
                return url, b"", False

            return url, content, True
        except Exception as e:
            # Log and mark failure; the caller will decide whether to retry or abort.
            self.logger.warning("Segment download failed: %s -> %s", url, e)
            return url, b"", False

    async def download(
        self,
        configuration: DownloadConfigHLS | DownloadConfigHTTP
    ) -> DownloadReport | bool:
        """Download one media source; the configuration's type picks the transport.

        `DownloadConfigHLS` takes the segmented playlist path, `DownloadConfigHTTP`
        the progressive single-stream one. The transport is never inferred from
        the URL, the file extension or the response's content type: a caller
        that built an HLS configuration gets the HLS engine even if the URL ends
        in `.mp4`, and a configuration this engine has no transport for is an
        error rather than a silent fallback onto the wrong one.

        :param configuration: the transport-specific download configuration
        :return: a DownloadReport when one was requested, else a bool
        """

        if configuration.callback is None:
            # Use a terminal text progressbar by default
            configuration.callback = Callback.text_progress_bar
            self.logger.debug("download: no callback provided, using default text progress bar")

        if isinstance(configuration, DownloadConfigHTTP):
            return await self.progressive_download(configuration)

        if not isinstance(configuration, DownloadConfigHLS):
            raise TypeError(
                "BaseCore.download() takes a DownloadConfigHLS or a DownloadConfigHTTP, "
                f"got {type(configuration).__name__}"
            )

        media_source = configuration.media_source
        if media_source is None:
            from base_api.modules.errors import MediaSourceError
            raise MediaSourceError("No media source provided.")
            
        if getattr(media_source, "source_type", None) != "HLS":
            from base_api.modules.errors import UnsupportedProtocolError
            raise UnsupportedProtocolError(f"Unsupported source type: {getattr(media_source, 'source_type', 'None')}")

        m3u8_url = getattr(media_source, "url", "")

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if m3u8_url:
            self.logger.debug("Download media_source.url=%s", m3u8_url)

        self.logger.debug("download: dispatching to threaded downloader (timeout=%s)", self.configuration.timeout)

        # 2. Call the downloader method directly
        return await self.threaded_download(
            configuration=configuration,
            pre_resolved_m3u8=m3u8_url,
            timeout=self.configuration.timeout,
            max_workers=self.configuration.max_workers_download,
        )

    # --- progressive HTTP transport -------------------------------------------
    #
    # One media file, one stream, resumable by byte offset. Deliberately not a
    # degenerate case of the HLS engine: there is no playlist to re-resolve, no
    # segment directory to reconcile, and the unit of progress is a byte rather
    # than a fragment - so the resume contract, the failure classification and
    # the cleanup rules are all different ones.

    #: The only schemes this transport fetches. Checked on the source URL and
    #: again on every redirect target.
    _PROGRESSIVE_SCHEMES: ClassVar[frozenset[str]] = frozenset({"http", "https"})
    #: A redirect chain is remote input; without a bound it is a denial of
    #: service against ourselves.
    _PROGRESSIVE_MAX_REDIRECTS: ClassVar[int] = 5
    _PROGRESSIVE_REDIRECT_STATUS: ClassVar[frozenset[int]] = frozenset({301, 302, 303, 307, 308})
    #: Worth another attempt at the same offset. 5xx is handled alongside these.
    _PROGRESSIVE_TRANSIENT_STATUS: ClassVar[frozenset[int]] = frozenset({408, 425, 429})

    @classmethod
    def _validate_progressive_url(cls, url: Any) -> str:
        """Return `url` iff this transport is allowed to fetch it.

        Applied to the source's own URL and again to every redirect target,
        because a redirect is a destination chosen by a remote party: a chain
        that starts at https and ends at a `file://` URL or at one carrying
        credentials has to be refused at the hop that introduces it.

        Private and link-local address ranges are deliberately *not* filtered
        here - that is a separate decision with its own failure modes (split
        DNS, self-hosted instances on a LAN) and is not part of this transport.
        """
        if not isinstance(url, str) or not url.strip():
            raise MediaSourceError("A progressive download needs a URL.")

        candidate = url.strip()
        try:
            parts = urlsplit(candidate)
            scheme = (parts.scheme or "").lower()
            username, password, hostname = parts.username, parts.password, parts.hostname
        except ValueError as exc:
            raise MediaSourceError(f"Unusable media URL: {exc}") from exc

        if scheme not in cls._PROGRESSIVE_SCHEMES:
            raise MediaSourceError(
                f"Refusing to fetch scheme {scheme!r}; only http and https are downloaded."
            )
        # Userinfo turns the URL itself into a credential: it would be logged,
        # written into the resume state and replayed on every retry and redirect.
        if username is not None or password is not None:
            raise MediaSourceError("Refusing a media URL that carries credentials in its userinfo.")
        if not hostname:
            raise MediaSourceError("Refusing a media URL without a host.")
        return candidate

    def _progressive_headers(
        self,
        source_headers: Dict[str, str] | None,
        offset: int,
        validators: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build the headers for one progressive attempt. Request-local, always.

        Session headers plus this source's own, exactly like every other media
        request, and then the three headers this transport owns. The source's
        dict is never written to, so two downloads running on one core cannot
        pick up each other's Range.
        """
        headers = self._merged_headers(dict(source_headers) if source_headers else None)
        # The transport owns these three; a source that carries its own is
        # overruled rather than merged with, so the request cannot end up with
        # two Ranges or with a compressed body.
        for existing in [
            key for key in headers if key.lower() in {"accept-encoding", "range", "if-range"}
        ]:
            del headers[existing]
        # Identity encoding: a compressed body makes Content-Length describe the
        # transfer instead of the resource, which would silently break both the
        # resume offset and the completeness check.
        headers["Accept-Encoding"] = "identity"

        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
            entity_tag = validators.get("etag")
            if entity_tag and not validators.get("etag_weak"):
                # Only a strong entity tag. RFC 9110 allows nothing else as a
                # range precondition, and a weak one handed to a server that
                # accepts it anyway produces a silently wrong file.
                headers["If-Range"] = entity_tag
        return headers

    def _truncate_file(self, path: str, size: int) -> bool:
        """Cut `path` back to `size` bytes. False means the file is now gone."""
        if size <= 0:
            self._safe_remove(path)
            return False
        try:
            with open(path, "r+b") as handle:
                handle.truncate(size)
            return True
        except OSError as exc:
            self.logger.warning(
                "Could not truncate %s to %s bytes (%s); discarding it.", path, size, exc
            )
            self._safe_remove(path)
            return False

    def _plan_progressive_resume(
        self, state: Mapping[str, Any] | None, tmp_path: str
    ) -> tuple[int, Dict[str, Any]]:
        """Decide where to continue, with the temporary file as the truth.

        The state records what the last run *believed* it had written; the file
        records what actually reached the disk. They can disagree - a crash
        between the two writes is the case the state file exists for - and the
        only safe reconciliation is the smaller of the two, with the file cut
        back to it so that the next byte written really is byte `offset`.
        """
        try:
            tmp_size = os.path.getsize(tmp_path)
        except OSError:
            tmp_size = 0

        if state is None:
            # A temporary file with no state carries no validators, so nothing
            # can establish that its bytes belong to the resource being fetched
            # now. Appending to it would produce a file made of two videos.
            if tmp_size:
                self.logger.info(
                    "Discarding a %s byte temporary file that has no resume state.", tmp_size
                )
            self._safe_remove(tmp_path)
            return 0, {}

        try:
            recorded = int(state.get("downloaded_bytes") or 0)
        except (TypeError, ValueError):
            recorded = 0

        offset = max(0, min(recorded, tmp_size))
        if offset != tmp_size:
            self.logger.warning(
                "Resume state records %s bytes and the temporary file holds %s; continuing at %s.",
                recorded, tmp_size, offset,
            )
            if not self._truncate_file(tmp_path, offset):
                offset = 0

        total = state.get("total_size")
        return offset, {
            "etag": state.get("etag"),
            "etag_weak": bool(state.get("etag_weak")),
            "last_modified": state.get("last_modified"),
            "total": int(total) if isinstance(total, int) else None,
        }

    @staticmethod
    def _progressive_validator_mismatch(
        known: Mapping[str, Any],
        *,
        etag: str | None,
        etag_weak: bool,
        last_modified: str | None,
        total: int | None,
    ) -> str | None:
        """Why a resumed answer is not the resource we started on, or None.

        Compared here rather than delegated to `If-Range`, because a server that
        mishandles `If-Range` answers 206 for a resource that changed - and the
        result is a file assembled from two different videos that nothing
        downstream detects. Anything that cannot be *confirmed* identical counts
        as changed: a state that had a validator and an answer that has none
        means the only evidence we had is gone.

        A weak ETag is compared for equality here even though it may never be
        sent as a range precondition. Weak equality plus an unchanged total is
        the strongest statement such a server offers; the alternative would be
        to refuse every resume against it.
        """
        known_etag = known.get("etag")
        if known_etag:
            if not etag:
                return "the response no longer carries an ETag"
            if etag != known_etag or bool(known.get("etag_weak")) != etag_weak:
                return "the ETag changed"

        known_modified = known.get("last_modified")
        if known_modified:
            if not last_modified and not known_etag:
                return "the response no longer carries Last-Modified"
            if last_modified and last_modified != known_modified:
                return "Last-Modified changed"

        known_total = known.get("total")
        if known_total is not None and total is not None and int(known_total) != int(total):
            return f"the total size changed from {known_total} to {total}"
        return None

    @staticmethod
    def _parse_content_length(value: Any) -> int | None:
        try:
            length = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return length if length >= 0 else None

    def _progressive_backoff(self, attempt: int) -> float:
        """The same bounded exponential backoff `request()` retries on."""
        settings = self.configuration
        delay = settings.request_retry_initial_delay * (
            settings.request_multiplier ** max(0, attempt - 1)
        )
        return min(delay, settings.request_retry_max_delay) + random.uniform(
            0.0, settings.request_retry_jitter
        )

    async def _sleep_unless_stopped(self, delay: float, stop_event: Any) -> bool:
        """Wait `delay` seconds; True means the stop event fired first.

        Backoff is where a cancelled download would otherwise sit for up to the
        maximum retry delay before noticing, so the wait itself is cancellable.
        Both event flavors work: `asyncio.Event.wait()` is awaited,
        `threading.Event.wait(timeout)` is offloaded to a thread.
        """
        if stop_event is None:
            if delay > 0:
                await asyncio.sleep(delay)
            return False
        if stop_event.is_set():
            return True
        if delay <= 0:
            return False

        wait_method = getattr(stop_event, "wait", None)
        if asyncio.iscoroutinefunction(wait_method):
            try:
                await asyncio.wait_for(stop_event.wait(), delay)
                return True
            except (asyncio.TimeoutError, TimeoutError):
                return False
        if wait_method is None:
            await asyncio.sleep(delay)
            return bool(stop_event.is_set())
        return bool(await asyncio.to_thread(cast(Any, wait_method), delay))

    async def _progressive_attempt(
        self,
        *,
        url: str,
        tmp_path: str,
        offset: int,
        validators: Mapping[str, Any],
        expected_size: int | None,
        chunk_size: int,
        timeout: float,
        stop_event: Any,
        source_headers: Dict[str, str] | None,
        on_progress: Callable[[int, int | None, Mapping[str, Any]], None],
    ) -> ProgressiveOutcome:
        """One request, its redirects, and as much of the body as it delivers.

        Returns a `ProgressiveOutcome` for everything the download can survive
        and raises for everything it cannot: a terminal status, an oversized
        body, a filesystem failure. The body is never materialized - it is
        streamed and written in bounded pieces, so a 4 GB file costs a chunk of
        memory rather than 4 GB of it.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        current_url = url
        redirects = 0

        while True:
            if stop_event is not None and stop_event.is_set():
                return ProgressiveOutcome(action=ProgressiveAction.CANCELLED, written=offset)

            headers = self._progressive_headers(source_headers, offset, validators)
            cookies = self._merged_cookies(None)
            # Redacted once per attempt: every message below is a place a signed
            # URL would otherwise reach a log file or a user-visible error.
            logged_url = format_url_for_log(current_url)
            await self.enforce_delay()
            self.total_requests += 1
            self.logger.debug(
                "Progressive GET %s offset=%s header_names=%s",
                logged_url, offset, sorted(headers),
            )

            async with cast(Any, session).stream(
                "GET",
                current_url,
                headers=headers,
                cookies=cookies,
                timeout=timeout,
                allow_redirects=False,
                accept_encoding="identity",
            ) as response:
                status = int(response.status_code)

                if status in self._PROGRESSIVE_REDIRECT_STATUS:
                    location = response.headers.get("Location")
                    if not location:
                        raise HTTPStatusError(
                            f"HTTP {status} without a Location header for {logged_url}",
                            status_code=status,
                            url=current_url,
                        )
                    redirects += 1
                    if redirects > self._PROGRESSIVE_MAX_REDIRECTS:
                        raise HTTPStatusError(
                            f"More than {self._PROGRESSIVE_MAX_REDIRECTS} redirects starting at "
                            f"{format_url_for_log(url)}",
                            status_code=status,
                            url=current_url,
                        )
                    # Validated again: the previous hop's approval says nothing
                    # about where this one points.
                    current_url = self._validate_progressive_url(urljoin(current_url, location))
                    self.logger.debug(
                        "Progressive redirect %s -> %s", status, format_url_for_log(current_url)
                    )
                    continue

                if status == 429:
                    retry_after = parse_retry_after(logger=self.logger, response=response)
                    return ProgressiveOutcome(
                        action=ProgressiveAction.RETRY,
                        written=offset,
                        retry_after=retry_after,
                        reason="HTTP 429 rate limited",
                    )

                if status in self._PROGRESSIVE_TRANSIENT_STATUS or 500 <= status < 600:
                    return ProgressiveOutcome(
                        action=ProgressiveAction.RETRY, written=offset, reason=f"HTTP {status}"
                    )

                if status == 416:
                    # The one status whose recovery depends on the local file: a
                    # server refusing our range while stating the resource's size
                    # may just be telling us we already have all of it.
                    stated_total = parse_unsatisfied_content_range(
                        response.headers.get("Content-Range")
                    )
                    try:
                        local_size = os.path.getsize(tmp_path)
                    except OSError:
                        local_size = 0
                    if stated_total is not None and local_size > 0 and stated_total == local_size:
                        entity_tag, tag_is_weak = normalize_etag(response.headers.get("ETag"))
                        return ProgressiveOutcome(
                            action=ProgressiveAction.ALREADY_COMPLETE,
                            written=local_size,
                            total=stated_total,
                            etag=entity_tag,
                            etag_weak=tag_is_weak,
                            last_modified=response.headers.get("Last-Modified"),
                            reason="HTTP 416 states the local file is already the whole resource",
                        )
                    return ProgressiveOutcome(
                        action=ProgressiveAction.RESTART,
                        reason=f"HTTP 416 (stated total {stated_total}, local file {local_size} bytes)",
                    )

                if status == 412:
                    # A precondition we set was rejected, so the resource is not
                    # the one the local bytes came from. Not logged through
                    # `log_precondition_failed`: that helper previews the body,
                    # and this response's body is a stream we must not consume.
                    return ProgressiveOutcome(
                        action=ProgressiveAction.RESTART, reason="HTTP 412 precondition failed"
                    )

                if status in {401, 403}:
                    raise AccessDeniedError(
                        f"Request blocked by server (HTTP {status}) for {logged_url}"
                    )
                if status == 410:
                    raise ResourceGone(f"Resource gone (HTTP 410) for URL: {logged_url}")
                if status not in {200, 206}:
                    raise HTTPStatusError(
                        f"HTTP {status} for {logged_url}", status_code=status, url=logged_url
                    )

                entity_tag, tag_is_weak = normalize_etag(response.headers.get("ETag"))
                last_modified = response.headers.get("Last-Modified")
                meta: Dict[str, Any] = {
                    "etag": entity_tag,
                    "etag_weak": tag_is_weak,
                    "last_modified": last_modified,
                }

                if status == 206:
                    if offset <= 0:
                        # We asked for the whole resource. A partial answer has
                        # no place we could put it.
                        return ProgressiveOutcome(
                            action=ProgressiveAction.RESTART,
                            reason="HTTP 206 for a request that carried no Range",
                        )
                    parsed = parse_content_range(response.headers.get("Content-Range"))
                    if parsed is None:
                        return ProgressiveOutcome(
                            action=ProgressiveAction.RESTART,
                            reason=(
                                "HTTP 206 without a parsable Content-Range "
                                f"({response.headers.get('Content-Range')!r})"
                            ),
                        )
                    first, _last, complete = parsed
                    if first != offset:
                        return ProgressiveOutcome(
                            action=ProgressiveAction.RESTART,
                            reason=f"HTTP 206 starts at byte {first}, expected {offset}",
                        )
                    mismatch = self._progressive_validator_mismatch(
                        validators,
                        etag=entity_tag,
                        etag_weak=tag_is_weak,
                        last_modified=last_modified,
                        total=complete,
                    )
                    if mismatch is not None:
                        return ProgressiveOutcome(action=ProgressiveAction.RESTART, reason=mismatch)
                    write_offset = offset
                    total = complete if complete is not None else expected_size
                else:
                    # HTTP 200: either we never asked for a range, or the server
                    # ignored the one we sent. Both mean this body is the whole
                    # resource, so the file is rewritten from byte zero -
                    # appending would splice a second copy onto the first.
                    if offset > 0:
                        self.logger.warning(
                            "Server answered 200 to a Range request for %s; "
                            "rewriting the file from byte zero.",
                            current_url,
                        )
                    write_offset = 0
                    body_length = self._parse_content_length(response.headers.get("Content-Length"))
                    total = body_length if body_length is not None else expected_size

                if write_offset and not os.path.exists(tmp_path):
                    # Nothing to append to after all; the resource is fetched whole.
                    self.logger.warning(
                        "The temporary file for %s vanished before the body arrived; "
                        "writing from byte zero.",
                        current_url,
                    )
                    write_offset = 0

                if (
                    total is not None
                    and expected_size is not None
                    and int(total) != int(expected_size)
                ):
                    self.logger.warning(
                        "The provider stated %s bytes for %s but the response states %s; "
                        "the response wins for this body.",
                        expected_size, logged_url, total,
                    )

                written = write_offset
                on_progress(written, total, meta)

                handle = open(tmp_path, "wb" if write_offset == 0 else "r+b")
                try:
                    if write_offset:
                        handle.seek(write_offset)
                        handle.truncate()

                    buffer = bytearray()

                    async def flush(piece: bytes) -> None:
                        nonlocal written
                        await asyncio.to_thread(handle.write, piece)
                        written += len(piece)
                        if total is not None and written > total:
                            raise OversizedBody(
                                f"{logged_url} sent more than the {total} bytes it stated",
                                written=written,
                                expected=int(total),
                            )
                        on_progress(written, total, meta)

                    try:
                        async for raw_chunk in response.aiter_content():
                            if stop_event is not None and stop_event.is_set():
                                return ProgressiveOutcome(
                                    action=ProgressiveAction.CANCELLED,
                                    written=written,
                                    total=total,
                                    **meta,
                                )
                            if raw_chunk:
                                buffer += raw_chunk
                            # curl decides how much it hands us; the write size,
                            # the progress granularity and the oversize check are
                            # ours, so the stream is re-cut to `chunk_size`.
                            while len(buffer) >= chunk_size:
                                # Checked per written chunk, not only per chunk
                                # curl delivers: one curl chunk can be the whole
                                # small file, and a stop must not have to wait
                                # for the next network read to be noticed.
                                if stop_event is not None and stop_event.is_set():
                                    return ProgressiveOutcome(
                                        action=ProgressiveAction.CANCELLED,
                                        written=written,
                                        total=total,
                                        **meta,
                                    )
                                piece = bytes(buffer[:chunk_size])
                                del buffer[:chunk_size]
                                await flush(piece)
                        if buffer:
                            if stop_event is not None and stop_event.is_set():
                                return ProgressiveOutcome(
                                    action=ProgressiveAction.CANCELLED,
                                    written=written,
                                    total=total,
                                    **meta,
                                )
                            await flush(bytes(buffer))
                    except (RequestsError, asyncio.TimeoutError, TimeoutError) as exc:
                        # The connection died mid-body. Everything already
                        # written is a valid prefix, so this is a retry at the
                        # new offset - and the validators travel back with it so
                        # the resume can prove those bytes still belong together.
                        self.logger.warning(
                            "Progressive stream for %s broke after %s bytes: %s",
                            current_url, written, exc,
                        )
                        if buffer:
                            # Bytes that arrived before the break are still a
                            # valid prefix; discarding them would make the resume
                            # re-fetch up to one chunk for no reason.
                            await flush(bytes(buffer))
                        return ProgressiveOutcome(
                            action=ProgressiveAction.RETRY,
                            written=written,
                            total=total,
                            reason=f"{type(exc).__name__}: {exc}",
                            **meta,
                        )
                finally:
                    handle.close()

                return ProgressiveOutcome(
                    action=ProgressiveAction.COMPLETED, written=written, total=total, **meta
                )

    async def progressive_download(self, configuration: DownloadConfigHTTP) -> bool:
        """Download one progressive media file as a single resumable stream.

        Sequential by design: one connection, one file, appended to. Parallel
        multipart would need every part's server to agree on ranges *and* on the
        resource staying byte-identical for the whole download, which is exactly
        the assumption the resume contract below refuses to make.

        Returns True when the file is complete and in place, False when the stop
        event ended it. Every other failure raises a typed transport error the
        caller can act on, rather than a bare False that says only "something".
        """
        media_source = configuration.media_source
        if media_source is None:
            raise MediaSourceError("No media source provided.")
        if getattr(media_source, "source_type", None) != "HTTP":
            raise UnsupportedProtocolError(
                "The progressive transport downloads HTTP sources, got "
                f"{getattr(media_source, 'source_type', 'None')!r}"
            )

        source_url = self._validate_progressive_url(getattr(media_source, "url", None))
        # The target file name is the caller's, always. Neither the URL's last
        # path segment nor a Content-Disposition ever names a file here: both are
        # remote input, and a download must not be able to choose where on the
        # filesystem it lands.
        target = str(configuration.path)
        tmp_path = f"{target}.tmp"
        state_path = configuration.state_path
        callback = configuration.callback
        stop_event = configuration.stop_event
        chunk_size = max(1, int(configuration.chunk_size))
        flush_bytes = max(chunk_size, int(configuration.state_flush_bytes))
        timeout = float(
            configuration.read_timeout
            if configuration.read_timeout is not None
            else self.configuration.timeout
        )
        max_attempts = max(
            1,
            int(
                configuration.max_attempts
                if configuration.max_attempts is not None
                else self.configuration.request_attempts
            ),
        )
        # One redaction for the whole download: the URL below is only ever
        # shown, never fetched, so nothing downstream needs the real query.
        logged_url = format_url_for_log(source_url)
        # What the resume state is keyed on. The provider's own name for the
        # track when it has one, the URL when it does not.
        identity = getattr(media_source, "identity", None)
        resume_key = identity or source_url
        raw_headers = getattr(media_source, "headers", None)
        source_headers: Dict[str, str] = dict(raw_headers) if raw_headers else {}
        expected_size = configuration.expected_size
        if expected_size is None:
            expected_size = getattr(media_source, "expected_size", None)

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.logger.info(
            "Progressive download start: path=%s state_path=%s expected_size=%s chunk_size=%s "
            "max_attempts=%s timeout=%s header_names=%s",
            target, state_path, expected_size, chunk_size, max_attempts, timeout,
            sorted(source_headers),
        )

        state = load_progressive_state(state_path) if state_path else None
        if state is not None and (state.get("identity") or state.get("url")) != resume_key:
            # Not the resource these bytes belong to, whatever the file holds.
            #
            # Which key answers that depends on what the provider gave us. A URL
            # is an honest identity only while it stays stable: a signed one
            # expires within hours, and re-resolving the same track yields a
            # different URL for byte-identical content, so comparing URLs throws
            # away every resume across a restart. An identity, where a provider
            # supplies one, says "same track" straight through that change.
            # Where none is supplied, nothing here behaves differently.
            self.logger.info(
                "Resume state at %s was written for another resource; starting fresh.",
                state_path,
            )
            self._safe_remove(tmp_path)
            self._safe_remove(state_path)
            state = None

        offset, validators = self._plan_progressive_resume(state, tmp_path)
        if expected_size is None and validators.get("total") is not None:
            expected_size = int(cast(int, validators["total"]))

        state_created_at: str | None = cast(Any, state.get("created_at")) if state else None
        last_persisted = offset

        def persist(written: int, total: int | None, meta: Mapping[str, Any]) -> None:
            """Write the resume state. Always after the bytes, never before.

            A state ahead of the file would claim progress the file cannot back
            up; a state behind it only costs the difference on the next run,
            because the resume takes the smaller of the two.
            """
            nonlocal state_created_at, last_persisted
            if not state_path:
                return
            record = build_progressive_state(
                url=source_url,
                identity=identity,
                output_path=target,
                temp_path=tmp_path,
                total_size=total,
                downloaded_bytes=written,
                etag=cast(Any, meta.get("etag")),
                etag_weak=bool(meta.get("etag_weak")),
                last_modified=cast(Any, meta.get("last_modified")),
                created_at=state_created_at,
            )
            try:
                write_progressive_state(state_path, record)
            except OSError as exc:
                self.logger.warning("Could not persist resume state %s: %s", state_path, exc)
                return
            state_created_at = cast(str, record.created_at)
            last_persisted = written

        def on_progress(written: int, total: int | None, meta: Mapping[str, Any]) -> None:
            if callback is not None:
                # An unknown total is reported as 0 rather than guessed at: a
                # progress bar that invents a denominator lies about the end.
                callback(written, int(total) if total else 0)
            if state_path and written - last_persisted >= flush_bytes:
                persist(written, total, meta)

        def replan(written: int, total: int | None, meta: Mapping[str, Any]) -> None:
            nonlocal offset, validators, expected_size
            if total is not None:
                expected_size = int(total)
            offset, validators = self._plan_progressive_resume(
                {
                    "downloaded_bytes": written,
                    "etag": meta.get("etag"),
                    "etag_weak": meta.get("etag_weak"),
                    "last_modified": meta.get("last_modified"),
                    "total_size": total,
                },
                tmp_path,
            )

        def cancel(written: int, total: int | None, meta: Mapping[str, Any]) -> bool:
            """An explicit stop. The partial file goes, unless asked otherwise."""
            self.logger.warning(
                "Progressive download cancelled after %s bytes (cleanup_on_stop=%s).",
                written, configuration.cleanup_on_stop,
            )
            if configuration.cleanup_on_stop:
                self._safe_remove(tmp_path)
                self._safe_remove(state_path)
            else:
                persist(written, total, meta)
            return False

        restarts = 0
        attempts = 0
        last_error: Exception | None = None

        while True:
            if stop_event is not None and stop_event.is_set():
                return cancel(offset, expected_size, validators)

            try:
                outcome = await self._progressive_attempt(
                    url=source_url,
                    tmp_path=tmp_path,
                    offset=offset,
                    validators=validators,
                    expected_size=expected_size,
                    chunk_size=chunk_size,
                    timeout=timeout,
                    stop_event=stop_event,
                    source_headers=source_headers,
                    on_progress=on_progress,
                )
            except (RequestsError, asyncio.TimeoutError, TimeoutError) as exc:
                # A failure before any response header arrived: nothing new was
                # learned about the resource, so the current plan stands.
                last_error = exc
                outcome = ProgressiveOutcome(
                    action=ProgressiveAction.RETRY,
                    written=offset,
                    total=expected_size,
                    reason=f"{type(exc).__name__}: {exc}",
                    etag=cast(Any, validators.get("etag")),
                    etag_weak=bool(validators.get("etag_weak")),
                    last_modified=cast(Any, validators.get("last_modified")),
                )
            except OversizedBody:
                # The bytes on disk contain data that is not the resource as
                # described; unlike a short body this cannot be resumed from.
                self._safe_remove(tmp_path)
                self._safe_remove(state_path)
                raise

            if outcome.action is ProgressiveAction.CANCELLED:
                return cancel(outcome.written, outcome.total, outcome.validators)

            if outcome.action is ProgressiveAction.RETRY:
                if outcome.written > last_persisted:
                    persist(outcome.written, outcome.total, outcome.validators)
                attempts += 1
                if attempts >= max_attempts:
                    error = last_error or NetworkRequestError(outcome.reason or "transient failure")
                    self.logger.error(
                        "Progressive download of %s failed after %s attempts.", logged_url, attempts
                    )
                    raise RequestRetriesExhausted(logged_url, attempts, error)
                delay = (
                    outcome.retry_after
                    if outcome.retry_after is not None
                    else self._progressive_backoff(attempts)
                )
                self.logger.warning(
                    "Progressive attempt %s/%s for %s failed (%s); retrying in %.2fs.",
                    attempts, max_attempts, logged_url, outcome.reason, delay,
                )
                if await self._sleep_unless_stopped(delay, stop_event):
                    return cancel(outcome.written, outcome.total, outcome.validators)
                replan(outcome.written, outcome.total, outcome.validators)
                continue

            if outcome.action is ProgressiveAction.RESTART:
                if restarts >= 1:
                    # One automatic restart, never two: a server that keeps
                    # contradicting itself has to surface as an error rather than
                    # re-downloading the same file forever.
                    raise ResumeConflict(
                        f"Resuming {logged_url} failed again after a fresh start: {outcome.reason}"
                    )
                restarts += 1
                self.logger.warning("Restarting %s at byte zero: %s", logged_url, outcome.reason)
                self._safe_remove(tmp_path)
                self._safe_remove(state_path)
                offset, validators = 0, {}
                last_persisted = 0
                state_created_at = None
                continue

            # COMPLETED or ALREADY_COMPLETE. The file on disk decides, not the
            # byte counter: the counter is what we think we wrote, the size is
            # what is actually there to move into place.
            total = outcome.total
            try:
                actual = os.path.getsize(tmp_path)
            except OSError as exc:
                raise UnknownError(
                    f"The temporary file {tmp_path} disappeared before it could be finalized: {exc}"
                ) from exc

            if total is not None and actual < total:
                # A valid prefix: keep it and its state so a later run continues.
                persist(actual, total, outcome.validators)
                raise IncompleteBody(
                    f"{logged_url} delivered {actual} of {total} bytes",
                    written=actual,
                    expected=int(total),
                )
            if total is not None and actual > total:
                self._safe_remove(tmp_path)
                self._safe_remove(state_path)
                raise OversizedBody(
                    f"{logged_url} delivered {actual} bytes but stated {total}",
                    written=actual,
                    expected=int(total),
                )

            # No PyAV, no remux, no container inspection: a progressive MP4 is
            # already the file the user asked for, and opening it would only add
            # a way to fail after the bytes are correct.
            self._replace_with_retry(tmp_path, target)
            self._safe_remove(state_path)
            if callback is not None:
                # The total is known now even when the server never stated one:
                # it is exactly what arrived.
                callback(actual, int(total) if total is not None else actual)
            self.logger.info(
                "Progressive download completed: path=%s bytes=%s restarts=%s retries=%s",
                target, actual, restarts, attempts,
            )
            return True

    async def threaded_download(
        self: "BaseCore",
        timeout: int,
        max_workers: int,
        pre_resolved_m3u8: str,
        configuration: DownloadConfigHLS,
    ) -> DownloadReport | bool:
        """
        Threaded HLS segment downloader with optional resume state and stop flag.
        """
        try:
            cleanup_on_stop = configuration.cleanup_on_stop
            keep_segment_dir = configuration.keep_segment_dir
            quality = configuration.quality
            path = configuration.path
            remux = configuration.remux
            start_segment = configuration.start_segment
            segment_state_path = configuration.segment_state_path
            segment_dir = configuration.segment_dir
            return_report = configuration.return_report
            callback = configuration.callback
            callback_remux = configuration.callback_remux
            stop_event = configuration.stop_event
            ios_support = configuration.ios_support
            timeout = timeout
            pre_resolved_m3u8_url = pre_resolved_m3u8
            # Per-source transport headers, applied to every request this
            # download makes (playlists and segments alike, fresh or resumed).
            # Names only in the log - header values are not ours to print.
            raw_source_headers = getattr(configuration.media_source, "headers", None)
            source_headers: Dict[str, str] | None = (
                dict(raw_source_headers) if raw_source_headers else None
            )
            if source_headers:
                self.logger.debug(
                    "Applying %d source-specific request header(s): %s",
                    len(source_headers), sorted(source_headers),
                )

            self.logger.info(
                f"Threaded download start: quality={quality} path={path} remux={remux} start_segment={start_segment} "
                f"segment_state_path={segment_state_path} segment_dir={segment_dir} return_report={return_report} "
                f"cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir} max_workers={max_workers} "
                f"timeout={timeout} stop_event_set={bool(stop_event and stop_event.is_set())}"
            )
            self.logger.debug(
                f"Threaded download callbacks: callback_set={bool(callback)} callback_remux_set={bool(callback_remux)}"
            )
            resume_state = None
            resume_mode = False
            created_at = None

            # Help type checker with initial types
            if segment_state_path:
                if os.path.exists(segment_state_path):
                    self.logger.info(f"Found segment state file: {segment_state_path}. Attempting resume.")
                else:
                    self.logger.debug(f"No segment state file found at: {segment_state_path}. Starting fresh.")

            if segment_state_path and os.path.exists(segment_state_path):
                try: # This starts resuming from previous download
                    resume_state = load_segment_state(segment_state_path)
                    resume_mode = True
                except Exception as e: # Shouldn't happen, but if it does, we just do a new download
                    self.logger.warning(f"Failed to load segment state {segment_state_path}: {e}. Starting fresh.")
                    resume_state = None
                    resume_mode = False

            if resume_mode:
                assert resume_state is not None
                loaded_segments = resume_state.get("segments") or []
                if (
                    int(resume_state.get("version") or 1) < 2
                    and loaded_segments
                    and all(isinstance(entry, str) for entry in loaded_segments)
                    and len(set(loaded_segments)) < len(loaded_segments)
                ):
                    # A version-1 state cannot express byte ranges, so duplicate
                    # URLs in one mean it was written for a byte-range playlist
                    # by an engine without range support. Resuming it would
                    # fetch the full resource once per fragment - start fresh.
                    self.logger.warning(
                        "Segment state %s predates byte-range support and repeats URLs; discarding it and starting fresh.",
                        segment_state_path,
                    )
                    resume_mode = False
                    resume_state = None

            if resume_mode:
                assert resume_state is not None
                segments = resume_state.get("segments") or []  # This fetches the list of segments from the resume state
                if not segments:
                    raise UnknownError("Segment state is invalid or empty.") # Shouldn't happen ;)

                segment_dir = resume_state.get("segment_dir") or segment_dir
                if not segment_dir:
                    raise UnknownError("Segment state is missing segment_dir.")

                created_at = resume_state.get("created_at")
                width = int(resume_state.get("segment_index_width") or get_segment_index_width(len(segments)))
                state_start = int(resume_state.get("start_segment", 0) or 0) # Where we start segments

                """
            Because every segment has a different binary offset, we can't just inject specific segments into specific
            parts of the file. That's why I can only start after xx successful segments.

            So, let's say 0-12 segments were successful, but 13 was not and from 14-17 everything went smooth.
            In this case, I need to start from 13 and STILL override 14-17.
                """

                if start_segment and state_start != start_segment:
                    self.logger.warning(
                        f"start_segment={start_segment} ignored; resuming from state start_segment={state_start}."
                    )

                start_segment = state_start
                m3u8_url = resume_state.get("m3u8_url") or ""
                state_quality = resume_state.get("quality", quality)
                self.logger.info(
                    f"Resume state loaded: segments={len(segments)} start_segment={start_segment} "
                    f"segment_dir={segment_dir} segment_index_width={width} created_at={created_at} "
                    f"quality={state_quality} m3u8_url={m3u8_url}"
                )

            else:
                m3u8_master = pre_resolved_m3u8_url
                self.logger.info(f"Fetching segments for quality={quality} media_source.url={m3u8_master}")
                
                # Mocking a source temporarily for threaded_download compatibility with get_segments
                class _TempSource:
                    source_type = "HLS"
                    url = m3u8_master
                    headers = source_headers or {}

                segments = await self.get_segments(quality=quality, source=_TempSource())
                total_before = len(segments)
                if start_segment > 0:
                    self.logger.debug(
                        f"Applying start_segment offset: {start_segment} (from total={total_before})"
                    )
                    segments = segments[start_segment:]
                if segment_state_path and segment_dir is None:
                    segment_dir = f"{path}.segments"
                    self.logger.debug(f"segment_dir set from state path: {segment_dir}")
                width = get_segment_index_width(len(segments)) if segment_dir else 0
                m3u8_url = m3u8_master
                state_quality = quality
                self.logger.info(
                    f"Segments ready: count={len(segments)} segment_dir={segment_dir} "
                    f"segment_index_width={width} m3u8_url={m3u8_url}"
                )

            # One internal representation for both worlds: plain URLs (str),
            # parsed playlist entries (HLSSegment) and resumed state entries
            # (JSON dicts) all become HLSSegment before anything downloads.
            def _as_spec(entry: Any) -> HLSSegment:
                if isinstance(entry, HLSSegment):
                    return entry
                if isinstance(entry, Mapping):
                    return HLSSegment(
                        url=str(entry.get("url", "")),
                        length=entry.get("length"),
                        offset=entry.get("offset"),
                    )
                return HLSSegment(url=str(entry))

            segment_specs = [_as_spec(entry) for entry in segments]
            ranged_mode = any(spec.has_range for spec in segment_specs)
            # What the state file stores. Version 1 keeps the plain-string
            # layout every existing state uses; ranged playlists persist
            # url+range per entry and bump the version, so a range entry can
            # never be mistaken for a URL by mistake.
            if ranged_mode:
                state_version = 2
                state_segments: List[Any] = [
                    {"url": spec.url, "length": spec.length, "offset": spec.offset}
                    if spec.has_range else {"url": spec.url}
                    for spec in segment_specs
                ]
                self.logger.info(
                    "Byte-range playlist: %d of %d segments are sub-ranges.",
                    sum(1 for spec in segment_specs if spec.has_range), len(segment_specs),
                )
            else:
                state_version = 1
                state_segments = [spec.url for spec in segment_specs]

            n = len(segment_specs) # Total amount of segments
            if n == 0:
                raise UnknownError("No segments found for this playlist.")
                # Shouldn't happen

            if segment_dir:
                os.makedirs(segment_dir, exist_ok=True) # Creates the segment directory for later resuming
                self.logger.debug(f"Segment directory ready: {segment_dir}")
            self.logger.info(f"Segment plan: total={n} segment_dir={segment_dir}")

            downloaded = [False] * n # Keeps track of total downloaded segments

            """
            We write a list with [False, False, n] where n is the value of the total amount of segments.
            This creates a lit with as many False entries as segments. Since `self.download_segment` returns a bool
            along with the data, we can use that to keep track, since we just change the bool to True for every 
            downloaded segments.
            """

            if segment_dir: # Tries to find existing segments that we already downloaded
                existing_segments = 0
                for i in range(n): # Does that for every segment
                    seg_path = segment_file_path(segment_dir, i, width) # Gets the file path
                    try:
                        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                            # if it exists, we treat it as already downloaded (makes sense)
                            downloaded[i] = True
                            existing_segments += 1
                    except Exception as exc:
                        self.logger.warning(f"Couldn't download segment: {i}, retrying later.  ->: {exc}")
                        # If something goes wrong, we treat it as not downloaded and re-fetch it later
                        downloaded[i] = False
                self.logger.info(
                    f"Existing segments detected: {existing_segments}/{n} in {segment_dir}"
                )

            progressed = sum(downloaded) # Amount of already downloaded segments
            downloaded_count = progressed
            if progressed and callback: # Does an initial callback, so that Porn Fetch can start showing the user how
                # many segments have already been downloaded
                callback(progressed, n)
            if progressed:
                self.logger.info(f"Resume progress: already_downloaded={progressed}/{n}")

            target_indices = [i for i in range(n) if not downloaded[i]] # The segments we still need to fetch
            self.logger.info(f"Target segments to download: {len(target_indices)}/{n}")

            tmp_path = f"{path}.tmp" # Creates a temporary path where we write stuff to
            cancelled = False # This is the cancellation event that stops the download
            max_seg_retries = 2 # Maximum retries to get segments
            progress_log_step = max(1, n // 20)
            next_progress_log = ((progressed // progress_log_step) + 1) * progress_log_step

            if stop_event is not None and stop_event.is_set():
                cancelled = True
                target_indices = [] # Empty list stops the download :)
                self.logger.warning("Stop event already set; cancelling before scheduling segments.")

            if target_indices:
                workers = max(1, min(max_workers, len(target_indices)))
                parts: List[bytes | None] | None = None
                next_to_write = 0
                out_fp = None
                self.logger.info(
                    f"Starting segment download pool: workers={workers} targets={len(target_indices)}"
                )

                if not segment_dir:
                    parts = [None] * n
                    out_fp = cast(Any, open(tmp_path, "wb"))
                    self.logger.debug(f"Using in-memory segment assembly. tmp_path={tmp_path}")
                else:
                    self.logger.debug(f"Writing segments to disk. segment_dir={segment_dir} tmp_path={tmp_path}")

                segment_tasks: set[asyncio.Task[Tuple[int, bool, bytes]]] = set()
                stop_waiter: asyncio.Task[bool] | None = None
                try:
                    # Use asyncio.gather to fetch segments concurrently instead of ThreadPoolExecutor

                    # Create a semaphore to limit concurrent requests
                    semaphore = asyncio.Semaphore(workers)

                    async def fetch_segment_with_semaphore(idx: int, spec: HLSSegment) -> Tuple[int, bool, bytes]:
                        url = spec.url
                        byte_range = (spec.offset or 0, spec.length) if spec.has_range else None
                        async with semaphore:
                            if stop_event is not None and stop_event.is_set():
                                return idx, False, b""

                            # Handle retries inside the coroutine. Every attempt
                            # repeats the identical URL, Range and source headers.
                            for attempt in range(max_seg_retries + 1):
                                if stop_event is not None and stop_event.is_set():
                                    return idx, False, b""

                                try:
                                    _, segment_data, is_success = await self.download_segment(
                                        url, timeout, stop_event, headers=source_headers,
                                        byte_range=byte_range,
                                    )
                                    if is_success and segment_data:
                                        return idx, True, segment_data
                                except Exception as exception:
                                    self.logger.error(f"Worker exception for segment {idx}: {exception}", exc_info=True)

                                if attempt < max_seg_retries:
                                    self.logger.warning(
                                        f"Segment {idx} failed; retrying {attempt + 1}/{max_seg_retries}"
                                    )
                                    # Optional short backoff delay could go here
                                else:
                                    self.logger.error(
                                        f"Segment {idx} failed after {attempt} retries."
                                    )
                            return idx, False, b""

                    segment_tasks = {
                        asyncio.create_task(
                            fetch_segment_with_semaphore(i, segment_specs[i]),
                            name=f"hls-segment-{i}",
                        )
                        for i in target_indices
                    }
                    if stop_event is not None:
                        if hasattr(stop_event, "is_set") and not asyncio.iscoroutinefunction(getattr(stop_event, "wait", None)):
                            # Handle threading.Event by offloading to thread
                            stop_waiter = asyncio.create_task(asyncio.to_thread(stop_event.wait), name="hls-stop-waiter")
                        else:
                            # Handle asyncio.Event natively
                            stop_waiter = asyncio.create_task(stop_event.wait(), name="hls-stop-waiter")
                    else:
                        stop_waiter = None

                    while segment_tasks:
                        waiters = set(segment_tasks)
                        if stop_waiter is not None:
                            waiters.add(stop_waiter)
                        done, _ = await asyncio.wait(
                            waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        if stop_waiter is not None and stop_waiter in done:
                            cancelled = True
                            for task in segment_tasks:
                                task.cancel()
                            await asyncio.gather(*segment_tasks, return_exceptions=True)
                            segment_tasks.clear()
                            self.logger.info("Cancelled all in-flight HLS segment requests.")
                            break

                        completed_tasks = done.intersection(segment_tasks)
                        for task in completed_tasks:
                            segment_tasks.remove(task)
                            i, success, data = task.result()

                            if success and data:
                                downloaded[i] = True # Successfully got segment, mark it as done
                                downloaded_count += 1
                                if segment_dir:
                                    # Write to a temp path (good for resuming, but not I/O efficient)
                                    seg_path = segment_file_path(segment_dir, i, width)
                                    tmp_seg = f"{seg_path}.part"
                                    # Offload segment file writing to a thread
                                    def write_part(ts_path: str, t_data: bytes) -> None:
                                        with open(ts_path, "wb") as f:
                                            f.write(t_data)
                                    await asyncio.to_thread(write_part, tmp_seg, data)
                                    os.replace(tmp_seg, seg_path)
                                else:
                                    assert parts is not None
                                    parts[i] = data # Keep in memory (I/O efficient)

                                progressed += 1 # Fetched +1 segment, so we give back callback
                                if callback:
                                    callback(progressed, n)
                                if progressed >= next_progress_log or progressed == n:
                                    remaining = n - downloaded_count
                                    self.logger.debug(
                                        f"Segment progress: processed={progressed}/{n} "
                                        f"downloaded={downloaded_count} remaining={remaining}"
                                    )
                                    next_progress_log += progress_log_step

                            else:
                                # Handling failure (already retried in fetch_segment_with_semaphore)
                                progressed += 1
                                if callback:
                                    callback(progressed, n)
                                if progressed >= next_progress_log or progressed == n:
                                    remaining = n - downloaded_count
                                    self.logger.debug(
                                        f"Segment progress: processed={progressed}/{n} "
                                        f"downloaded={downloaded_count} remaining={remaining}"
                                    )
                                    next_progress_log += progress_log_step

                            if not segment_dir and parts is not None:
                                chunks_to_write = []
                                while next_to_write < n and parts[next_to_write] is not None:
                                    if parts[next_to_write]:
                                        chunks_to_write.append(parts[next_to_write])
                                    next_to_write += 1
                                if chunks_to_write:
                                    # Write memory chunks to thread to prevent IO block
                                    def write_chunks(fp: Any, list_of_data: List[bytes]) -> None:
                                        for c_data in list_of_data:
                                            fp.write(c_data)
                                    await asyncio.to_thread(write_chunks, cast(Any, out_fp), chunks_to_write)

                finally:
                    if stop_waiter is not None:
                        stop_waiter.cancel()
                        await asyncio.gather(stop_waiter, return_exceptions=True)
                    if segment_tasks:
                        for task in segment_tasks:
                            task.cancel()
                        await asyncio.gather(*segment_tasks, return_exceptions=True)
                    if out_fp is not None:
                        out_fp.close()

            if stop_event is not None and stop_event.is_set():
                cancelled = True

            missing = [i for i, ok in enumerate(downloaded) if not ok] # Missing segments
            missing_urls = [segment_specs[i].url for i in missing] # Missing URLs of segments
            self.logger.info(
                "Segment download finished: downloaded=%s/%s missing=%s cancelled=%s",
            downloaded_count, n, len(missing), cancelled)
            if missing:
                sample = missing[:10]
                self.logger.error(
                    "Missing segments detected: count=%s sample=%s", len(missing), sample
                )

            report = DownloadReport(
                status= "cancelled" if cancelled else ("failed" if missing else "completed"),
                total=n,
                downloaded= n - len(missing),
                missing=missing,
                missing_urls=missing_urls,
                segment_dir=segment_dir,
                segment_state_path=segment_state_path,
                start_segment=start_segment,
                quality=quality

            )

            if cancelled: # If user cancels, we clean up stuff
                self.logger.warning(
                    f"Download cancelled. cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir}"
                )
                if cleanup_on_stop:
                    self._safe_remove(tmp_path)
                    if segment_dir and not keep_segment_dir:
                        self._safe_rmtree(segment_dir)

                if segment_state_path:
                    # This is the segment state that is saved as a file, this is NOT the returned report!
                    assert  isinstance(segment_state_path, str)
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = build_segment_state(
                        segments=state_segments,
                        version=state_version,
                        missing=missing,
                        segment_dir=segment_dir,
                        segment_index_width=width if segment_dir else 0,
                        path=path,
                        quality=str(state_quality),
                        start_segment=start_segment,
                        m3u8_url=m3u8_url,
                        created_at=created_at,
                    )
                    write_segment_state(segment_state_path, state)

                if return_report:
                    missing = report.missing
                    self.logger.debug(
                        f"Returning cancelled report: downloaded={report.downloaded} missing={len(missing)}"
                    )
                    return report
                return False

            if missing:
                self.logger.error(
                    f"Download incomplete: {len(missing)} segments missing. Writing state={bool(segment_state_path)}"
                )
                self._safe_remove(tmp_path)
                if segment_state_path:
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = build_segment_state(
                        segments=state_segments,
                        version=state_version,
                        missing=missing,
                        segment_dir=segment_dir,
                        segment_index_width=width if segment_dir else 0,
                        path=path,
                        quality=str(state_quality),
                        start_segment=start_segment,
                        m3u8_url=m3u8_url,
                        created_at=created_at,
                    )
                    write_segment_state(segment_state_path, state)
                if return_report:
                    self.logger.debug(
                        f"Returning failed report: downloaded={report.downloaded} missing={len(report.missing)}"
                    )
                    return report
                return False

            if segment_dir:
                self.logger.info(
                    f"Assembling {n} segments from {segment_dir} into {tmp_path}"
                )
                def assemble_segments() -> List[int]:
                    with open(tmp_path, "wb") as out_file_path:
                        for idx in range(n):
                            segment_path = segment_file_path(segment_dir, idx, width)
                            if not os.path.exists(segment_path):
                                return [idx]
                            with open(segment_path, "rb") as seg_fp:
                                shutil.copyfileobj(seg_fp, out_file_path, length=1024 * 1024) # type: ignore[arg-type]
                    return []

                # Offload heavy IO segment assembly
                missing_assemble = await asyncio.to_thread(assemble_segments)
                if missing_assemble:
                    missing = missing_assemble

                if missing:
                    self.logger.error(
                        f"Missing segment file during assemble: index={missing[0]} segment_dir={segment_dir}"
                    )
                    self._safe_remove(tmp_path)
                    if segment_state_path:
                        self.logger.info(f"Writing segment state to: {segment_state_path}")
                        state = build_segment_state(
                            segments=state_segments,
                            version=state_version,
                            missing=missing,
                            segment_dir=segment_dir,
                            segment_index_width=width if segment_dir else 0,
                            path=path,
                            quality=str(state_quality),
                            start_segment=start_segment,
                            m3u8_url=m3u8_url,
                            created_at=created_at,
                        )
                        write_segment_state(segment_state_path, state)
                    report.status = "failed"
                    report.missing = missing
                    report.missing_urls = [segments[i] for i in missing]
                    if return_report:
                        self.logger.debug(
                            f"Returning failed report after assemble: downloaded={report.downloaded} "
                            f"missing={len(report.missing)}"
                        )
                        return report
                    return False

            if remux:
                self.logger.info(f"Remuxing TS to MP4: input={tmp_path} output={path}")
                # Offload heavy CPU/IO bound task
                try:
                    await asyncio.to_thread(self._convert_ts_to_mp4, tmp_path, path, callback_remux, ios_support)
                except Exception:
                    self._safe_remove(path)
                    raise
                # This is important, because not all players can play MPEG-TS AND I want to write
                # metadata to the files, and this doesn't work without a container.
                self._safe_remove(tmp_path)
                self.logger.info(f"Remux completed: output={path}")

            else:
                self.logger.debug("Remux disabled; moving temporary file into place.")
                try:
                    os.replace(tmp_path, path) # If we don't remux, we just rename it to mp4 and treat it as done :)
                except Exception as exc: # Shouldn't happen and I also don't know what this does lol
                    self.logger.warning(f"os.replace failed: {exc}, falling back to manual copy.")
                    def manual_copy() -> None:
                        with open(path, "wb") as final_fp, open(tmp_path, "rb") as in_fp:
                            for chunk in iter(lambda: in_fp.read(1024 * 1024), b""):
                                final_fp.write(chunk)
                    await asyncio.to_thread(manual_copy)
                    self._safe_remove(tmp_path) # Remove stuff I guess

            if segment_dir and not keep_segment_dir:
                self._safe_rmtree(segment_dir) # Delete segment dir (cleanup) (optional)
            if segment_state_path: # Delete segment state (optional)
                self._safe_remove(segment_state_path)
            self.logger.info(f"Download completed successfully: path={path}")

            if return_report: # Do a report, if user asked to
                self.logger.debug(
                    f"Returning completed report: downloaded={report.downloaded} missing={len(report.missing)}"
                )
                return report
            return True
        except Exception as e:
            self.logger.exception(f"Unhandled exception in download wrapper: {e}")
            return False

    # A download that finished every segment must not be thrown away because our
    # own process still holds the assembled file open. Windows refuses to move an
    # open file, so closure is deterministic and the move is retried briefly.
    _RENAME_RETRY_ATTEMPTS = 4
    _RENAME_RETRY_INITIAL_DELAY = 0.05

    @staticmethod
    def _describe_path(path: str) -> str:
        """Metadata about a file, never its contents."""
        try:
            return f"exists size={os.path.getsize(path)}"
        except OSError:
            return "missing"

    def _replace_with_retry(self, source: str, target: str) -> None:
        """Move `source` onto `target`, tolerating a brief sharing violation.

        Every handle this process owns is closed before this runs. Windows can
        still hold a file for a moment afterwards - a scanner, the indexer, or the
        filesystem itself - and that window is short. Only WinError 32 is retried;
        any other PermissionError means something is genuinely wrong and must not
        be papered over by waiting.

        os.replace rather than os.rename: rename refuses to overwrite an existing
        target on Windows, which turns a re-download into a spurious failure.
        """
        delay = self._RENAME_RETRY_INITIAL_DELAY
        for attempt in range(1, self._RENAME_RETRY_ATTEMPTS + 1):
            try:
                os.replace(source, target)
                return
            except PermissionError as error:
                if getattr(error, "winerror", None) != 32:
                    raise
                if attempt >= self._RENAME_RETRY_ATTEMPTS:
                    self.logger.error(
                        "Final rename blocked by WinError 32 after %s attempts; giving up. tmp=[%s] target=[%s]",
                        attempt,
                        self._describe_path(source),
                        self._describe_path(target),
                    )
                    raise
                self.logger.warning(
                    "Final rename blocked by WinError 32; retry %s/%s after %.0f ms",
                    attempt + 1,
                    self._RENAME_RETRY_ATTEMPTS,
                    delay * 1000,
                )
                time.sleep(delay)
                delay *= 2

    def _close_quietly(self, container: Any, role: str) -> None:
        if container is None:
            return
        try:
            container.close()
        except Exception as exc:
            # Never let a failure to close replace the error we are actually
            # reporting - but do not hide it either.
            self.logger.debug("Closing %s container failed: %s", role, exc)

    def _convert_ts_to_mp4(self, input_path: str, output_path: str,
                           callback: Callable[[int, int], None] | None = None, ios_support: bool = False) -> None:
        start_ts = time.perf_counter()
        self.logger.info("Remux start: input=%s output=%s", input_path, output_path)

        try:
            input_size = os.path.getsize(input_path)
            self.logger.debug("Remux input size: %s bytes", input_size)
        except Exception as e:
            self.logger.debug("Remux input size unavailable: %s", e)

        try:
            from av import open as av_open  # type: ignore[import-not-found]
            from av.audio.resampler import AudioResampler  # type: ignore[import-not-found]
            import av.audio.frame  # Used for runtime isinstance check
        except (ModuleNotFoundError, ImportError) as e:
            self.logger.error("PyAV import failed for remux: %s", e, exc_info=True)
            raise ModuleNotFoundError(
                f"PyAV is required for remuxing. Install with pip install av. Not supported on Termux! {e}") from e

        self.logger.debug("Opening input for remux: %s", input_path)
        input_ = av_open(input_path)
        output = None
        pass_through = False

        try:
            fmt_name = (input_.format.name or "").lower()
            self.logger.info("Input format detected: %s", fmt_name or '<unknown>')

            if fmt_name == "mpegts":
                # Fix 1: Suppress the stub mismatch for av.open
                output = av_open(output_path, mode="w", format="mp4",
                                 options={"movflags": "faststart"})  # type: ignore[arg-type]

                # --- VIDEO ---
                in_video = input_.streams.video[0]
                out_video = output.add_stream_from_template(template=in_video)
                self.logger.debug(
                    "Video stream: codec=%s bit_rate=%s",
                    getattr(in_video.codec_context, 'name', None), getattr(in_video.codec_context, 'bit_rate', None)
                )

                # --- AUDIO ---
                in_audio = next((s for s in input_.streams if s.type == "audio"), None)
                out_audio = None
                transcode_audio = False
                resampler = None

                if in_audio:
                    # Fix 3: Explicitly narrow out None
                    assert in_audio is not None

                    # Fix 2: Cast context to AudioCodecContext so IDE knows about sample_rate and layout
                    audio_ctx = cast('AudioCodecContext', in_audio.codec_context)

                    copy_ok = {"aac"} if ios_support else {"aac", "alac", "mp3"}
                    codec_name = (audio_ctx.name or "").lower()
                    sample_rate = audio_ctx.sample_rate or 0
                    layout_name = audio_ctx.layout.name if getattr(audio_ctx, "layout", None) else "unknown"

                    self.logger.debug(
                        "Audio stream: codec=%s sample_rate=%s layout=%s", codec_name, sample_rate, layout_name
                    )

                    if codec_name in copy_ok:
                        out_audio = output.add_stream_from_template(template=in_audio)
                        self.logger.info("Audio codec MP4-compatible; remuxing without transcoding.")
                    else:
                        transcode_audio = True
                        sample_rate = audio_ctx.sample_rate or 48000
                        layout = audio_ctx.layout.name if getattr(audio_ctx, "layout", None) else "stereo"

                        out_audio = output.add_stream("aac", rate=sample_rate)
                        self.logger.info("Transcoding audio to AAC: sample_rate=%s layout=%s", sample_rate, layout)

                        try:
                            out_audio.layout = layout
                        except Exception as exc:
                            self.logger.warning("Exception in getting audio layout (doesn't matter): %s", exc)

                        resampler = AudioResampler(format="fltp", layout=layout, rate=sample_rate)
                else:
                    self.logger.info("No audio stream detected; remuxing video only.")

                # --- DEMUX ---
                demux_streams = [in_video] + ([in_audio] if in_audio else [])
                packets = input_.demux(demux_streams)

                try:
                    total = os.path.getsize(input_path)
                except Exception as exc:
                    self.logger.warning("Exception while getting path size for demuxing progress??? %s", exc)
                    total = 100

                self.logger.info("Demuxing packets: total_bytes=%s", total)
                progress_step = max(1, total // 10) if total else 0
                next_progress_log = progress_step if progress_step else 0
                current_progress = 0
                timestamp_offsets: dict[int, int] = {}
                last_dts: dict[int, int] = {}
                last_durations: dict[int, int] = {}

                for idx, packet in enumerate(packets):
                    pkt_size = getattr(packet, "size", 0) or 0
                    current_progress += pkt_size

                    if packet.dts is None:
                        if callback:
                            callback(current_progress, total)
                        continue

                    timestamp_correction = _normalize_packet_timestamps(
                        packet,
                        timestamp_offsets,
                        last_dts,
                        last_durations,
                    )
                    if timestamp_correction:
                        self.logger.info(
                            "Normalized HLS timestamp discontinuity: stream=%s correction=%s time_base=%s",
                            packet.stream.index,
                            timestamp_correction,
                            packet.time_base,
                        )

                    if packet.stream == in_video:
                        packet.stream = out_video
                        output.mux(packet)

                    elif in_audio and packet.stream == in_audio:
                        if not transcode_audio:
                            packet.stream = out_audio
                            output.mux(packet)
                        else:
                            assert out_audio is not None
                            for frame in packet.decode():
                                # Fix 4: Ensure the frame is recognized as an AudioFrame
                                if not isinstance(frame, av.audio.frame.AudioFrame):
                                    continue

                                frames = resampler.resample(frame) if resampler else [frame]
                                for f in frames:
                                    for enc_pkt in out_audio.encode(f):
                                        output.mux(enc_pkt)

                    if callback:
                        callback(current_progress, total)
                    if progress_step and current_progress >= next_progress_log:
                        self.logger.debug("Remux progress: bytes=%s/%s", current_progress, total)
                        next_progress_log += progress_step

                if transcode_audio and out_audio:
                    self.logger.debug("Flushing AAC encoder.")
                    for enc_pkt in out_audio.encode(None):
                        output.mux(enc_pkt)

            else:
                # Already MP4 - typically fragmented-MP4 HLS. Nothing to remux, the
                # assembled file only has to be moved into place. The move happens
                # after the finally below, because PyAV still has this very file
                # open right now and Windows will not move an open file.
                self.logger.info("Stream seems to be already in MP4! Skipping remux...")
                pass_through = True

        finally:
            # Deterministic, and in this order: the writer first, then the reader.
            # Relying on garbage collection here is what produced WinError 32.
            self._close_quietly(output, "output")
            self._close_quietly(input_, "input")

        if pass_through:
            self._replace_with_retry(input_path, output_path)
            elapsed = time.perf_counter() - start_ts
            self.logger.info("Remux skipped; file moved. elapsed=%.2fs", elapsed)
            return

        elapsed = time.perf_counter() - start_ts
        try:
            out_size = os.path.getsize(output_path)
            self.logger.info("Remux complete: output=%s size=%s bytes elapsed=%.2fs", output_path, out_size,
                             elapsed)
        except Exception as e:
            self.logger.info("Remux complete: output=%s elapsed=%.2fs (size unavailable: %s)", output_path,
                             elapsed, e)

    async def legacy_download(self, url: str, configuration: DownloadConfigRAW) -> bool:
        """
        Download a file using streaming with stall tolerance and resume.
        Supports fast concurrent range downloading if the server supports it and allow_multipart is True.
        Assumes self.session is an AsyncSession.
        """
        path = configuration.path
        max_retries = configuration.max_retries
        read_timeout = configuration.read_timeout
        stop_event = configuration.stop_event
        allow_multipart = configuration.allow_multipart
        callback = configuration.callback
        chunk_size = configuration.chunk_size
        max_workers = configuration.max_workers

        self.logger.info(
"""Legacy download start: url=%s path=%s
max_retries=%s read_timeout=%s
stop_event_set=%s
allow_multipart=%s""", url, path, max_retries, read_timeout, bool(stop_event and stop_event.is_set()),
        allow_multipart)

        if stop_event is not None and stop_event.is_set():
            self.logger.warning("Stop event already set; cancelling legacy download.")
            raise DownloadCancelled("Download cancelled.")

        # Ensure session is initialized
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        progress_bar = None
        if callback is None:
            progress_bar = Callback()
            self.logger.debug("legacy_download: no callback provided, using default progress bar")

        timeout = read_timeout

        # 1. Check if the server supports Range requests and get file size (if multipart is allowed)
        file_size = 0
        accept_ranges = ""

        if allow_multipart:
            # We MUST request uncompressed content for range downloads, otherwise:
            # 1) Content-Length from HEAD reflects the compressed size, not the real file size.
            # 2) Mid-file Range requests on compressed streams cause libcurl error 61
            #    ("incorrect header check") because partial gzip lacks a valid header.
            no_compress = {"Accept-Encoding": "identity"}
            try:
                head_resp = await session.head(url, timeout=timeout, allow_redirects=True, headers=no_compress)
                if head_resp.status_code == 405:  # Method Not Allowed, fallback to streaming GET
                    head_resp_stream = await session.request("GET", url, timeout=timeout, allow_redirects=True,
                                                             stream=True, headers=no_compress)
                    file_size = int(head_resp_stream.headers.get("Content-Length", 0))
                    accept_ranges = head_resp_stream.headers.get("Accept-Ranges", "")
                else:
                    file_size = int(head_resp.headers.get("Content-Length", 0))
                    accept_ranges = head_resp.headers.get("Accept-Ranges", "")
            except Exception as e:
                self.logger.warning("Failed to fetch HEAD info for concurrent check: %s.", e)

        # 2. Execute Fast Multipart Download if supported and allowed
        if allow_multipart and file_size > 0 and accept_ranges == "bytes":
            self.logger.info("Server supports Range requests. Starting fast multipart download"
                             "or %s bytes.", file_size)

            # Pre-allocate file
            def allocate_file() -> None:
                if not os.path.exists(path):
                    with open(path, "wb") as file_alloc:
                        file_alloc.truncate(file_size)
                elif os.path.getsize(path) != file_size:
                    # File exists but size mismatch, truncate to correct size
                    with open(path, "r+b") as file_alloc_size:
                        file_alloc_size.truncate(file_size)
            await asyncio.to_thread(allocate_file)

            # We will use an array to track progress of chunks
            # A chunk map: {chunk_index: bytes_downloaded}
            chunk_progress = {}
            total_downloaded = [0]  # List to allow modification in inner func
            # Determine chunk sizes based on file size, but keep reasonable bounds
            # For massive files, don't create 10,000 workers.
            target_chunk_size = max(chunk_size, min(10 * 1024 * 1024, file_size // 10)) # Between 1MB and 10MB

            semaphore = asyncio.Semaphore(max_workers)

            async def download_chunk(start_chunk: int, end_chunk: int, chunk_idx_now: int) -> bool:
                nonlocal total_downloaded
                headers_chunk = {"Range": f"bytes={start_chunk}-{end_chunk}", "Accept-Encoding": "identity"}
                chunk_progress[chunk_idx_now] = 0

                for attempt_chunk in range(max_retries + 1):
                    if stop_event is not None and stop_event.is_set():
                        return False

                    try:
                        async with semaphore:
                            resp = await cast(Any, session).request(
                                "GET", url, headers=headers_chunk, timeout=timeout, allow_redirects=True, stream=True
                            )
                            resp.raise_for_status()

                            # Open file once for this chunk download attempt
                            file = await asyncio.to_thread(lambda: open(path, "rb+"))
                            try:
                                await asyncio.to_thread(file.seek, start_chunk + chunk_progress[chunk_idx_now])
                                async for data in resp.aiter_content():
                                    if stop_event is not None and stop_event.is_set():
                                        return False

                                    await asyncio.to_thread(cast(Any, file).write, data)

                                    data_len = len(data)
                                    chunk_progress[chunk_idx_now] += data_len
                                    total_downloaded[0] += data_len

                                    if callback:
                                        callback(total_downloaded[0], file_size)
                                    elif progress_bar:
                                        progress_bar.text_progress_bar(downloaded=total_downloaded[0], total=file_size)
                            finally:
                                await asyncio.to_thread(file.close)

                            return True # Chunk success

                    except Exception as exc:
                        if attempt_chunk < max_retries:
                            self.logger.warning("Chunk %s failed (attempt %s/%s): %s",
                                                chunk_idx_now, attempt_chunk + 1, max_retries, exc)
                            # Reset progress for this chunk before retry
                            total_downloaded[0] -= chunk_progress[chunk_idx_now]
                            chunk_progress[chunk_idx_now] = 0
                            await asyncio.sleep(1 * attempt_chunk)
                        else:
                            self.logger.error("Chunk %s permanently failed: %s", chunk_idx_now, exc, exc_info=True)
                            return False
                return False

            tasks = []
            chunk_idx = 0
            for start in range(0, file_size, target_chunk_size):
                end = min(start + target_chunk_size - 1, file_size - 1)
                tasks.append(download_chunk(start, end, chunk_idx))
                chunk_idx += 1

            results = await asyncio.gather(*tasks)

            if progress_bar:
                # We set it to None instead of del to avoid analyzer confusion about potential unassigned reference
                progress_bar = None

            if stop_event is not None and stop_event.is_set():
                raise DownloadCancelled("Download cancelled.")

            if not all(results):
                raise NetworkRequestError("One or more chunks failed to download completely.")

            self.logger.info("Fast multipart download complete: path=%s", path)
            return True

        # 3. Fallback to standard linear streaming download
        if not allow_multipart:
            self.logger.info("allow_multipart=False. Forcing linear streaming download.")
        else:
            self.logger.info("Server does not support Range requests or size is 0. Falling back to linear streaming.")

        downloaded_so_far = 0
        attempt = 0
        etag = None

        while True:
            if stop_event is not None and stop_event.is_set():
                self.logger.warning("Stop event set; cancelling legacy download.")
                raise DownloadCancelled("Download cancelled.")
            headers = {}
            if downloaded_so_far:
                headers["Range"] = f"bytes={downloaded_so_far}-"

            try:
                response = await cast(Any, session).request(
                    "GET", url, headers=headers, allow_redirects=True, timeout=timeout, stream=True
                )
                if downloaded_so_far and response.status_code == 200:
                    self.logger.warning("Server ignored Range request; restarting download from scratch.")
                    downloaded_so_far = 0
                response.raise_for_status()

                etag_cur = response.headers.get("ETag")
                if etag is None:
                    etag = etag_cur
                elif etag_cur and etag_cur != etag:
                    raise RuntimeError("Remote content changed during download")

                total = None
                cr = response.headers.get("Content-Range")
                if cr and "/" in cr:
                    try: total = int(cr.rsplit("/", 1)[1])
                    except ValueError: pass
                if total is None:
                    try: total = int(response.headers.get("Content-Length", "0")) or None
                    except ValueError: pass

                # Fix fallback if total size is still missing
                if total is None:
                    total = 0

                mode = "ab" if downloaded_so_far else "wb"
                f = await asyncio.to_thread(cast(Any, open), path, mode)
                try:
                    await asyncio.to_thread(f.seek, 0, 2)  # Move to EOF
                    async for chunk in response.aiter_content():
                        if stop_event is not None and stop_event.is_set():
                            raise DownloadCancelled("Download cancelled.")
                        if not chunk:
                            continue
                        await asyncio.to_thread(f.write, chunk)
                        downloaded_so_far += len(chunk)

                        if callback:
                            callback(downloaded_so_far, total)
                        elif progress_bar:
                            progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=total)
                finally:
                    await asyncio.to_thread(f.close)

                if progress_bar:
                    progress_bar = None
                self.logger.info("Legacy download complete: bytes=%s path=%s", downloaded_so_far, path)
                return True
            except RequestsError as e:
                err_str = str(e).lower()
                if "timeout" in err_str or "read" in err_str:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    backoff = min(2 ** attempt, 30)
                    self.logger.warning("Read timeout; retrying %s/%s in %s", attempt, max_retries,  backoff)
                    if stop_event is not None and stop_event.wait(backoff):
                        raise DownloadCancelled("Download cancelled.") from e
                    else:
                        await asyncio.sleep(backoff)
                    continue
                else:
                    raise NetworkRequestError(f"Stream for: {url} was closed or failed: {e}") from e
            except DownloadCancelled:
                raise
            except Exception as exc:
                error = traceback.format_exc()
                raise NetworkRequestError(f"Unknown error for: {url} -->: {error}") from exc

        return False
