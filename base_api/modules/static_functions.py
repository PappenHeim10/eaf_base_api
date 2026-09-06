import os
import re
import math
import json
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import PurePath
from .type_hints import (
    DownloadState,
    ProgressiveDownloadState,
    PROGRESSIVE_STATE_KIND,
    PROGRESSIVE_STATE_VERSION,
)
from datetime import timezone, datetime
from curl_cffi.requests import Response
from typing import Dict, Any, cast, List, Callable, Literal, Union
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit


HEIGHT_FROM_URI = re.compile(r'(?<!\d)(\d{3,4})[pP](?!\d)')  # e.g., 1080p, 720P


def eval_flags(flags: list[int]) -> int:
    """
    Evaluate flags.

    Args:
        flags (list[int]): List of flags arguments.

    Returns:
        int: The flag(s) value.
    """

    if len(flags):
        return flags[0]

    return 0


def subc(*args: Any) -> Callable[..., Any]:
    """
    Compile a substraction regex and apply its replacement to each call.

    Returns:
        Callable: Wrapped regex callable.
    """

    *flags, pattern, repl = args
    flags_val = eval_flags(flags)

    regex = re.compile(pattern, flags_val)

    def wrapper(*args_: Any) -> Any:
        return regex.sub(repl, *args_)

    return wrapper

parse_challenge = subc(re.DOTALL, r'(?:var )|(?:/\*.*?\*/)|\s|\n|\t|(?:n;)', '') # Parse challenge syntax
other_challenge = subc(re.DOTALL, r'(if.*?&1\)|else)', r'\1:'                  ) # Convert challenge syntax



def least_factors(n: int) -> int:
    """
    Returns the least factor of a number.
    """
    if n <= 0:
        return 0
    if n % 2 == 0:
        return 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return i
    return n


type QualityPreference = int | Literal["best", "half", "worst"]

QUALITY_LABELS = frozenset({"best", "half", "worst"})


def is_video_playlist(variant: Any) -> bool:
    """Filter out I-frames/audio-only playlists."""
    # m3u8 lib sometimes sets is_iframe if EXT-X-I-FRAME-STREAM-INF is present.
    if getattr(variant, "is_iframe", False):
        return False

    # If codecs known and contain only audio (mp4-a, ac-3, ec-3, etc.)
    codecs = getattr(variant.stream_info, "codecs", None) if getattr(variant, "stream_info", None) else False
    if codecs:
        # very light heuristic: if no video codec substring, probably audio-only.
        # video: avc1, hvc1, hev1, vp9, av01, dvh
        codecs_text = str(codecs).lower()
        if not any(v in codecs_text for v in ("avc1", "hvc1", "hev1", "av01", "vp9", "dvh")):
            return False

    return True


def get_segment_index_width(total: int) -> int:
    return max(6, len(str(max(0, total - 1))))


COMMON_QUALITIES = frozenset({
    144,
    240,
    250,
    360,
    480,
    540,
    720,
    1080,
    1440,
    2160,
})
# Kept as an alias for callers that imported the name during the 4.0 rollout.
ALLOWED_QUALITIES = COMMON_QUALITIES


def validate_quality(value: int) -> int:
    """Validate a normalized quality without restricting provider-specific tiers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Invalid quality type: {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Invalid video quality: {value!r}")

    return value


def normalize_quality(value: str | int) -> int:
    """
    Convert a quality into a canonical integer.

    Accepted:
        720
        "720"
        "720p"

    Rejected:
        "best"
        "half"
        "worst"
        "720p60"
        zero or negative values
    """

    if isinstance(value, bool):
        raise TypeError("A boolean is not a valid video quality.")

    if isinstance(value, int):
        quality = value

    elif isinstance(value, str):
        value = value.strip().lower()

        match = re.fullmatch(r"(\d+)[pP]?", value)

        if not match:
            raise ValueError(
                f"Invalid video quality: {value!r}"
            )

        quality = int(match.group(1))

    else:
        raise TypeError(
            f"Invalid quality type: {type(value).__name__}"
        )

    return validate_quality(quality)

def normalize_quality_preference(
    value: str | int,
) -> QualityPreference:

    if isinstance(value, str):
        value = value.strip().lower()

        if value in QUALITY_LABELS:
            return cast(QualityPreference, value)

    return normalize_quality(value)



def choose_quality_from_list(
    available: Iterable[str | int],
    target: str | int,
    default_fallback: str | int | None = None,
) -> int:
    """Choose a quality, falling back to the nearest available numeric tier."""

    available_ints = normalize_qualities(available)

    if not available_ints:
        if default_fallback is not None:
            return normalize_quality(default_fallback)
        raise ValueError(
            "No valid video qualities are available."
        )

    try:
        preference = normalize_quality_preference(target)
    except (TypeError, ValueError):
        if default_fallback is not None:
            return normalize_quality(default_fallback)
        raise

    if preference == "best":
        return available_ints[-1]

    if preference == "worst":
        return available_ints[0]

    if preference == "half":
        return available_ints[len(available_ints) // 2]

    # Prefer the higher tier when two variants are equally close.
    return min(
        available_ints,
        key=lambda quality: (abs(quality - preference), -quality),
    )


def normalize_qualities(
    values: Iterable[str | int],
) -> list[int]:
    """
    Return canonical, unique qualities sorted worst -> best.
    """

    qualities: set[int] = set()
    for value in values:
        try:
            qualities.add(normalize_quality(value))
        except (TypeError, ValueError):
            # Provider data can contain labels such as "auto" alongside real
            # qualities.  They should not make the usable variants disappear.
            continue

    return sorted(qualities)


def quality_from_variant(variant: Any) -> int | None:
    """Extract a quality tier from a landscape or portrait HLS variant."""
    stream_info = getattr(variant, "stream_info", None)
    resolution = getattr(stream_info, "resolution", None)
    if resolution:
        try:
            width, height = resolution
            # Quality names describe the shorter side: 1920x1080 and
            # 1080x1920 are both 1080p.
            return normalize_quality(min(int(width), int(height)))
        except (TypeError, ValueError):
            pass

    uri = getattr(variant, "uri", None)
    if isinstance(uri, str):
        match = HEIGHT_FROM_URI.search(uri)
        if match:
            try:
                return normalize_quality(match.group(1))
            except (TypeError, ValueError):
                pass

    return None


def collect_variants(master: Any) -> list[dict[str, Any]]:
    """Return video variants with normalized, comparable metadata."""
    variants: list[dict[str, Any]] = []
    for variant in getattr(master, "playlists", ()):
        if not is_video_playlist(variant):
            continue

        stream_info = getattr(variant, "stream_info", None)
        variants.append({
            "uri": getattr(variant, "uri", ""),
            "quality": quality_from_variant(variant),
            "bandwidth": int(getattr(stream_info, "bandwidth", 0) or 0),
            "frame_rate": float(getattr(stream_info, "frame_rate", 0.0) or 0.0),
            "resolution": getattr(stream_info, "resolution", None),
            "raw": variant,
        })

    return variants


def available_qualities(variants: Iterable[dict[str, Any]]) -> list[int]:
    """Return sorted, unique integer qualities from normalized variants."""
    return normalize_qualities(
        variant["quality"]
        for variant in variants
        if variant.get("quality") is not None
    )


def choose_variant(
    variants: Iterable[dict[str, Any]],
    target: str | int,
) -> dict[str, Any]:
    """Select an HLS variant with quality and bandwidth fallbacks."""
    candidates = list(variants)
    if not candidates:
        raise ValueError("No video variants are available.")

    preference = normalize_quality_preference(target)
    qualities = available_qualities(candidates)
    if qualities:
        selected_quality = choose_quality_from_list(qualities, preference)
        matching = [
            variant
            for variant in candidates
            if variant.get("quality") == selected_quality
        ]
        return max(
            matching,
            key=lambda variant: (
                variant.get("bandwidth", 0),
                variant.get("frame_rate", 0.0),
            ),
        )

    # Some masters expose only bandwidth. Labels can still be ranked, while a
    # numeric request falls back to the highest-bandwidth usable variant.
    ordered = sorted(
        candidates,
        key=lambda variant: (
            variant.get("bandwidth", 0),
            variant.get("frame_rate", 0.0),
        ),
    )
    if preference == "worst":
        return ordered[0]
    if preference == "half":
        return ordered[len(ordered) // 2]
    return ordered[-1]


# Compatibility wrappers for callers of the pre-4.0 helper names.
def normalize_quality_value(quality: str | int) -> QualityPreference:
    return normalize_quality_preference(quality)


def height_from_variant(variant: Any) -> int | None:
    return quality_from_variant(variant)


def pick_by_label(
    variants: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    return choose_variant(variants, label)


def pick_by_height(
    variants: list[dict[str, Any]],
    target: int,
) -> dict[str, Any]:
    return choose_variant(variants, target)


def segment_file_path(segment_dir, index: int, width: int) -> str:
    return os.path.join(segment_dir, f"seg_{index:0{width}d}.ts")


def write_segment_state(state_path: str, state: DownloadState) -> None:
    tmp_path = f"{state_path}.tmp"
    payload = asdict(state)
    for path_key in ("output_path", "segment_dir"):
        if isinstance(payload[path_key], PurePath):
            payload[path_key] = str(payload[path_key])
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def load_segment_state(state_path: str) -> Dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as fp:
        return cast(Dict[str, Any], json.load(fp))


def build_segment_state(
    *,
    segments: List[Any],
    missing: List[int],
    segment_dir: str | None,
    segment_index_width: int,
    path: str,
    quality: str,
    start_segment: int,
    m3u8_url: str | None,
    created_at: str | None = None,
    version: int = 1
) -> DownloadState:
    now = datetime.now(timezone.utc).isoformat()
    state = DownloadState(
        version=version,
        created_at=created_at or now,
        updated_at=None,
        m3u8_url=m3u8_url,
        quality=quality,
        output_path=path,
        segment_dir=segment_dir,
        segment_index_width=segment_index_width,
        start_segment=start_segment,
        total=len(segments),
        missing=missing,
        segments=segments
    )
    return state


# --- progressive HTTP transport ------------------------------------------------

#: `bytes <first>-<last>/<complete>` or `bytes <first>-<last>/*`. Only the
#: satisfied form; the unsatisfied `bytes */<complete>` form of a 416 is parsed
#: by `parse_unsatisfied_content_range` below, because the two answer different
#: questions and must never be confused for one another.
_CONTENT_RANGE = re.compile(
    r"\A\s*bytes\s+(?P<first>\d+)\s*-\s*(?P<last>\d+)\s*/\s*(?P<total>\d+|\*)\s*\Z",
    re.IGNORECASE,
)

_UNSATISFIED_CONTENT_RANGE = re.compile(
    r"\A\s*bytes\s+\*\s*/\s*(?P<total>\d+)\s*\Z", re.IGNORECASE
)


def parse_content_range(value: Any) -> tuple[int, int, int | None] | None:
    """Parse a satisfied `Content-Range` into `(first, last, complete)`.

    Returns `None` for anything this engine will not act on - a missing header,
    a different unit, a malformed value, or a range whose end precedes its
    start. `complete` is `None` for the `/*` form, which states the range but
    not the resource's total size.

    Unparsable is deliberately not "assume it is fine": a resumed download
    appends to a real file, so a range it cannot verify is a range it must not
    trust.
    """
    if not isinstance(value, str):
        return None
    match = _CONTENT_RANGE.match(value)
    if match is None:
        return None
    first = int(match.group("first"))
    last = int(match.group("last"))
    if last < first:
        return None
    raw_total = match.group("total")
    total = None if raw_total == "*" else int(raw_total)
    if total is not None and last >= total:
        return None
    return first, last, total


def parse_unsatisfied_content_range(value: Any) -> int | None:
    """Parse the `bytes */<complete>` form a 416 answers with.

    This is the one piece of information that turns a 416 from "start over"
    into "the local file may already be the whole thing": it is the resource's
    total size, stated by a server that just refused the range asked for.
    """
    if not isinstance(value, str):
        return None
    match = _UNSATISFIED_CONTENT_RANGE.match(value)
    return int(match.group("total")) if match else None


def normalize_etag(value: Any) -> tuple[str | None, bool]:
    """Split an `ETag` header into `(entity_tag, is_weak)`.

    The returned tag keeps its quotes, because an entity tag is compared as the
    opaque string the server sent - `"abc"` and `abc` are not the same tag.
    The weak flag is what decides whether it may ever appear in `If-Range`:
    RFC 9110 allows only a strong validator there, and a weak one silently
    turns a range request into a wrong answer on servers that accept it anyway.
    """
    if not isinstance(value, str):
        return None, False
    tag = value.strip()
    if not tag:
        return None, False
    weak = tag[:2].upper() == "W/"
    if weak:
        tag = tag[2:].strip()
        if not tag:
            return None, False
    return tag, weak


def build_progressive_state(
    *,
    url: str,
    output_path: str,
    temp_path: str,
    total_size: int | None,
    downloaded_bytes: int,
    etag: str | None,
    etag_weak: bool,
    last_modified: str | None,
    created_at: str | None = None,
    identity: str | None = None,
) -> ProgressiveDownloadState:
    now = datetime.now(timezone.utc).isoformat()
    return ProgressiveDownloadState(
        version=PROGRESSIVE_STATE_VERSION,
        kind=PROGRESSIVE_STATE_KIND,
        created_at=created_at or now,
        updated_at=now,
        # With an identity to compare against, the URL is only ever read by a
        # human looking at a stale state file, so it is stored redacted. Without
        # one it *is* the comparison and has to be stored as given.
        url=url if identity is None else format_url_for_log(url),
        identity=identity,
        output_path=output_path,
        temp_path=temp_path,
        total_size=total_size,
        downloaded_bytes=downloaded_bytes,
        etag=etag,
        etag_weak=etag_weak,
        last_modified=last_modified,
    )


def write_progressive_state(state_path: str, state: ProgressiveDownloadState) -> None:
    """Persist a resume state crash-safely: full temp file, then atomic replace.

    A half-written state is worse than none at all, because the reader would
    take its byte count as truth about a real file. `os.replace` is atomic on
    every platform this runs on, so a crash leaves either the previous state or
    the new one - never a torn one.
    """
    tmp_path = f"{state_path}.tmp"
    payload = asdict(state)
    for path_key in ("output_path", "temp_path"):
        if isinstance(payload[path_key], PurePath):
            payload[path_key] = str(payload[path_key])
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def load_progressive_state(state_path: str) -> Dict[str, Any] | None:
    """Read a progressive resume state, or `None` if it is not ours to read.

    `None` rather than an exception for every "cannot use this" case: an HLS
    state file, a future schema version, a truncated file and a missing one all
    mean the same thing to the caller - start at byte zero. Only the schema is
    checked here; whether the state still matches the *file* on disk is decided
    against that file, not against the state.
    """
    try:
        with open(state_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != PROGRESSIVE_STATE_KIND:
        return None
    try:
        version = int(payload.get("version"))
    except (TypeError, ValueError):
        return None
    if version != PROGRESSIVE_STATE_VERSION:
        return None
    if not isinstance(payload.get("url"), str) or not payload["url"]:
        return None
    return cast(Dict[str, Any], payload)


def truncate(name: str, max_bytes: int = 245) -> str:  # only 245, because we need to append .mp4
    """
    Some websites have titles that are so long (looking at you missav.ws) that you can't name a file like
    that, and thus we need to make sure the file name doesn't exceed the OS limits lol
    """
    encoded = name.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        # Ensure not to cut in middle of a UTF-8 sequence
        while encoded[-1] & 0b11000000 == 0b10000000:
            encoded = encoded[:-1]
        return cast(bytes, cast(Any, encoded)).decode("utf-8", errors="ignore")
    return name


def str_to_bool(value: str) -> bool:
    # Some function that I have for some reason I don't know if this has ever been used lmao
    """
    This function is needed for the ArgumentParser for the CLI version of my APIs. It basically maps the
    booleans for the --no-title option to valid Python boolean values.
    """
    val = value.lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


#: What a redacted query parameter shows instead of its value. ASCII on
#: purpose: this string ends up in log files whose handlers may be writing in
#: the console's code page, and a log line must never fail to be written.
REDACTED_VALUE = "***"


def format_url_for_log(url: Any) -> str:
    """A URL with its query *values* removed, for logs, errors and state files.

    Scheme, host and path stay: they are what makes a line useful - which CDN,
    which resource - and none of them is a secret. The query is where signed
    media URLs put the parts that are. A `googlevideo` URL carries the viewer's
    IP, a session id, an expiry and a signature there; an S3 presigned URL
    carries credentials. Any of those in a log file, in an error message a UI
    shows, or in a resume state sitting in the user's download folder is a leak
    that no amount of care further up can prevent, because it is this library
    that writes those lines.

    Parameter *names* are kept. "Which parameters were present" answers a real
    diagnostic question - whether a range was signed, whether an expiry was
    set - without answering "what were they".

    A URL with no query is returned unchanged, which is the overwhelming
    majority of sources: PeerTube's object storage, a plain `.m3u8`, a direct
    MP4. Those go on logging exactly as they always have.
    """
    if not isinstance(url, str) or not url:
        return "<no url>"

    try:
        parts = urlsplit(url)
    except ValueError:
        # An unparsable URL is not worth guessing at, and printing it raw is
        # the one thing this function exists to avoid.
        return "<unparsable url>"

    if not parts.query and not parts.fragment:
        return url

    names: List[str] = []
    for pair in parts.query.split("&"):
        if not pair:
            continue
        name = pair.split("=", 1)[0]
        if name and name not in names:
            names.append(name)

    redacted = "&".join(f"{name}={REDACTED_VALUE}" for name in names)
    # The fragment is dropped rather than redacted: nothing in this library
    # sends one, so its presence is noise rather than signal.
    return urlunsplit((parts.scheme, parts.netloc, parts.path, redacted, ""))


def format_headers_for_log(headers: Any) -> Dict[str, str]:
    """Redact sensitive headers but keep enough signal for debugging."""
    sensitive = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-csrf-token",
        "x-xsrf-token",
    }
    out: Dict[str, str] = {}
    for key, value in headers.items():
        l_key = key.lower()
        if l_key in sensitive:
            if l_key == "cookie":
                parts = [p.split("=", 1)[0].strip() for p in str(value).split(";") if p.strip()]
                value = f"<redacted:{','.join(parts)}>" if parts else "<redacted>"
            else:
                value = "<redacted>"
        if key in out:
            out[key] = f"{out[key]}, {value}"
        else:
            out[key] = str(value)
    return out


def response_body_preview(logger, response: Response, max_bytes: int = 512) -> str:
    try:
        raw = response.content[:max_bytes]
    except Exception as e:
        return f"<failed to read body: {e}>"
    if not raw:
        return "<empty>"
    enc = getattr(response, "encoding", None) or "utf-8"
    try:
        text = cast(bytes, cast(Any, raw)).decode(enc, errors="replace")
    except Exception as exc:
        logger.error(f"There was an error while decoding text from the response body preview: {exc}")
        text = cast(bytes, cast(Any, raw)).decode("utf-8", errors="replace")
    return text.replace("\r", "\\r").replace("\n", "\\n")


def parse_retry_after(logger, response: Response) -> float | None:
    """Parse Retry-After (seconds or http-date) into seconds; None if not present/invalid."""
    v = response.headers.get("Retry-After")
    if not v:
        return None
    try:
        # numeric seconds
        return float(v)
    except ValueError:
        try:
            dt = parsedate_to_datetime(v)
            # Convert to seconds from now
            delta = (dt - dt.now(dt.tzinfo)).total_seconds()
            # clamp: negative -> 0
            return max(0.0, delta)
        except Exception as exc:
            logger.warning(f"Couldn't parse retry after in 429 error: {exc}")
            return None


def log_precondition_failed(logger, response: Response, attempt: int) -> None:
    req = response.request
    try:
        req_headers = format_headers_for_log(req.headers) if req is not None else {}
    except Exception as e:
        req_headers = {"<error>": f"failed to format request headers: {e}"}

    try:
        resp_headers = format_headers_for_log(response.headers)
    except Exception as e:
        resp_headers = {"<error>": f"failed to format response headers: {e}"}

    try:
        cond_headers = [
            k for k in req.headers.keys() if k.lower().startswith("if-")
        ] if req is not None else []
    except Exception as exc:
        logger.warning(f"Could not get the conditional headers: {exc}")
        cond_headers = []

    cond_note = f" conditional_headers={cond_headers}" if cond_headers else ""
    body_preview = response_body_preview(logger=logger, response=response)

    logger.warning(
        "HTTP 412 precondition failed (attempt %d) for %s %s.%s request_headers=%s response_headers=%s body_preview=%s",
        attempt + 1,
        getattr(req, "method", "UNKNOWN") if req is not None else "UNKNOWN",
        response.url,
        cond_note,
        req_headers,
        resp_headers,
        body_preview,
    )


def strip_title(
    title: str,
    max_length: int = 255,
    default_name: str = "untitled",
    max_bytes: int = 245,
) -> str:
    """Sanitize a filename to be safe across Windows, macOS, Linux, and Android.

    Prevents path traversal, replaces illegal characters, handles Windows reserved
    names, and trims to a safe length in both characters and encoded bytes.

    ``max_bytes`` defaults to 245 to leave room for the extension the caller
    appends (see :func:`truncate`).
    """
    if not title:
        return default_name

    # 1. Normalize Unicode (converts full-width slashes/characters to standard ASCII where applicable)
    sanitized = unicodedata.normalize("NFKC", title)

    # 2. Extract only the filename component (strips drive letters & path prefixes)
    sanitized = PurePath(sanitized).name

    # 3. Replace illegal filename characters & explicit path separators with underscores
    illegal_chars = r'[<>:"/\\|?*\x00-\x1F]'
    sanitized = re.sub(illegal_chars, "_", sanitized)

    # 4. Strip invisible zero-width & non-printable Unicode control characters
    sanitized = re.sub(r"[\u200B-\u200D\uFEFF]", "", sanitized)

    # 5. Strip LEADING and TRAILING spaces and dots
    # (Prevents '.' and '..' traversal tokens and hidden files)
    sanitized = sanitized.strip(" .")

    # 6. Prevent Windows reserved filenames (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    name_only = sanitized.split(".")[0].upper()
    if name_only in reserved_names:
        sanitized = f"_{sanitized}"

    # 7. Trim to the character limit AND to the byte limit, then re-strip trailing
    # dots/spaces in case a cut landed mid-string.
    # The byte limit matters because filesystems cap the *encoded* name, not the
    # character count: ext4 and Android allow 255 bytes, so 255 CJK characters are
    # roughly three times over the limit even though they pass `max_length`.
    sanitized = truncate(sanitized[:max_length], max_bytes=max_bytes).rstrip(" .")

    # 8. Return default fallback if sanitization leaves an empty string
    return sanitized if sanitized else default_name
