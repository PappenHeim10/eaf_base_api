"""A signed media URL must not reach a log file, an error message or a state file.

Signed URLs are ordinary now: object storage presigns with credentials in the
query, and media CDNs put the viewer's IP, a session id, an expiry and a
signature there. This library is what logs the media URL - at INFO when a
download starts, at WARNING on every retry and restart - so no amount of
discipline in a caller can keep those values out of a log. The redaction has to
live here.

What must survive is the diagnostic value: which host, which path, which
parameters were present. A redaction that logged nothing would pass a naive
"no secrets" assertion and make every future download bug unreadable, so the
tests below check for the host and path as explicitly as they check against the
secrets.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path

import pytest

from base_api.base import BaseCore
from base_api.models import MediaSource
from base_api.modules.errors import AccessDeniedError, RequestRetriesExhausted
from base_api.modules.static_functions import format_url_for_log
from tests.loopback_server import serving
from tests.test_http_progressive import BLOB, MediaPlan, handler_for, http_config, make_core


#: The shape a real signed URL has, with values that cannot occur by accident.
SECRETS = {
    "expire": "1788668116",
    "ip": "2001-db8-SENTINEL",
    "ei": "EI-SENTINEL-VALUE",
    "sig": "SIG-SENTINEL-VALUE",
    "lsig": "LSIG-SENTINEL-VALUE",
    "spc": "SPC-SENTINEL-VALUE",
}
SIGNED_QUERY = "&".join(f"{name}={value}" for name, value in SECRETS.items())


def signed(base_url: str) -> str:
    return f"{base_url}/media.mp4?{SIGNED_QUERY}"


def assert_no_secret(text: str) -> None:
    for name, value in SECRETS.items():
        assert value not in text, f"{name}'s value leaked: {text[:400]}"


# --- the helper itself -----------------------------------------------------------


def test_a_url_without_a_query_is_returned_unchanged():
    """The common case must keep logging exactly as it always has."""
    plain = "https://media.example/object-storage/web_videos/9d3c1f52-720.mp4"

    assert format_url_for_log(plain) == plain


def test_query_values_go_and_parameter_names_stay():
    redacted = format_url_for_log(f"https://cdn.example/videoplayback?{SIGNED_QUERY}")

    assert_no_secret(redacted)
    assert redacted.startswith("https://cdn.example/videoplayback?")
    for name in SECRETS:
        assert f"{name}=" in redacted, f"the parameter name {name} should survive"


def test_a_presigned_object_storage_url_keeps_only_its_shape():
    """The other signed URL this library will meet: S3-style presigning."""
    redacted = format_url_for_log(
        "https://s3.example/bucket/9d3c1f52-720.mp4"
        "?X-Amz-Credential=AKIA-SENTINEL&X-Amz-Signature=DEADBEEF-SENTINEL&X-Amz-Expires=900"
    )

    assert "AKIA-SENTINEL" not in redacted
    assert "DEADBEEF-SENTINEL" not in redacted
    assert redacted.startswith("https://s3.example/bucket/9d3c1f52-720.mp4?")
    assert "X-Amz-Signature=" in redacted


def test_a_fragment_is_dropped():
    assert format_url_for_log("https://cdn.example/a.mp4#t=30") == "https://cdn.example/a.mp4"


@pytest.mark.parametrize("value", [None, "", 42, b"https://x/y"])
def test_anything_that_is_not_a_url_never_gets_printed_raw(value):
    assert format_url_for_log(value) == "<no url>"


def test_an_unparsable_url_is_not_echoed():
    redacted = format_url_for_log("https://[not-an-ipv6-literal/a.mp4?sig=SIG-SENTINEL-VALUE")

    assert_no_secret(redacted)


# --- the transport ---------------------------------------------------------------


class _Collector(logging.Handler):
    def __init__(self, sink: list[logging.LogRecord]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record)


def run_and_collect(core: BaseCore, configuration) -> tuple[list[str], BaseException | None]:
    """Drive one download at DEBUG and return every rendered record.

    Collected off the core's own logger, which no longer propagates to root -
    the same reason `test_no_segment_payload_reaches_the_log` does it this way.
    """
    records: list[logging.LogRecord] = []
    handler = _Collector(records)
    core.logger.addHandler(handler)
    previous = core.logger.level
    core.logger.setLevel(logging.DEBUG)
    error: BaseException | None = None
    try:
        asyncio.run(core.download(configuration))
    except BaseException as raised:  # noqa: BLE001 - the failure is the subject
        error = raised
    finally:
        core.logger.removeHandler(handler)
        core.logger.setLevel(previous)
    return [record.getMessage() for record in records], error


def test_a_successful_download_logs_the_host_and_path_but_no_query_value(tmp_path: Path):
    plan = MediaPlan()
    with serving(handler_for(plan)) as base_url:
        core = make_core()
        messages, error = run_and_collect(
            core, http_config(tmp_path, signed(base_url))
        )

    assert error is None
    rendered = "\n".join(messages)
    assert_no_secret(rendered)
    assert "/media.mp4" in rendered, "the path is the diagnostic value; it must survive"


def test_a_refused_download_leaks_nothing_through_its_error(tmp_path: Path):
    """403 is where a signed URL most often ends up in front of a user."""
    plan = MediaPlan(script=[{"status": 403}])
    with serving(handler_for(plan)) as base_url:
        core = make_core()
        messages, error = run_and_collect(
            core, http_config(tmp_path, signed(base_url))
        )

    assert isinstance(error, AccessDeniedError)
    assert_no_secret(str(error))
    assert_no_secret("\n".join(messages))


def test_an_exhausted_retry_leaks_nothing_through_its_error(tmp_path: Path):
    """Every retry warning names the URL, and so does the final exception."""
    plan = MediaPlan(script=[{"status": 503}, {"status": 503}, {"status": 503}])
    with serving(handler_for(plan)) as base_url:
        core = make_core()
        messages, error = run_and_collect(
            core, http_config(tmp_path, signed(base_url))
        )

    assert isinstance(error, RequestRetriesExhausted)
    assert_no_secret(str(error))
    assert_no_secret(error.url)
    joined = "\n".join(messages)
    assert_no_secret(joined)
    assert "retrying in" in joined, "the retry warnings are the records under test"


def test_a_redirect_hop_is_logged_redacted(tmp_path: Path):
    plan = MediaPlan(script=[{"location": "/elsewhere.mp4?sig=SIG-SENTINEL-VALUE"}])
    with serving(handler_for(plan)) as base_url:
        core = make_core()
        messages, error = run_and_collect(
            core, http_config(tmp_path, signed(base_url))
        )

    assert error is None
    joined = "\n".join(messages)
    assert_no_secret(joined)
    assert "/elsewhere.mp4" in joined


def test_an_unsigned_source_still_logs_its_url_in_full(tmp_path: Path):
    """No collateral damage: redaction must not blank out ordinary sources."""
    plan = MediaPlan()
    with serving(handler_for(plan)) as base_url:
        plain = f"{base_url}/object-storage/web_videos/9d3c1f52-720.mp4"
        core = make_core()
        messages, error = run_and_collect(core, http_config(tmp_path, plain))

    assert error is None
    assert plain in "\n".join(messages), "an unsigned URL should log exactly as before"


def test_no_progressive_message_interpolates_a_raw_url_variable():
    """Guard the source: a new log line must not reintroduce the leak.

    The transport binds one redacted `logged_url` per attempt and per download.
    Any message formatting `current_url` or `source_url` directly is a line that
    would print a signature the moment a provider signs its URLs.
    """
    source = "\n".join(
        inspect.getsource(function)
        for function in (BaseCore.progressive_download, BaseCore._progressive_attempt)
    )

    assert "{current_url}" not in source
    assert "{source_url}" not in source
    assert ", current_url," not in source
    assert ", source_url," not in source
    assert "logged_url" in source
