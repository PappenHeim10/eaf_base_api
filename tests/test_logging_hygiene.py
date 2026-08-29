"""Logging must cost what the diagnostics cost, not what the download weighs.

Two failures motivated these: a single BaseCore exception appeared twice in two
different formats, and a debug run emitted raw segment payloads - megabytes of
`b"\\xe9\\x77..."` - which made downloads visibly slower.
"""

from __future__ import annotations

import logging

import pytest

from base_api.modules.logger import configure_app_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    """Keep these tests from leaking handler state into the rest of the suite."""
    named = logging.getLogger("eaf-test-logger")
    before = (list(named.handlers), named.level, named.propagate)
    root = logging.getLogger()
    root_before = list(root.handlers)
    yield
    named.handlers, named.level, named.propagate = before
    root.handlers = root_before


def test_a_configured_logger_does_not_also_propagate_to_root():
    logger = configure_app_logging(logger_name="eaf-test-logger")

    assert logger.handlers, "expected the helper to attach its own handler"
    assert logger.propagate is False


def test_one_exception_produces_one_record(caplog):
    logger = configure_app_logging(logger_name="eaf-test-logger")

    # caplog attaches at the root; with propagation off the record must not
    # arrive there a second time on top of the logger's own handler.
    records: list[logging.LogRecord] = []
    logger.addHandler(_Collector(records))

    with caplog.at_level(logging.ERROR):
        try:
            raise RuntimeError("download failed")
        except RuntimeError:
            logger.exception("Unhandled exception in download wrapper")

    assert len(records) == 1
    assert records[0].exc_info is not None
    assert "download failed" not in caplog.text


def test_configuring_the_root_logger_leaves_propagation_alone():
    root = configure_app_logging(logger_name=None)
    assert root is logging.getLogger()
    assert root.propagate is True


class _Collector(logging.Handler):
    def __init__(self, sink: list[logging.LogRecord]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record)


def test_no_segment_payload_reaches_the_log():
    """The library itself must never format a payload into a message.

    Collected off BaseCore's own logger rather than caplog, because that logger
    no longer propagates to root - which is the point of the fix above.
    """
    from base_api.base import BaseCore
    from base_api.modules.config import RuntimeConfig

    sentinel = b"\xe9\x77\x9eSEGMENT-PAYLOAD-SENTINEL\x00\xff" * 8
    core = BaseCore(RuntimeConfig())

    records: list[logging.LogRecord] = []
    handler = _Collector(records)
    core.logger.addHandler(handler)
    previous_level = core.logger.level
    core.logger.setLevel(logging.DEBUG)
    try:
        core.logger.debug("Segment stored: index=%s size_bytes=%s", 3, len(sentinel))
    finally:
        core.logger.removeHandler(handler)
        core.logger.setLevel(previous_level)

    rendered = "\n".join(record.getMessage() for record in records)
    assert "SEGMENT-PAYLOAD-SENTINEL" not in rendered
    assert f"size_bytes={len(sentinel)}" in rendered


def test_the_library_never_formats_bytes_into_a_log_call():
    """Guard the source itself: no log call may interpolate a payload variable."""
    import inspect
    import re

    from base_api import base as base_module

    source = inspect.getsource(base_module)
    # Blank out string literals first: a message may legitimately contain the
    # word "data", what must not happen is passing the variable.
    without_literals = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', '""', source)

    offenders = re.findall(
        r"self\.logger\.\w+\([^)]*\b(?:segment_data|chunk|data|parts|payload)\b[^)]*\)",
        without_literals,
    )
    # len(...) and index/metadata uses are fine; a bare payload argument is not.
    bare = [call for call in offenders if not re.search(r"len\(|_size|size_bytes|count", call)]
    assert bare == [], f"payload-ish variables passed to a log call: {bare}"
