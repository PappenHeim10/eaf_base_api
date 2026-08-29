"""Tests for the canonical title -> filename sanitizer.

`strip_title` is the single sanitization boundary between untrusted remote media
titles and the filesystem. Everything here is a regression guard for that
contract, so provider packages and applications can rely on it instead of
inventing their own escaping.
"""

from pathlib import Path

import pytest

from base_api.modules.static_functions import strip_title


def test_ordinary_title_is_left_alone() -> None:
    assert strip_title("Ein ganz normaler Titel") == "Ein ganz normaler Titel"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Why: does this work?", "Why_ does this work_"),
        ("star*name", "star_name"),
        ('quote"name', "quote_name"),
        ("pipe|name", "pipe_name"),
        ("angle<brackets>", "angle_brackets_"),
    ],
)
def test_windows_illegal_characters_become_underscores(raw: str, expected: str) -> None:
    assert strip_title(raw) == expected


def test_control_characters_become_underscores() -> None:
    assert strip_title("a\x00b\x1fc") == "a_b_c"


def test_zero_width_characters_are_removed() -> None:
    assert strip_title("a\u200bb\ufeffc") == "abc"


def test_unicode_is_normalised_before_filtering() -> None:
    # NFKC turns the full-width solidus into a plain "/", which must then be
    # filtered like any other separator rather than surviving into the name.
    assert "/" not in strip_title("a\uff0fb")


@pytest.mark.parametrize("raw", ["CON", "NUL", "COM1", "LPT9"])
def test_windows_reserved_names_are_escaped(raw: str) -> None:
    assert strip_title(raw) == f"_{raw}"


def test_reserved_name_with_extension_is_escaped() -> None:
    assert strip_title("NUL.mp4") == "_NUL.mp4"


@pytest.mark.parametrize(
    "raw",
    # Backslash-only inputs are deliberately absent: PurePath treats "\" as a
    # separator on Windows but as an ordinary character elsewhere, so the result
    # is platform-dependent. The path-escape tests below cover them instead, via
    # an invariant that holds on every platform.
    ["", "..", ".", "///", "   ", "..."],
)
def test_titles_that_sanitise_to_nothing_fall_back(raw: str) -> None:
    assert strip_title(raw) == "untitled"


def test_fallback_name_is_configurable() -> None:
    assert strip_title("", default_name="fallback") == "fallback"


# --- length -----------------------------------------------------------------


def test_long_ascii_title_fits_the_byte_budget() -> None:
    result = strip_title("x" * 300)
    assert len(result.encode("utf-8")) <= 245


def test_long_multibyte_title_is_capped_by_bytes_not_characters() -> None:
    # Regression: the character cap alone let 255 CJK characters through as 765
    # bytes, which no ext4/Android filesystem accepts.
    result = strip_title("\u6f22" * 300)
    assert len(result.encode("utf-8")) <= 245


def test_truncation_never_splits_a_multibyte_character() -> None:
    # A naive byte slice would cut a 3-byte character in half and leave a lone
    # continuation byte behind.
    result = strip_title("\u6f22" * 300)
    assert set(result) == {"\u6f22"}
    assert result.encode("utf-8").decode("utf-8") == result


def test_sanitised_name_plus_extension_stays_within_the_filesystem_limit() -> None:
    for raw in ("x" * 300, "\u6f22" * 300, "a" * 254):
        filename = f"{strip_title(raw)}.mp4"
        assert len(filename.encode("utf-8")) <= 255


# --- path escape ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "../outside",
        r"..\outside",
        "../../../../etc/passwd",
        "/etc/passwd",
        r"C:\Windows\System32\evil",
        r"\\server\share\evil",
        "..",
        "sub/dir/file",
        r"sub\dir\file",
    ],
)
def test_a_remote_title_cannot_escape_the_output_directory(tmp_path: Path, raw: str) -> None:
    # Resolved with real path semantics, not a string-prefix comparison.
    destination = (tmp_path / f"{strip_title(raw)}.mp4").resolve()
    assert destination.is_relative_to(tmp_path.resolve())
    assert destination.parent == tmp_path.resolve()


@pytest.mark.parametrize("raw", ["../outside", "sub/dir/file", r"C:\Windows\evil"])
def test_sanitised_title_carries_no_separators(raw: str) -> None:
    sanitised = strip_title(raw)
    assert "/" not in sanitised
    assert "\\" not in sanitised
