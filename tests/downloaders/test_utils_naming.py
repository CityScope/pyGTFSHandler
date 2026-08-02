"""Tests for `pyGTFSHandler.downloaders.utils.naming`."""

from pyGTFSHandler.downloaders.utils.naming import (
    build_feed_filename,
    normalize_text,
    sanitize_filename,
)


def test_normalize_text_lowercases_and_strips_accents():
    assert normalize_text("Ámbito Público") == "ambito publico"


def test_normalize_text_strips_whitespace():
    assert normalize_text("  Hello  ") == "hello"


def test_sanitize_filename_replaces_unsafe_characters():
    assert sanitize_filename("Línea 12/34: Centro") == "linea_12_34__centro"


def test_sanitize_filename_keeps_safe_characters():
    assert sanitize_filename("abc-123_XYZ") == "abc-123_xyz"


def test_build_feed_filename_combines_truncated_parts():
    filename = build_feed_filename(feed_id="mdb-123456", name="Metro Line", provider="City Bus")
    # each component is truncated to 10 chars before sanitization
    parts = filename.split("_", 2)
    assert parts[0] == sanitize_filename("mdb-123456"[:10])
    assert filename.count("_") >= 2


def test_build_feed_filename_skips_empty_components():
    filename = build_feed_filename(feed_id="", name="OnlyName", provider="")
    assert filename == sanitize_filename("OnlyName"[:10])


def test_build_feed_filename_all_empty_returns_empty_string():
    assert build_feed_filename() == ""
