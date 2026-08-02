# -*- coding: utf-8 -*-
"""Filename and text normalization shared by every downloader.

Feed/provider names returned by GTFS catalog APIs are arbitrary,
API-supplied strings (accents, spaces, punctuation, mixed case) that are
not safe to use directly as filenames on every platform. `normalize_text`
and `sanitize_filename` give every downloader one consistent way to turn
such strings into filesystem-safe names, and `build_feed_filename` builds
the actual on-disk file stem from a feed's id/name/provider.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase a string and strip diacritics/accents.

    Args:
        text: Input text.

    Returns:
        The lowercased, accent-stripped text.
    """
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def sanitize_filename(name: str) -> str:
    """Replace characters unsafe for filenames with underscores.

    Args:
        name: Raw string to sanitize (e.g. a feed or provider name).

    Returns:
        A filesystem-safe string containing only letters, digits,
        underscores, and hyphens.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", normalize_text(name))


def build_feed_filename(
    feed_id: str = "", name: str = "", provider: str = "", max_chars: int = 10
) -> str:
    """Build a filesystem-safe filename stem identifying a feed.

    Combines a truncated id, name, and provider so files stay short but
    distinguishable, e.g. "mdb-123_Metro_CityBus".

    Args:
        feed_id: Feed identifier from its source catalog.
        name: Feed or dataset name.
        provider: Agency/operator name.
        max_chars: Maximum number of characters kept from each component.

    Returns:
        A sanitized filename stem built from the non-empty components,
        joined with underscores.
    """
    parts = [
        sanitize_filename(str(value)[:max_chars])
        for value in (feed_id, name, provider)
        if value
    ]
    return "_".join(parts)
