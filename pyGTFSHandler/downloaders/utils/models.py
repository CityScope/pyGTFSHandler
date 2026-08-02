# -*- coding: utf-8 -*-
"""Source-agnostic feed metadata shared across all downloaders.

Every GTFS catalog API (Mobility Database, Transitland, Spain's NAP, ...)
describes feeds with its own JSON shape and field names. To let
`downloaders.utils.download.download_feeds` and any other shared code work
identically regardless of the source, each downloader's `search_feeds()`
method maps the raw API response into a list of `GTFSFeedMetadata`
instances defined here. The `raw` field keeps the original API payload
available for source-specific needs without polluting the common shape.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GTFSFeedMetadata:
    """A single GTFS feed's metadata, normalized across data sources.

    Attributes:
        id: Stable identifier of the feed within its source catalog
            (e.g. a Mobility Database feed id or a Transitland onestop_id).
        download_url: Direct URL to download the feed's GTFS static ZIP.
        name: Human-readable feed name, if the source provides one.
        provider: Name of the agency/operator that publishes the feed.
        country_code: ISO 3166-1 alpha-2 country code of the feed, if known.
        source: Short identifier of the catalog the feed came from
            (e.g. "mobility_database", "transitland", "nap_es").
        raw: The original, source-specific metadata dictionary, kept for
            callers that need fields beyond this common subset.
    """

    id: str
    download_url: str
    name: Optional[str] = None
    provider: Optional[str] = None
    country_code: Optional[str] = None
    source: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
