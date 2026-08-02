# -*- coding: utf-8 -*-
"""Clients for downloading GTFS feeds from various public catalogs.

Each submodule implements one GTFS source as a subclass of
`downloaders.base.BaseGTFSDownloader`, sharing a uniform interface:

- Construction takes an optional `api_key`, falling back to a
  source-specific environment variable (e.g. `NAP_API_KEY`) when omitted.
- `search_feeds(...)` returns a list of
  `downloaders.utils.models.GTFSFeedMetadata`, normalizing whatever shape
  that source's API returns feeds in.
- `download_feeds(feeds, download_folder, ...)` downloads (and optionally
  unzips) a list of `GTFSFeedMetadata`, via the shared implementation in
  `downloaders.utils.download`.

Available downloaders:

- `mobility_database.MobilityDatabaseDownloader`: the global Mobility
  Database catalog (https://mobilitydatabase.org).
- `transitland.TransitLandDownloader`: the global Transitland catalog
  (https://www.transit.land).
- `spain.NAPDownloader`: Spain's National Access Point
  (https://nap.transportes.gob.es), for country-specific sources that
  aren't covered by the global catalogs above. Other countries would get
  their own `downloaders/<country>/` package following the same pattern.

Helpers shared by more than one downloader (AOI-to-bbox conversion,
filename sanitization, the shared download loop, ...) live in
`downloaders.utils`; helpers specific to a single country's source(s) live
under that country's own package (e.g. `downloaders.spain.utils`).
"""

from . import spain
from .base import BaseGTFSDownloader
from .mobility_database import MobilityDatabaseDownloader
from .transitland import TransitLandDownloader

__all__ = [
    "BaseGTFSDownloader",
    "MobilityDatabaseDownloader",
    "TransitLandDownloader",
    "spain",
]
