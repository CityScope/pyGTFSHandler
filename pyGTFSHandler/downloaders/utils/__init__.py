# -*- coding: utf-8 -*-
"""Shared helpers for the `downloaders` package.

Every module under `pyGTFSHandler.downloaders` implements a client for a
specific GTFS catalog/source (e.g. the Mobility Database, Transitland, or a
national access point such as Spain's NAP). Rather than each client
re-implementing the same low level plumbing, the pieces that are common to
*any* GTFS source live here:

- `config`: resolves each downloader's API key without ever hardcoding one
  in the package -- an explicit argument, an environment variable, or an
  optional local, gitignored `api_keys.json` file, in that priority order.
- `models`: the source-agnostic `GTFSFeedMetadata` dataclass that every
  downloader's `search_feeds()` returns. Having a single shared shape lets
  `download_feeds()` and any downstream code work the same way regardless of
  which catalog the feed came from.
- `http`: a small wrapper around `requests` (`request_json`) that applies a
  consistent timeout, error message, and JSON-decoding behaviour to every
  outgoing API call.
- `aoi`: converts the area-of-interest objects accepted throughout the
  package (Shapely `Polygon`/`MultiPolygon`, GeoPandas `GeoDataFrame`/
  `GeoSeries`) into a plain `(min_lon, min_lat, max_lon, max_lat)` bounding
  box, since most GTFS catalog APIs only support bounding-box filtering.
- `naming`: filename/string sanitization so that feed names coming from
  arbitrary external APIs can be safely used as filenames across platforms.
- `download`: the shared "download a list of `GTFSFeedMetadata` into a
  folder, optionally unzip, optionally skip/overwrite existing files" loop
  used by every downloader's `download_feeds()` implementation.

Subclasses of `pyGTFSHandler.downloaders.base.BaseGTFSDownloader` should
prefer these helpers over re-implementing equivalent logic, so behaviour
(retries, filename rules, AOI handling, etc.) stays uniform across sources.
"""

from . import aoi, config, download, http, naming
from .models import GTFSFeedMetadata

__all__ = ["aoi", "config", "download", "http", "naming", "GTFSFeedMetadata"]
