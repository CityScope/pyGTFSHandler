# -*- coding: utf-8 -*-
"""Downloader client for BusMaps' Public Transit API GTFS catalog.

BusMaps (https://busmaps.com) exposes a global, multi-country transit API
(base URL `https://capi.busmaps.com:8443`, documented at
https://capi.busmaps.com:8443/docs or via the MCP server at
https://mcp.busmaps.com/mcp) covering routing, real-time vehicle
positions, timetables, and a GTFS feed catalog across 90+ countries. This
module implements a client for its `GET /getGtfsFeedsDownloads` endpoint,
which is the only endpoint of the API that returns downloadable GTFS
static data -- the other endpoints (`/routes`, `/nextDepartures`,
`/stopsInRadius`, `/line`, `/trip`, `/rawVehiclePositions`, ...) serve
live routing/timetable/vehicle-position queries, not GTFS files, and are
out of scope for this GTFS-catalog downloader.

Authentication is header-based rather than the query-param style used by
`downloaders.transitland`: every request must carry both a `capi-key`
header (`"Bearer <API_KEY>"`) and a `capi-host` header identifying which
underlying data platform to query -- `"busmaps.com"` (official data,
the default here) or `"wikiroutes.info"` (crowdsourced; not exposed via
`getGtfsFeedsDownloads`, which is a busmaps.com-only endpoint per the
API docs, but the header is kept configurable for forward-compatibility
with other endpoints of this same API).

`getGtfsFeedsDownloads` is a *global* feed catalog -- calling it with no
filters returns every feed BusMaps has for every country (~10 MB
response, per the API docs); `countryIso` and/or `feedName` narrow that
down. Each matching feed lists several `derivatives` (processed/source
GTFS zips, validation reports, geojson, ...); `search_feeds` picks the
best available GTFS zip per feed -- preferring the `improved_gtfs.zip`
processed derivative over the raw `source_gtfs.zip` -- and returns one
`GTFSFeedMetadata` per feed, following the same `BaseGTFSDownloader`
interface as every other downloader in this package so its results slot
into the shared `download_feeds()` loop unchanged.

Rate limits (free tier, per the API docs): `getGtfsFeedsDownloads` is
capped at 50 requests/day and 1,000/month -- much tighter than this
package's other catalog downloaders -- since one call already returns an
entire country's (or the entire world's) feed catalog in one shot.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from .base import BaseGTFSDownloader
from .utils.http import request_json
from .utils.models import GTFSFeedMetadata

logger = logging.getLogger(__name__)

#: Derivative `fileName`s that are valid standalone GTFS static feeds,
#: in preference order: the "improved" (BusMaps-cleaned) GTFS first,
#: falling back to the untouched source GTFS if no improved derivative
#: is published for a feed.
DEFAULT_GTFS_DERIVATIVE_NAMES = ("improved_gtfs.zip", "source_gtfs.zip")


class BusMapsDownloader(BaseGTFSDownloader):
    """Client for BusMaps' GTFS feed catalog (`/getGtfsFeedsDownloads`).

    See https://capi.busmaps.com:8443 (docs linked from https://busmaps.com)
    for the full API reference. Requires a BusMaps API key.
    """

    BASE_URL = "https://capi.busmaps.com:8443"
    GTFS_FEEDS_ENDPOINT = f"{BASE_URL}/getGtfsFeedsDownloads"

    API_KEY_ENV_VAR = "BUSMAPS_API_KEY"
    SOURCE_NAME = "busmaps"

    #: Default `capi-host` header value. The other documented option,
    #: `"wikiroutes.info"`, does not serve `getGtfsFeedsDownloads` but is
    #: accepted here for symmetry with the rest of the BusMaps API.
    DEFAULT_CAPI_HOST = "busmaps.com"

    def __init__(self, api_key: Optional[str] = None, capi_host: str = DEFAULT_CAPI_HOST):
        """Initialize the client.

        Args:
            api_key: BusMaps API key. If not provided, it is resolved via
                `downloaders.utils.config.get_api_key`: the
                `BUSMAPS_API_KEY` environment variable, then the
                `"busmaps"` entry of the local API keys file.
            capi_host: Value sent as the `capi-host` header, selecting
                which underlying platform's data to query.

        Raises:
            ValueError: If no API key is available.
        """
        super().__init__(api_key=api_key)
        if not self.api_key:
            raise ValueError(
                "A BusMaps API key is required (pass api_key, set "
                f"{self.API_KEY_ENV_VAR}, or add a 'busmaps' entry to the "
                "local API keys file)."
            )
        self.capi_host = capi_host

    def _headers(self) -> Dict[str, str]:
        """Build the `capi-key`/`capi-host` headers required by every request."""
        return {"capi-key": f"Bearer {self.api_key}", "capi-host": self.capi_host}

    def _get(self, params: Dict[str, Any]) -> Any:
        """Issue an authenticated GET against `getGtfsFeedsDownloads`.

        Args:
            params: Query parameters (`countryIso` and/or `feedName`),
                with `None` values dropped.

        Returns:
            The decoded JSON response: a list of per-country catalog
            entries, each with a `feeds` list (see module/API docs).

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        params = {k: v for k, v in params.items() if v is not None}
        return request_json(
            "GET", self.GTFS_FEEDS_ENDPOINT, headers=self._headers(), params=params
        )

    @staticmethod
    def _pick_derivative(
        derivatives: List[Dict[str, Any]], derivative_names: Sequence[str]
    ) -> Optional[Dict[str, Any]]:
        """Pick the best available GTFS-zip derivative from a feed's `derivatives`.

        Args:
            derivatives: A feed's `derivatives` list, as returned by the API.
            derivative_names: `fileName`s to look for, in preference order.

        Returns:
            The first matching derivative dict with a usable `path`, or
            `None` if none of `derivative_names` is present.
        """
        by_filename = {d.get("fileName"): d for d in derivatives if d.get("path")}
        for name in derivative_names:
            derivative = by_filename.get(name)
            if derivative is not None:
                return derivative
        return None

    def search_feeds(
        self,
        country_iso: Optional[str] = None,
        feed_name: Optional[str] = None,
        derivative_names: Sequence[str] = DEFAULT_GTFS_DERIVATIVE_NAMES,
    ) -> List[GTFSFeedMetadata]:
        """Search BusMaps' global GTFS catalog for downloadable feeds.

        This is a *global* downloader: it is not tied to any particular
        city or country. Passing neither filter returns every feed in
        BusMaps' worldwide catalog (a large, ~10 MB response per the API
        docs); in practice callers should nearly always pass at least
        `country_iso` to scope the request (and stay well under the
        endpoint's 50-requests/day free-tier limit).

        Args:
            country_iso: ISO 3166-1 alpha-3 country code to filter feeds
                by (e.g. `"CHL"`, `"USA"`, `"JPN"`).
            feed_name: Exact feed name to return a single feed and its
                derivatives (e.g. `"kotoden"`).
            derivative_names: Which `derivatives[].fileName` values count
                as a usable standalone GTFS zip, in preference order.
                Defaults to `DEFAULT_GTFS_DERIVATIVE_NAMES` (BusMaps'
                cleaned "improved" GTFS, falling back to the raw source
                GTFS).

        Returns:
            One `GTFSFeedMetadata` per feed that has a matching GTFS-zip
            derivative. `GTFSFeedMetadata.raw` holds the full feed,
            country, and chosen-derivative dicts for anything beyond the
            common fields (validation reports, route type breakdown,
            license, ...).
        """
        data = self._get({"countryIso": country_iso, "feedName": feed_name})
        results: List[GTFSFeedMetadata] = []
        for country in data or []:
            for feed in country.get("feeds", []) or []:
                derivative = self._pick_derivative(
                    feed.get("derivatives", []) or [], derivative_names
                )
                if derivative is None:
                    logger.warning(
                        f"Feed '{feed.get('feedName')}' has none of {list(derivative_names)} "
                        "among its derivatives. Skipping."
                    )
                    continue
                results.append(
                    GTFSFeedMetadata(
                        id=str(feed.get("feedId", feed.get("feedName", ""))),
                        download_url=derivative["path"],
                        name=feed.get("feedName"),
                        provider=country.get("countryName"),
                        country_code=country.get("countryIso"),
                        source=self.SOURCE_NAME,
                        raw={"feed": feed, "country": country, "derivative": derivative},
                    )
                )
        return results
