# -*- coding: utf-8 -*-
"""Downloader client for the Transitland REST API (transit.land).

Transitland (https://www.transit.land) is a global, crowd-sourced catalog
of GTFS and GTFS Realtime feeds, documented at
https://www.transit.land/documentation/rest-api. This module implements a
client for the v2 REST API's `/feeds` endpoint, following the same
`BaseGTFSDownloader` interface used by every other downloader in this
package.

Authentication uses an API key (obtained by signing up at transit.land),
sent as the `apikey` query parameter on every request, as required by the
API. It can be passed explicitly or read from the `TRANSITLAND_API_KEY`
environment variable.

`search_feeds` supports the subset of `/feeds` query parameters most
relevant to locating GTFS static feeds:

- `bbox` (derived from an AOI via `downloaders.utils.aoi.bbox_from_aoi`)
- `lat`/`lon`/`radius` for a point-radius search
- `country_code` (mapped to the API's `adm0_iso` filter)
- `state` (mapped to `adm1_iso` or `adm1_name`)
- `city` (mapped to `city_name`)
- `search` for a free-text feed/operator name search
- `spec`, fixed to `"gtfs"` so only GTFS static feeds are returned
  (Transitland also catalogs GTFS Realtime and other specs)

Since the API paginates results via a cursor (`meta.next`) rather than a
plain offset, `search_feeds` follows that cursor automatically until
`limit` feeds have been collected or the catalog is exhausted.

Each feed's static GTFS download URL is read from
`urls.static_current`, which Transitland documents as the feed's latest
published GTFS static ZIP.
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from ..utils.stack_gtfs import historic_stack
from .base import BaseGTFSDownloader
from .utils.aoi import AOIType, bbox_from_aoi
from .utils.dates import DateLike, normalize_date_range
from .utils.historic import (
    cleanup_version_paths,
    download_and_stitch_versions,
    select_versions_covering_range,
    zip_stitched_feed,
)
from .utils.http import request_json
from .utils.models import GTFSFeedMetadata
from .utils.naming import sanitize_filename

logger = logging.getLogger(__name__)


class TransitLandDownloader(BaseGTFSDownloader):
    """Client for searching and downloading feeds from Transitland.

    See https://www.transit.land/documentation/rest-api for the full API
    reference. Requires a Transitland API key.
    """

    BASE_URL = "https://transit.land/api/v2/rest"
    FEEDS_ENDPOINT = f"{BASE_URL}/feeds"

    API_KEY_ENV_VAR = "TRANSITLAND_API_KEY"
    SOURCE_NAME = "transitland"

    #: Maximum number of feeds the API returns per page.
    MAX_PAGE_SIZE = 100

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client.

        Args:
            api_key: Transitland API key. If not provided, it is read
                from the `TRANSITLAND_API_KEY` environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        super().__init__(api_key=api_key)
        if not self.api_key:
            raise ValueError(
                "A Transitland API key is required (pass api_key or set "
                f"{self.API_KEY_ENV_VAR})."
            )

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Issue an authenticated GET request against the Transitland API.

        Args:
            url: Full endpoint URL.
            params: Query parameters, excluding the API key.

        Returns:
            The decoded JSON response body.

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = self.api_key
        return request_json("GET", url, params=params)

    def search_feeds(
        self,
        aoi: Optional[AOIType] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius: Optional[float] = None,
        country_code: Optional[str] = None,
        state: Optional[str] = None,
        city: Optional[str] = None,
        search: Optional[str] = None,
        operator_onestop_id: Optional[str] = None,
        limit: Optional[int] = 1000,
        spec: str = "gtfs",
    ) -> List[GTFSFeedMetadata]:
        """Search Transitland's catalog for feeds matching filters.

        Args:
            aoi: Polygon, MultiPolygon, GeoDataFrame, or GeoSeries defining
                a geographic area; converted to a bounding box (`bbox`
                query parameter). Mutually exclusive with `lat`/`lon`.
            lat: Latitude for a point-radius search. Requires `lon`.
            lon: Longitude for a point-radius search. Requires `lat`.
            radius: Search radius in meters around `(lat, lon)`.
            country_code: ISO 3166-1 alpha-2 country code to filter by
                (Transitland's `adm0_iso`).
            state: State/province/subdivision name to filter by
                (Transitland's `adm1_name`).
            city: City name to filter by (Transitland's `city_name`).
            search: Free-text search over feed and operator names.
            operator_onestop_id: Restrict to feeds of a specific operator,
                identified by its Transitland onestop id.
            limit: Maximum number of feeds to return. Results are fetched
                page by page (up to `MAX_PAGE_SIZE` per request) following
                the API's cursor until this many feeds are collected or
                the catalog is exhausted.
            spec: GTFS specification to filter by. Defaults to `"gtfs"`
                (static GTFS); Transitland also catalogs other specs such
                as `"gtfs-rt"`.

        Returns:
            A list of `GTFSFeedMetadata`, one per matching feed with a
            usable static GTFS download URL.

        Raises:
            ValueError: If only one of `lat`/`lon` is given.
        """
        if (lat is None) != (lon is None):
            raise ValueError("Both 'lat' and 'lon' must be provided together.")

        params: Dict[str, Any] = {
            "spec": spec,
            "search": search,
            "adm0_iso": country_code,
            "adm1_name": state,
            "city_name": city,
            "operator_onestop_id": operator_onestop_id,
        }

        if aoi is not None:
            min_lon, min_lat, max_lon, max_lat = bbox_from_aoi(aoi)
            params["bbox"] = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        if lat is not None:
            params["lat"] = lat
            params["lon"] = lon
            if radius is not None:
                params["radius"] = radius

        raw_feeds = self._paginated_search(params, limit=limit)
        return [
            feed
            for feed in (self._to_feed_metadata(raw) for raw in raw_feeds)
            if feed is not None
        ]

    def _paginated_search(
        self, params: Dict[str, Any], limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Fetch feed pages by following the API's cursor until exhausted.

        Args:
            params: Query parameters shared by every page.
            limit: Maximum number of feeds to collect, or `None` for as
                many as the catalog returns.

        Returns:
            A list of raw feed dictionaries.
        """
        results: List[Dict[str, Any]] = []
        url = self.FEEDS_ENDPOINT
        page_params = {**params, "limit": min(self.MAX_PAGE_SIZE, limit or self.MAX_PAGE_SIZE)}

        while url is not None:
            try:
                data = self._get(url, page_params)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Transitland API request failed: {e}")
                break

            feeds = data.get("feeds", [])
            results.extend(feeds)

            if limit is not None and len(results) >= limit:
                return results[:limit]

            url = (data.get("meta") or {}).get("next")
            page_params = {}  # the "next" URL already embeds all query params

        return results

    def _to_feed_metadata(self, raw: Dict[str, Any]) -> Optional[GTFSFeedMetadata]:
        """Convert a raw Transitland feed dict to `GTFSFeedMetadata`.

        Args:
            raw: A single feed dictionary from the `/feeds` response.

        Returns:
            The corresponding `GTFSFeedMetadata`, or `None` if the feed
            has no usable static GTFS download URL.
        """
        urls = raw.get("urls") or {}
        download_url = urls.get("static_current")
        if not download_url:
            logger.warning(
                f"Feed '{raw.get('onestop_id')}' has no 'urls.static_current'. Skipping."
            )
            return None

        operators = raw.get("operators") or []
        provider = operators[0].get("name") if operators else None

        return GTFSFeedMetadata(
            id=str(raw.get("onestop_id", "")),
            download_url=download_url,
            name=raw.get("name") or raw.get("onestop_id"),
            provider=provider,
            country_code=None,
            source=self.SOURCE_NAME,
            raw=raw,
        )

    # -------------------------------------------------------------------
    # Historic-version stitching
    # -------------------------------------------------------------------

    @staticmethod
    def _version_start(version: Dict[str, Any]) -> Optional[datetime]:
        d = version.get("earliest_calendar_date")
        return datetime.strptime(d, "%Y-%m-%d") if d else None

    @staticmethod
    def _version_end(version: Dict[str, Any]) -> Optional[datetime]:
        d = version.get("latest_calendar_date")
        return datetime.strptime(d, "%Y-%m-%d") if d else None

    def find_feed_version_history(self, feed_key: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """List every published static GTFS version recorded for a feed.

        Uses `GET /feeds/{feed_key}/feed_versions`, which returns
        Transitland's full archive of past fetches for the feed, each
        tagged with `earliest_calendar_date`/`latest_calendar_date` -- the
        real service period Transitland computed from that version's own
        `calendar.txt`/`calendar_dates.txt`, not just when it was fetched
        (`fetched_at`).

        Args:
            feed_key: Feed lookup key: an integer id or onestop id.
            limit: Maximum number of feed versions to fetch.

        Returns:
            Raw feed_version dictionaries, sorted oldest to newest by
            `earliest_calendar_date` (entries missing that field sort
            last, in API order).
        """
        data = self._get(f"{self.BASE_URL}/feeds/{feed_key}/feed_versions", {"limit": limit})
        versions = data.get("feed_versions", [])
        versions.sort(key=lambda v: self._version_start(v) or datetime.max)
        return versions

    def _download_feed_version(self, version: Dict[str, Any], dest_zip_path: str) -> bool:
        """Download a single feed_version's zip to `dest_zip_path`.

        Downloading a *historic* (non-latest) feed_version requires a
        paid Transitland Professional/Enterprise plan (or a free
        Hobbyist/Academic plan); this surfaces that as a clear
        `PermissionError` rather than a generic HTTP failure.
        """
        sha1 = version.get("sha1")
        url = f"{self.BASE_URL}/feed_versions/{sha1}/download"
        try:
            response = requests.get(
                url, params={"apikey": self.api_key}, stream=True, timeout=60
            )
            if response.status_code in (401, 402, 403):
                raise PermissionError(
                    f"Transitland denied downloading historic feed_version '{sha1}' "
                    f"(HTTP {response.status_code}). Downloading historic (non-latest) "
                    "feed versions requires a paid Transitland Professional/Enterprise "
                    "plan, or a free Hobbyist/Academic plan."
                )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error downloading feed_version '{sha1}': {e}")
            return False

        with open(dest_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True

    def download_historic_stack(
        self,
        output_path: str,
        feed_key: str,
        start_date: DateLike,
        end_date: DateLike,
        day_separation: int = 1,
        overwrite: bool = False,
        aoi: Optional[AOIType] = None,
    ) -> Optional[str]:
        """Download and stitch a feed's version history into one GTFS zip.

        Mirrors `NAPDownloader.download_historic_stack`/
        `MobilityDatabaseDownloader.download_historic_stack`, using
        `find_feed_version_history`'s `earliest_calendar_date` as the
        trim anchor -- already a real, Transitland-computed service start
        date, unlike NAP's raw publication timestamp.

        Note: downloading any feed_version other than the single most
        recent one requires a paid (or free Hobbyist/Academic) Transitland
        plan; see `_download_feed_version`.

        Args:
            output_path: Directory to assemble the stitched feed in.
            feed_key: Feed lookup key: an integer id or onestop id.
            start_date: Start of the date range to cover. Accepts
                `"today"`, a `date`/`datetime`, or an ISO `"YYYY-MM-DD"`
                string.
            end_date: End of the date range to cover.
            day_separation: Minimum number of days a version is assumed
                to stay valid for, if service periods don't force it
                shorter.
            overwrite: If True, redo the feed even if its stitched output
                zip already exists.
            aoi: Optional AOI passed through to `historic_stack` to
                restrict stops.

        Returns:
            Path to the written `{SOURCE_NAME}_{feed_key}_{start}_{end}.zip`,
            or `None` if no feed versions were found covering the
            requested range.
        """
        os.makedirs(output_path, exist_ok=True)
        start_date, end_date = normalize_date_range(start_date, end_date)
        main_name = sanitize_filename(str(feed_key))
        zip_source_name = f"{self.SOURCE_NAME}_{main_name}"

        final_zip = os.path.join(
            output_path,
            f"{zip_source_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.zip",
        )
        if not overwrite and os.path.isfile(final_zip):
            logger.info(f"'{final_zip}' already exists. Skipping.")
            return final_zip

        history = self.find_feed_version_history(feed_key)
        versions = select_versions_covering_range(
            history, start_date, end_date, get_start=self._version_start
        )
        for version in versions:
            version["start_date"] = self._version_start(version)

        if not versions:
            logger.warning(f"No feed versions found for feed '{feed_key}' in the requested range.")
            return None

        main_path = os.path.normpath(os.path.join(output_path, main_name))
        path_stack = download_and_stitch_versions(
            versions, main_path, day_separation, end_date, overwrite, self._download_feed_version
        )
        if not path_stack:
            return None

        if os.path.isfile(path_stack[-1] + ".zip"):
            os.remove(path_stack[-1] + ".zip")

        historic_stack(path_stack, main_path, aoi)
        zip_path = zip_stitched_feed(main_path, zip_source_name, start_date, end_date, output_path)
        logger.info(f"Finished stitching historic feed '{main_name}'.")

        cleanup_version_paths(path_stack)
        shutil.rmtree(main_path)

        return zip_path
