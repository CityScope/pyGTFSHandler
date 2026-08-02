# -*- coding: utf-8 -*-
"""Downloader client for the Mobility Database (api.mobilitydatabase.org).

The Mobility Database (https://mobilitydatabase.org) is a global, crowd
-sourced catalog of GTFS (and GTFS Realtime) feeds. Its REST API uses
OAuth2-style bearer tokens obtained from a long-lived refresh token: every
authorized request first ensures a fresh access token
(`MobilityDatabaseDownloader._ensure_access_token`), refreshing it
automatically when it is close to expiry.

Feed search (`search_feeds`) supports the API's filters (provider,
producer URL, country code, subdivision, municipality, official status,
bounding box) plus multi-value filters and pagination beyond the API's
per-request page-size limit, which are handled client-side:

- Multiple values for a single filter (e.g. several `country_code`s) are
  expanded into the cartesian product of per-filter API calls, with
  results de-duplicated by feed id.
- A `None` entry inside a filter list requests feeds where that field is
  *absent* from the catalog; since the API itself has no "is null" filter,
  this is implemented as a second pass that fetches the unfiltered result
  set and keeps only feeds whose field is missing.
- Requests for more than the API's per-page maximum are paginated
  automatically by repeating the search with increasing offsets.

Results are returned as a list of `downloaders.utils.models.GTFSFeedMetadata`,
so they can be downloaded via the shared `download_feeds()` implementation
from `downloaders.base.BaseGTFSDownloader`.
"""

import itertools
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from .base import BaseGTFSDownloader
from .utils.aoi import AOIType, bbox_from_aoi
from .utils.http import request_json
from .utils.models import GTFSFeedMetadata

logger = logging.getLogger(__name__)


class MobilityDatabaseDownloader(BaseGTFSDownloader):
    """Client for searching and downloading feeds from the Mobility Database.

    Authenticates via OAuth2 using a long-lived refresh token (obtained
    from the Mobility Database website), which is exchanged for short-lived
    access tokens as needed.
    """

    BASE_URL = "https://api.mobilitydatabase.org/v1"
    TOKEN_ENDPOINT = f"{BASE_URL}/tokens"
    GTFS_FEEDS_ENDPOINT = f"{BASE_URL}/gtfs_feeds"

    #: The Mobility Database uses a refresh token, not treated as a plain
    #: static API key; kept here only so `search_feeds` error messages and
    #: `BaseGTFSDownloader` stay consistent with other downloaders.
    API_KEY_ENV_VAR = "MOBILITY_DATABASE_REFRESH_TOKEN"
    SOURCE_NAME = "mobility_database"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client and obtain an initial access token.

        Args:
            api_key: Long-lived refresh token obtained from the Mobility
                Database website. If not provided, it is read from the
                `MOBILITY_DATABASE_REFRESH_TOKEN` environment variable.

        Raises:
            ValueError: If no refresh token is available.
        """
        super().__init__(api_key=api_key)
        if not self.api_key:
            raise ValueError(
                "A refresh token is required (pass api_key or set "
                f"{self.API_KEY_ENV_VAR})."
            )

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._ensure_access_token()

    # -------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------

    def _ensure_access_token(self) -> None:
        """Obtain or refresh the access token using the refresh token.

        Access tokens typically expire after one hour; this refreshes
        automatically whenever the cached token is missing or is within
        five minutes of expiry.

        Raises:
            ConnectionError: If the token request fails.
            ValueError: If the token response has no access token.
        """
        if (
            self._access_token
            and self._token_expires_at
            and datetime.now() < self._token_expires_at - timedelta(minutes=5)
        ):
            return

        logger.info("Obtaining or refreshing Mobility Database access token...")
        try:
            token_data = request_json(
                "POST",
                self.TOKEN_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json={"refresh_token": self.api_key},
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to obtain access token: {e}") from e

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("Access token missing in token response.")

        self._access_token = access_token
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        logger.info("Access token obtained successfully.")

    def _authorized_request(self, method: str, url: str, **kwargs) -> Any:
        """Perform an authorized request, refreshing the token if needed.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Target endpoint URL.
            **kwargs: Additional arguments forwarded to
                `downloaders.utils.http.request_json`.

        Returns:
            The decoded JSON response body.
        """
        self._ensure_access_token()
        headers = kwargs.pop("headers", {}) or {}
        headers["Authorization"] = f"Bearer {self._access_token}"
        return request_json(method, url, headers=headers, **kwargs)

    # -------------------------------------------------------------------
    # Parameter preparation
    # -------------------------------------------------------------------

    @staticmethod
    def _prepare_list_param(
        param_name: str, value: Optional[Union[str, List[Optional[str]]]]
    ) -> Tuple[Optional[List[str]], bool]:
        """Normalize a filter value into a list of strings plus a None-flag.

        Args:
            param_name: Parameter name, used only for error messages.
            value: A single string, a list of strings (optionally
                containing `None`), or `None`.

        Returns:
            A tuple `(values, include_none)` where `values` is the list
            of non-`None` string values (or `None` if there are none) and
            `include_none` indicates whether `None` was present in `value`.

        Raises:
            TypeError: If `value` (or one of its items) is not a string,
                `None`, or a list of those.
        """
        if value is None:
            return None, False
        if isinstance(value, str):
            return [value], False
        if isinstance(value, list):
            valid_items = []
            has_none = False
            for item in value:
                if item is None:
                    has_none = True
                elif isinstance(item, str):
                    valid_items.append(item)
                else:
                    raise TypeError(
                        f"All items in '{param_name}' must be strings or None; got {type(item)}"
                    )
            return valid_items or None, has_none
        raise TypeError(
            f"Expected string, list[str|None], or None for '{param_name}', got {type(value)}"
        )

    # -------------------------------------------------------------------
    # Search feeds
    # -------------------------------------------------------------------

    def _search_feeds_page(
        self,
        aoi: Optional[AOIType] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        provider: Optional[Union[str, List[Optional[str]]]] = None,
        producer_url: Optional[Union[str, List[Optional[str]]]] = None,
        country_code: Optional[Union[str, List[Optional[str]]]] = None,
        subdivision_name: Optional[Union[str, List[Optional[str]]]] = None,
        municipality: Optional[Union[str, List[Optional[str]]]] = None,
        bounding_filter_method: str = "partially_enclosed",
        is_official: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Perform a single (non-paginated) search request.

        Handles combining multi-value filters into per-request parameter
        combinations and merging/de-duplicating their results, including a
        second pass for filter values that requested `None` (missing
        field) matches.

        Returns:
            A list of raw feed dictionaries, as returned by the API.
        """
        param_requests = {
            "provider": self._prepare_list_param("provider", provider),
            "producer_url": self._prepare_list_param("producer_url", producer_url),
            "country_code": self._prepare_list_param("country_code", country_code),
            "subdivision_name": self._prepare_list_param("subdivision_name", subdivision_name),
            "municipality": self._prepare_list_param("municipality", municipality),
        }
        needs_none_pass = any(flag for _, flag in param_requests.values())

        base_params: Dict[str, Any] = {}
        if limit is not None:
            base_params["limit"] = str(limit)
        if offset is not None:
            if offset < 0:
                raise ValueError("Offset cannot be negative.")
            base_params["offset"] = str(offset)
        if is_official is not None:
            base_params["is_official"] = str(is_official).lower()

        if aoi is not None:
            min_lon, min_lat, max_lon, max_lat = bbox_from_aoi(aoi)
            base_params["dataset_latitudes"] = f"{min_lat},{max_lat}"
            base_params["dataset_longitudes"] = f"{min_lon},{max_lon}"

            valid_methods = ["completely_enclosed", "partially_enclosed", "disjoint"]
            if bounding_filter_method not in valid_methods:
                raise ValueError(f"Invalid bounding_filter_method: {bounding_filter_method}")
            base_params["bounding_filter_method"] = bounding_filter_method

        non_none_values = {k: v[0] for k, v in param_requests.items() if v[0]}
        keys, lists = zip(*non_none_values.items()) if non_none_values else ([], [])

        all_feed_ids = set()
        results: List[Dict[str, Any]] = []

        for combo in itertools.product(*lists):
            params = {**base_params, **dict(zip(keys, combo))}
            try:
                data = self._authorized_request("GET", self.GTFS_FEEDS_ENDPOINT, params=params)
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed for params {params}: {e}")
                continue

            for feed in data:
                feed_id = feed.get("id")
                if feed_id and feed_id not in all_feed_ids:
                    results.append(feed)
                    all_feed_ids.add(feed_id)

        if needs_none_pass:
            omitted_values = {k: v[0] for k, v in param_requests.items() if v[0] and not v[1]}
            keys_o, lists_o = zip(*omitted_values.items()) if omitted_values else ([], [])

            for combo in itertools.product(*lists_o) if lists_o else [()]:
                params = {**base_params, **dict(zip(keys_o, combo))}
                try:
                    data = self._authorized_request("GET", self.GTFS_FEEDS_ENDPOINT, params=params)
                except requests.exceptions.RequestException as e:
                    logger.warning(f"API request failed for params {params}: {e}")
                    continue

                for feed in data:
                    feed_id = feed.get("id")
                    if not feed_id or feed_id in all_feed_ids:
                        continue

                    matches_none = True
                    for param_name, (_, include_none_flag) in param_requests.items():
                        if include_none_flag:
                            if param_name in ("provider", "producer_url"):
                                field_value = feed.get(param_name)
                            else:
                                locs = feed.get("locations")
                                field_value = locs[0].get(param_name) if locs else None
                            if field_value is not None:
                                matches_none = False
                                break

                    if matches_none:
                        results.append(feed)
                        all_feed_ids.add(feed_id)

        return results

    def search_feeds(
        self,
        aoi: Optional[AOIType] = None,
        limit: Optional[int] = 100_000,
        offset: Optional[int] = None,
        provider: Optional[Union[str, List[Optional[str]]]] = None,
        producer_url: Optional[Union[str, List[Optional[str]]]] = None,
        country_code: Optional[Union[str, List[Optional[str]]]] = None,
        subdivision_name: Optional[Union[str, List[Optional[str]]]] = None,
        municipality: Optional[Union[str, List[Optional[str]]]] = None,
        bounding_filter_method: str = "partially_enclosed",
        is_official: Optional[bool] = None,
    ) -> List[GTFSFeedMetadata]:
        """Search the Mobility Database for GTFS feeds matching filters.

        Supports multiple values per filter (client-side merged across
        combinations) and automatic pagination for `limit` values above
        the API's per-request page size (200).

        Args:
            aoi: Polygon, MultiPolygon, GeoDataFrame, or GeoSeries defining
                a geographic area to filter by.
            limit: Maximum number of feeds to return. Values above 200
                trigger automatic multi-page fetching.
            offset: Offset for pagination.
            provider: Provider name filter. Accepts a string, a list of
                strings, or a list including `None` to also match feeds
                with no provider set.
            producer_url: Producer URL filter, same semantics as `provider`.
            country_code: Country code filter, same semantics as `provider`.
            subdivision_name: Subdivision name filter, same semantics as
                `provider`.
            municipality: Municipality filter, same semantics as `provider`.
            bounding_filter_method: Spatial inclusion rule when `aoi` is
                set. One of "completely_enclosed", "partially_enclosed",
                "disjoint".
            is_official: If set, filter by whether the feed is official.

        Returns:
            A list of `GTFSFeedMetadata`, one per matching feed with a
            usable download URL.
        """
        MAX_PAGE = 200
        kwargs = dict(
            aoi=aoi,
            provider=provider,
            producer_url=producer_url,
            country_code=country_code,
            subdivision_name=subdivision_name,
            municipality=municipality,
            bounding_filter_method=bounding_filter_method,
            is_official=is_official,
        )

        if limit is None:
            limit = MAX_PAGE

        if limit <= MAX_PAGE:
            raw_feeds = self._search_feeds_page(limit=limit, offset=offset, **kwargs)
        else:
            raw_feeds = []
            used_ids = set()
            remaining = limit
            current_offset = offset if offset is not None else 0

            while remaining > 0:
                page_limit = min(MAX_PAGE, remaining)
                page_results = self._search_feeds_page(
                    limit=page_limit, offset=current_offset, **kwargs
                )
                if not page_results:
                    break

                for feed in page_results:
                    fid = feed.get("id")
                    if fid and fid not in used_ids:
                        raw_feeds.append(feed)
                        used_ids.add(fid)

                remaining -= page_limit
                current_offset += page_limit
                if len(page_results) < page_limit:
                    break

        return [
            feed
            for feed in (self._to_feed_metadata(raw) for raw in raw_feeds)
            if feed is not None
        ]

    def _to_feed_metadata(self, raw: Dict[str, Any]) -> Optional[GTFSFeedMetadata]:
        """Convert a raw Mobility Database feed dict to `GTFSFeedMetadata`.

        Args:
            raw: A single feed dictionary from the search API response.

        Returns:
            The corresponding `GTFSFeedMetadata`, or `None` if the feed
            has no usable download URL.
        """
        latest_dataset = raw.get("latest_dataset") or {}
        hosted_url = latest_dataset.get("hosted_url")
        if not hosted_url:
            logger.warning(f"Feed '{raw.get('id')}' has no 'latest_dataset.hosted_url'. Skipping.")
            return None

        locations = raw.get("locations") or []
        country_code = locations[0].get("country_code") if locations else None

        return GTFSFeedMetadata(
            id=str(raw.get("id", "")),
            download_url=hosted_url,
            name=raw.get("feed_name"),
            provider=raw.get("provider"),
            country_code=country_code,
            source=self.SOURCE_NAME,
            raw=raw,
        )
