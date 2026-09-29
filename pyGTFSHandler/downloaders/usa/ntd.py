# -*- coding: utf-8 -*-
"""Downloader client for the US National Transit Database (NTD) GTFS registry.

The Federal Transit Administration's NTD publishes a "General Transit Feed
Specification Weblinks" dataset (Socrata dataset id `2u7n-ub22`, hosted on
https://data.transportation.gov, itself the data portal for
https://www.transit.dot.gov/ntd) listing one GTFS weblink per reporting
transit agency/mode/service-type combination for all US fixed-route modes
FTA requires GTFS for. It's queried through Socrata's SODA API (SODA3 here,
since the dataset's endpoint is `api/v3/views/<id>/query.json`), which pages
results in blocks of up to 1,000 rows via `$offset`/`$limit` (SODA2-style
query params also work against the v3 endpoint and are simpler than SoQL, so
that's what `NTDDownloader` uses).

`NTDDownloader.search_feeds` returns one `GTFSFeedMetadata` per NTD row
(after dropping rows with no weblink or that are `waived`), with the row's
`mode_name`/`state`/`ntd_id`/etc. kept in `GTFSFeedMetadata.raw["ntd"]` for
downstream use — in particular `NTD_MODE_TO_SCORING_MODE`, which maps NTD's
15 `mode_name` values onto this project's 3 scoring modes (bus/tram/rail),
per the mapping agreed with the project owner.

`check_feed_links` is a separate validation pass (not part of
`search_feeds`) that HEAD/GETs every candidate weblink and reports which
ones actually resolve to a fetchable file, since NTD only guarantees the
weblink was valid whenever FTA last validated it, not that it still is.
"""

import logging
from typing import Dict, List, Optional, Union

import requests

from ..base import BaseGTFSDownloader
from ..utils.http import request_json
from ..utils.models import GTFSFeedMetadata
from .brt import apply_brt_overrides_for_feed

logger = logging.getLogger(__name__)

#: NTD's 15 fixed-route `mode_name` values, mapped to this project's 3
#: scoring modes. Repeats are intentional (e.g. commuter bus scores as
#: plain bus, streetcar scores the same as light rail).
NTD_MODE_TO_SCORING_MODE: Dict[str, str] = {
    "Bus": "bus",
    "Commuter Bus": "bus",
    "Trolleybus": "bus",
    "Ferryboat": "tram",
    "Streetcar Rail": "tram",
    "Bus Rapid Transit": "tram",
    "Light Rail": "tram",
    "Aerial Tramway": "tram",
    "Inclined Plane": "tram",
    "Cable Car": "tram",
    "Hybrid Rail": "rail",
    "Monorail/Automated Guideway": "rail",
    "Commuter Rail": "rail",
    "Alaska Railroad": "rail",
    "Heavy Rail": "rail",
}

#: One emoji per NTD mode, for the map's mode legend/icons. Emojis are
#: intentionally repeated across visually/functionally similar modes.
NTD_MODE_EMOJI: Dict[str, str] = {
    "Bus": "🚌",
    "Commuter Bus": "🚌",
    "Trolleybus": "🚎",
    "Ferryboat": "⛴️",
    "Streetcar Rail": "🚋",
    "Bus Rapid Transit": "🚍",
    "Light Rail": "🚈",
    "Aerial Tramway": "🚡",
    "Inclined Plane": "🚟",
    "Cable Car": "🚠",
    "Hybrid Rail": "🚆",
    "Monorail/Automated Guideway": "🚝",
    "Commuter Rail": "🚆",
    "Alaska Railroad": "🚂",
    "Heavy Rail": "🚇",
}

#: One emoji per GTFS `route_type`, for the map's per-route icon -- finer
#: grained than `NTD_MODE_EMOJI` (which is per NTD-reported agency/mode
#: row, not per actual route). Scoring still collapses everything to 3
#: buckets (bus/tram/rail) via `NTD_MODE_TO_SCORING_MODE`; this is display
#: only. Covers the basic GTFS enum (0-7) plus `brt.BRT_ROUTE_TYPE` (702,
#: this project's repurposed marker for Bus Rapid Transit -- see
#: `brt.py`).
ROUTE_TYPE_EMOJI: Dict[int, str] = {
    0: "🚊",  # Tram, Streetcar, Light rail
    1: "🚇",  # Subway, Metro
    2: "🚆",  # Rail (intercity/commuter)
    3: "🚌",  # Bus
    4: "⛴️",  # Ferry
    5: "🚋",  # Cable tram
    6: "🚡",  # Aerial lift / suspended cable car
    7: "🚞",  # Funicular
    702: "🚍",  # Bus Rapid Transit (this project's marker, see brt.py)
}


class NTDDownloader(BaseGTFSDownloader):
    """Client for searching and downloading feeds from the FTA's NTD registry.

    No API key is strictly required (Socrata allows unauthenticated
    requests at a lower throttling tier), but a Socrata app token is
    recommended for a 50-state crawl. See
    https://www.transit.dot.gov/ntd for the underlying program and
    https://data.transportation.gov/Public-Transit/General-Transit-Feed-Specification-Weblinks/2u7n-ub22
    for the dataset itself.
    """

    BASE_URL = "https://data.transportation.gov/api/v3/views/2u7n-ub22/query.json"
    API_KEY_ENV_VAR = "SOCRATA_APP_TOKEN"
    SOURCE_NAME = "ntd"

    #: Rows per page. This dataset's v3 endpoint ignores SODA2-style
    #: `$limit`/`$offset` (always returns every row regardless), so paging
    #: uses its own `pageNumber`/`pageSize` params instead.
    PAGE_SIZE = 1000

    def _get_page(self, page_number: int) -> List[Dict[str, object]]:
        """Fetch one `PAGE_SIZE`-row page of the NTD GTFS weblinks dataset.

        Args:
            page_number: 1-based page number to fetch.

        Returns:
            Raw NTD row dictionaries for this page (possibly fewer than
            `PAGE_SIZE` if this is the last page).
        """
        params = {"pageNumber": page_number, "pageSize": self.PAGE_SIZE}
        if self.api_key:
            params["$$app_token"] = self.api_key
        return request_json("GET", self.BASE_URL, params=params)

    def _fetch_all_rows(self) -> List[Dict[str, object]]:
        """Page through the entire NTD GTFS weblinks dataset.

        Returns:
            Every row in the dataset, in the order Socrata returns them.
        """
        rows: List[Dict[str, object]] = []
        page_number = 1
        while True:
            page = self._get_page(page_number)
            if not page:
                break
            rows.extend(page)
            if len(page) < self.PAGE_SIZE:
                break
            page_number += 1
        return rows

    def search_feeds(
        self,
        state: Optional[Union[str, List[str]]] = None,
        mode_name: Optional[Union[str, List[str]]] = None,
        include_waived: bool = False,
    ) -> List[GTFSFeedMetadata]:
        """Fetch NTD rows and translate them into `GTFSFeedMetadata`.

        Args:
            state: Two-letter state code(s) (NTD's `state` field) to keep.
                `None` keeps every state.
            mode_name: NTD `mode_name` value(s) to keep (see
                `NTD_MODE_TO_SCORING_MODE` for the full set). `None` keeps
                every mode.
            include_waived: If False (default), drop rows FTA granted a
                GTFS waiver for (`waived` is truthy) — those have no
                weblink to download by definition.

        Returns:
            One `GTFSFeedMetadata` per NTD row with a usable weblink,
            with the row's own fields kept in `raw["ntd"]`.
        """
        states = {s.upper() for s in state} if isinstance(state, list) else (
            {state.upper()} if state else None
        )
        modes = set(mode_name) if isinstance(mode_name, list) else (
            {mode_name} if mode_name else None
        )

        feeds: List[GTFSFeedMetadata] = []
        for row in self._fetch_all_rows():
            weblink_field = row.get("weblink")
            weblink = (
                weblink_field.get("url", "").strip()
                if isinstance(weblink_field, dict)
                else (weblink_field or "").strip()
            )
            if not weblink:
                continue
            if not include_waived and str(row.get("waived", "")).lower() in ("true", "1"):
                continue
            if states is not None and (row.get("state") or "").upper() not in states:
                continue
            if modes is not None and row.get("mode_name") not in modes:
                continue

            ntd_id = row.get("ntd_id", "")
            agency_id = row.get("agency_id", "")
            mode = row.get("mode", "")
            feeds.append(
                GTFSFeedMetadata(
                    id=f"{ntd_id}_{agency_id}_{mode}",
                    download_url=weblink,
                    name=row.get("agency_name"),
                    provider=row.get("agency_name"),
                    country_code="US",
                    source=self.SOURCE_NAME,
                    raw={"ntd": row},
                )
            )
        return feeds

    def download_feeds(
        self,
        feeds: List[GTFSFeedMetadata],
        download_folder: str,
        overwrite: bool = False,
        unzip: bool = True,
    ) -> List[str]:
        """Download NTD feeds, deduping shared URLs and applying BRT overrides.

        Several NTD rows commonly share one `download_url` (one agency's
        single GTFS feed reported once per fixed-route mode it operates),
        so this dedupes by URL before downloading -- each unique feed is
        fetched once, not once per NTD row pointing at it -- then, for
        every feed whose NTD rows include "Bus Rapid Transit", applies
        `apply_brt_overrides_for_feed` to the downloaded copy (never the
        upstream ZIP/source).

        Args:
            feeds: Feeds to download, as returned by `search_feeds()`.
            download_folder: Directory to store (and, if `unzip`,
                extract) the downloaded feeds into.
            overwrite: If True, re-download and replace files that
                already exist on disk.
            unzip: If True, extract each ZIP after downloading and delete
                the ZIP file. BRT overrides require this (they rewrite
                `routes.txt` inside the extracted folder), so a BRT feed
                downloaded with `unzip=False` is left un-overridden with
                a warning.

        Returns:
            Absolute paths to the downloaded feeds, one per entry in
            `feeds` (feeds sharing a URL with an earlier entry get that
            entry's path; skipped feeds are simply absent).
        """
        first_feed_by_url: Dict[str, GTFSFeedMetadata] = {}
        rows_by_url: Dict[str, List[Dict[str, object]]] = {}
        for feed in feeds:
            first_feed_by_url.setdefault(feed.download_url, feed)
            rows_by_url.setdefault(feed.download_url, []).append(feed.raw.get("ntd", {}))

        unique_feeds = list(first_feed_by_url.values())
        unique_paths = super().download_feeds(
            unique_feeds, download_folder, overwrite=overwrite, unzip=unzip
        )
        path_by_feed_id = {f.id: p for f, p in zip(unique_feeds, unique_paths)}

        if unzip:
            for feed in unique_feeds:
                path = path_by_feed_id.get(feed.id)
                ntd_rows = rows_by_url.get(feed.download_url, [])
                if path and any(r.get("mode_name") == "Bus Rapid Transit" for r in ntd_rows):
                    apply_brt_overrides_for_feed(path, ntd_rows)
        else:
            for feed in unique_feeds:
                ntd_rows = rows_by_url.get(feed.download_url, [])
                if any(r.get("mode_name") == "Bus Rapid Transit" for r in ntd_rows):
                    logger.warning(
                        f"Feed '{feed.id}' includes NTD BRT mode but unzip=False -- "
                        "BRT route_type override was NOT applied."
                    )

        path_by_url = {f.download_url: path_by_feed_id.get(f.id) for f in unique_feeds}
        return [path_by_url[f.download_url] for f in feeds if path_by_url.get(f.download_url)]

    @staticmethod
    def check_feed_links(
        feeds: List[GTFSFeedMetadata], timeout: int = 20
    ) -> "tuple[List[GTFSFeedMetadata], List[GTFSFeedMetadata]]":
        """Check which feeds' weblinks actually resolve.

        Issues a streamed GET (not just HEAD, since several NTD-listed
        hosts don't implement HEAD) against each feed's `download_url`
        and only reads enough of the body to confirm the request
        succeeded, closing the connection immediately after.

        Args:
            feeds: Feeds to check, as returned by `search_feeds()`.
            timeout: Per-request timeout in seconds.

        Returns:
            A tuple `(working, broken)` partitioning `feeds` by whether
            their weblink resolved with a 2xx status.
        """
        working: List[GTFSFeedMetadata] = []
        broken: List[GTFSFeedMetadata] = []
        for feed in feeds:
            try:
                with requests.get(
                    feed.download_url, stream=True, timeout=timeout, allow_redirects=True
                ) as response:
                    response.raise_for_status()
                    next(response.iter_content(chunk_size=1024), None)
                working.append(feed)
            except (requests.exceptions.RequestException, StopIteration) as e:
                logger.warning(f"NTD weblink broken for feed '{feed.id}': {e}")
                broken.append(feed)
        return working, broken
