# -*- coding: utf-8 -*-
"""Downloaders for US GTFS sources.

Currently holds `ntd`, the client for the Federal Transit Administration's
National Transit Database GTFS weblinks registry
(https://www.transit.dot.gov/ntd), the official nationwide source of GTFS
feeds required from US fixed-route transit agencies.
"""

from .brt import BRT_ROUTE_TYPE, KNOWN_BRT_ROUTES, apply_brt_overrides_for_feed
from .us_feeds import download_revised_feed, download_us_feeds, load_revised_feeds, replace_ntd_ids
from .ntd import NTDDownloader, NTD_MODE_EMOJI, NTD_MODE_TO_SCORING_MODE, ROUTE_TYPE_EMOJI

__all__ = [
    "download_us_feeds",
    "download_revised_feed",
    "load_revised_feeds",
    "NTDDownloader",
    "NTD_MODE_EMOJI",
    "NTD_MODE_TO_SCORING_MODE",
    "ROUTE_TYPE_EMOJI",
    "BRT_ROUTE_TYPE",
    "KNOWN_BRT_ROUTES",
    "apply_brt_overrides_for_feed",
]
