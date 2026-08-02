# -*- coding: utf-8 -*-
"""Route-type -> emoji/label lookup shared by `route_map.py` (both the
Python-side per-mode legend/checkbox generation and, serialized as JSON,
the client-side JS that builds stop icons and route badges).
"""

# Standard GTFS route_type 0-7, keyed as str so it round-trips cleanly
# through JSON (JS object keys are always strings).
ROUTE_TYPE_EMOJI = {
    "0": "\U0001F68A",  # tram
    "1": "\U0001F687",  # subway/metro
    "2": "\U0001F686",  # rail
    "3": "\U0001F68C",  # bus
    "4": "⛴️",  # ferry
    "5": "\U0001F68B",  # cable tram
    "6": "\U0001F681",  # aerial lift
    "7": "\U0001F69E",  # funicular
}
ROUTE_TYPE_EMOJI_FALLBACK = "\U0001F68F"  # bus stop, for unknown/extended/-1

ROUTE_TYPE_NAME = {
    "0": "Tram",
    "1": "Subway",
    "2": "Rail",
    "3": "Bus",
    "4": "Ferry",
    "5": "Cable tram",
    "6": "Aerial lift",
    "7": "Funicular",
}
ROUTE_TYPE_NAME_FALLBACK = "Other"

# route_type values that get a square/rounded-square badge (rail-like modes);
# everything else (bus, ferry, unknown/-1) gets a circular badge.
SQUARE_BADGE_ROUTE_TYPES = {0, 1, 2, 5, 6, 7}


def route_type_emoji(route_type) -> str:
    return ROUTE_TYPE_EMOJI.get(str(route_type), ROUTE_TYPE_EMOJI_FALLBACK)


def route_type_name(route_type) -> str:
    return ROUTE_TYPE_NAME.get(str(route_type), ROUTE_TYPE_NAME_FALLBACK)
