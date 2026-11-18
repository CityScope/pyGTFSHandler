# -*- coding: utf-8 -*-
"""Geocoding helpers: the implementation lives in geohierarchy (`geohierarchy.geocoding`), shared by every package.

`models/calendar.py` uses `get_country_region` (from a feed's mean stop location) to pick the public holidays of
`date_type="holiday"`; the other functions are user-facing helpers for picking an area.
"""
from geohierarchy.geocoding import (  # noqa: F401
    get_city_geometry,
    get_country_region,
    get_geographic_suggestions_from_aoi,
    get_geographic_suggestions_from_string,
    reverse_geocode,
)
