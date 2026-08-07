# -*- coding: utf-8 -*-
"""Geocoding/reverse-geocoding helpers: resolving a place name or an AOI
geometry to country/subdivision/municipality, and fetching OSM city
boundaries.

Why this module exists and how it's organized:
-----------------------------------------------
- **`get_country_region`**: the one geocoding call on the core `Feed`
  loading path -- `models/calendar.py` uses it (via `Feed`'s mean stop
  coordinates) to resolve which country/subdivision a feed is in, purely to
  look up public holidays for `date_type="holiday"` filtering. It talks to
  Nominatim directly over plain `requests` (no `geopy` dependency) and is
  the only function here without a lazy import, since `requests` is already
  a core dependency.
- **`get_city_geometry`**/**`get_geographic_suggestions_from_string`**/
  **`get_geographic_suggestions_from_aoi`**: user-facing convenience helpers
  (e.g. for interactively picking an AOI/city before constructing a `Feed`),
  not called anywhere internally. These import `geopy` lazily, inside the
  function body, so merely `import pyGTFSHandler` doesn't require installing
  that optional extra.
"""

import warnings
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pycountry
import requests
from difflib import get_close_matches
from shapely.geometry import MultiPolygon, Point, Polygon, shape as shapely_shape

import logging
logger = logging.getLogger(__name__)


def get_city_geometry(city_name: str, user_agent: str = "pyGTFSHandlerClient") -> gpd.GeoDataFrame:
    """Retrieve a place's boundary geometry from OpenStreetMap/Nominatim.

    Uses `geopy`'s Nominatim geocoder with GeoJSON polygon output, so it
    needs no extra dependency beyond the `geocoding` optional extra
    (`pip install pyGTFSHandler[geocoding]`) already used by the other
    functions in this module.
    """
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    location = geolocator.geocode(city_name, geometry="geojson")
    if location is None or "geojson" not in location.raw:
        raise ValueError(f"Could not find boundary geometry for {city_name!r}.")

    geometry = shapely_shape(location.raw["geojson"])
    gdf = gpd.GeoDataFrame({"name": [location.address]}, geometry=[geometry], crs="EPSG:4326")
    return gdf


def get_geographic_suggestions_from_string(
    query: str,
    user_agent: str = "MobilityDatabaseClient",
    max_results: int = 25
) -> Dict[str, List[str]]:
    """Suggest country codes, subdivisions, and municipalities from a query string."""
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    try:
        locations = geolocator.geocode(
            query, addressdetails=True, language='en', exactly_one=False, limit=max_results
        )
        if locations:
            for loc in locations:
                address = loc.raw.get('address', {})
                if country_code := address.get('country_code'):
                    suggested_country_codes.add(country_code.upper())
                for key in ['state', 'province', 'region', 'county']:
                    if value := address.get(key):
                        suggested_subdivision_names.add(value)
                for key in ['city', 'town', 'village', 'county']:
                    if value := address.get(key):
                        suggested_municipalities.add(value)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected geocoding error: {e}")

    return {
        'country_codes': sorted(suggested_country_codes),
        'subdivision_names': sorted(suggested_subdivision_names),
        'municipalities': sorted(suggested_municipalities)
    }


def get_geographic_suggestions_from_aoi(
    aoi: Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries],
    num_points: int = 1,
    user_agent: str = "MobilityDatabaseClient"
) -> Dict[str, List[str]]:
    """Reverse-geocode AOI geometry to suggest country, subdivision, and municipality."""
    import random
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if aoi.empty:
            raise ValueError("GeoDataFrame/GeoSeries is empty.")
        target_geometry = aoi.to_crs(4326).unary_union
    elif isinstance(aoi, (Polygon, MultiPolygon)):
        target_geometry = aoi
    else:
        raise TypeError("AOI must be Polygon, MultiPolygon, GeoDataFrame, or GeoSeries.")

    if target_geometry.is_empty:
        raise ValueError("AOI geometry is empty.")

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    points_to_geocode: List[Point] = []
    min_lon, min_lat, max_lon, max_lat = target_geometry.bounds

    if num_points <= 0:
        num_points = 1
    if num_points == 1:
        points_to_geocode.append(target_geometry.representative_point())
    else:
        for _ in range(num_points):
            points_to_geocode.append(Point(random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)))

    for i, point in enumerate(points_to_geocode):
        lat, lon = point.y, point.x
        logger.debug(f"Reverse geocoding point {i+1}/{len(points_to_geocode)}: ({lat}, {lon})")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location and location.raw:
                address = location.raw.get('address', {})
                if cc := address.get('country_code'):
                    suggested_country_codes.add(cc.upper())
                if subdivision := address.get('state') or address.get('province') or address.get('region') or address.get('county'):
                    suggested_subdivision_names.add(subdivision)
                if municipality := address.get('city') or address.get('town') or address.get('village'):
                    suggested_municipalities.add(municipality)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"Geocoding failed for point ({lat}, {lon}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error for point ({lat}, {lon}): {e}")

    return {
        'country_codes': sorted(list(suggested_country_codes)),
        'subdivision_names': sorted(list(suggested_subdivision_names)),
        'municipalities': sorted(list(suggested_municipalities))
    }


def get_country_region(lat: float, lon: float) -> (str, Optional[str]):
    """Return ISO country code and subdivision code for a lat/lon location."""
    url = "https://nominatim.openstreetmap.org/reverse"
    headers = {"User-Agent": "pyGTFSHandler/0.1.0"}
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1}

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("address", {})

    country_code = data.get("country_code", "").upper()
    region_name = data.get("state") or data.get("region")
    subdivision_code = None

    if country_code and region_name:
        try:
            subdivisions = list(pycountry.subdivisions.get(country_code=country_code))
            for subdiv in subdivisions:
                if subdiv.name.lower() == region_name.lower():
                    subdivision_code = subdiv.code
                    break
            if not subdivision_code:
                close_matches = get_close_matches(region_name, [s.name for s in subdivisions], n=3, cutoff=0.6)
                for match in close_matches:
                    for s in subdivisions:
                        if s.name == match:
                            subdivision_code = s.code
                            warnings.warn(f"Fuzzy match used for region '{region_name}' -> '{match}'")
                            break
                    if subdivision_code:
                        break
        except LookupError:
            subdivision_code = None

    return country_code, subdivision_code
