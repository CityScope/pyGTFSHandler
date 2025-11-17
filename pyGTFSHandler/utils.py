import os
import csv
import re
import hashlib
import warnings
import zipfile
from datetime import datetime, date, time
from typing import List, Optional, Dict, Union, Tuple, Any
from collections import Counter
from itertools import islice

import requests
import polars as pl
import pycountry
import geopandas as gpd
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from difflib import get_close_matches
from shapely.geometry import Polygon, MultiPolygon, Point

from . import gtfs_checker

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EPOCH = date(1970, 1, 1)
ID_COLS = ["trip_id", "service_id", "route_id", "stop_id", "shape_id", "parent_station"]
MANDATORY_COLS = ["trip_id", "service_id", "stop_id"]


# -------------------------
# Date and Time Utilities
# -------------------------
def datetime_to_days_since_epoch(dt: Union[datetime, date]) -> int:
    """Convert datetime/date to number of days since 1970-01-01."""
    if isinstance(dt, datetime):
        dt = dt.date()
    return (dt - EPOCH).days


def time_to_seconds(t: Union[datetime, time]) -> int:
    """Convert datetime/time object to seconds since midnight."""
    if isinstance(t, datetime):
        t = t.time()
    return t.hour * 3600 + t.minute * 60 + t.second


# -------------------------
# Geospatial Utilities
# -------------------------
def get_city_geometry(city_name: str) -> gpd.GeoDataFrame:
    """Retrieve city boundary geometry from OpenStreetMap using OSMnx."""
    gdf = ox.geocode_to_gdf(city_name)
    return gdf.to_crs(epsg=4326)


def get_geographic_suggestions_from_string(
    query: str,
    user_agent: str = "MobilityDatabaseClient",
    max_results: int = 25
) -> Dict[str, List[str]]:
    """Suggest country codes, subdivisions, and municipalities from a query string."""
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

# -------------------------
# CSV / GTFS Utilities
# -------------------------
def read_csv_lazy(
    path: str,
    schema_overrides: Optional[Dict[str, pl.DataType]] = None,
    file_id: Optional[int] = None,
    check_files: bool = True,
    mandatory_cols: List[str] = MANDATORY_COLS,
    id_cols: List[str] = ID_COLS
) -> Optional[pl.LazyFrame]:
    """Lazily read a CSV (GTFS) file into a Polars LazyFrame."""
    if not path or not os.path.isfile(path):
        return None

    try:
        lf = pl.scan_csv(path, infer_schema=False, raise_if_empty=False, truncate_ragged_lines=check_files)
    except Exception as e:
        warnings.warn(f"scan_csv failed ({e}). Falling back to read_csv.")
        try:
            lf = pl.read_csv(path, infer_schema=False, ignore_errors=check_files, truncate_ragged_lines=check_files).lazy()
        except Exception as e:
            warnings.warn(f"Failed to load CSV {path}: {e}")
            return None

    if check_files:
        lf = gtfs_checker.normalize_df(lf)

    if schema_overrides:
        for col, dtype in schema_overrides.items():
            if dtype == "int|bool":
                dtype = int 
            elif dtype == "time|None":
                dtype = str 
            elif dtype == "time":
                dtype = str 
            elif dtype == "date":
                dtype = int 
            elif dtype == "date|None":
                dtype = int 
            elif dtype == "seconds":
                dtype = int 
            elif dtype == "exception_type":
                dtype = int 
            elif dtype == "route_type":
                dtype = int 
            elif isinstance(dtype,str):
                dtype = str

            if col in lf.collect_schema().names():
                lf = lf.with_columns(pl.col(col).cast(dtype, strict=False))

    gtfs_name = os.path.basename(os.path.dirname(path))
    lf = lf.with_columns(pl.lit(gtfs_name).alias("gtfs_name"), pl.lit(file_id).alias("file_id"))

    columns = lf.collect_schema().names()
    for col in id_cols:
        if col in columns:
            lf = lf.with_columns(
                pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
                .then(pl.lit(None))
                .otherwise(pl.concat_str([pl.col(col), pl.lit("_file_"), pl.col("file_id")]))
                .alias(col)
            )
            if col in mandatory_cols:
                lf = lf.filter(pl.col(col).is_not_null())

    return lf


def read_csv_list(
    path_list: List[str],
    schema_overrides: Optional[Dict[str, pl.DataType]] = None,
    search_files: bool = False,
    min_file_id: int = 0,
    check_files: bool = True,
    mandatory_cols: List[str] = MANDATORY_COLS,
    id_cols: List[str] = ID_COLS
) -> Optional[pl.LazyFrame]:
    """Lazily read a list of CSV files into a single concatenated LazyFrame."""
    if search_files:
        path_list = [gtfs_checker.search_file(os.path.dirname(p), os.path.basename(p)) or p for p in path_list]

    file_lfs = [
        lf for i, p in enumerate(path_list)
        if (lf := read_csv_lazy(p, schema_overrides=schema_overrides, file_id=i+min_file_id, check_files=check_files, mandatory_cols=mandatory_cols, id_cols=id_cols)) is not None
    ]

    if not file_lfs:
        return None

    return pl.concat(file_lfs, how="diagonal_relaxed")


# -------------------------
# Country & Holidays
# -------------------------
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


def get_holidays(year: int, country_code: str, subdivision_code: Optional[str] = None) -> pl.DataFrame:
    """Fetch public holidays for a country and optional subdivision."""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"

    empty_df = pl.DataFrame(schema={
        "date": pl.Int32,
        "localName": pl.Utf8,
        "name": pl.Utf8,
        "countryCode": pl.Utf8,
        "fixed": pl.Boolean,
        "global": pl.Boolean,
        "counties": pl.List(pl.Utf8),
        "launchYear": pl.Int32,
        "types": pl.List(pl.Utf8),
    })

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            warnings.warn(f"Holiday API request failed: {resp.status_code}")
            return empty_df
        holidays = resp.json()
        if not holidays:
            return empty_df
    except Exception as e:
        warnings.warn(f"Holiday API request error: {e}")
        return empty_df

    if subdivision_code:
        holidays = [h for h in holidays if not h.get("counties") or subdivision_code in h.get("counties", [])]

    df = pl.DataFrame(holidays)
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .dt.epoch(time_unit="d")
        )
    return df


# -------------------------
# Polars Utilities
# -------------------------
def filter_by_id_column(lf, column, ids: list | None = []):
    "TODO: Deal with _file_id in ids and column"
    if ids is None:
        ids = []

    if lf is None:
        return None

    if len(ids) > 0:
        ids_df = pl.LazyFrame({column: ids})
        lf = lf.join(ids_df, on=column, how="semi")

    return lf


def mean_angle(column: str, over: Optional[List[str]] = None) -> pl.Expr:
    """Compute the mean angle in degrees of a column of angles."""
    if over is None:
        mean_cos = pl.col(column).radians().cos().mean()
        mean_sin = pl.col(column).radians().sin().mean()
    else:
        mean_cos = pl.col(column).radians().cos().mean().over(over)
        mean_sin = pl.col(column).radians().sin().mean().over(over)

    return pl.arctan2(mean_sin, mean_cos).degrees().mod(360)


def max_separation_angle(df: pl.DataFrame, column: str) -> pl.Series:
    """
    Compute the maximum separation angle in a list column of angles (degrees).
    Returns a Series of maximum separation angles.
    """
    df = (
        df.with_columns(pl.col(column).list.concat(pl.col(column) + 180).alias(column))
          .with_columns([(pl.col(column) % 360).list.min().alias(f"{column}_min_angle")])
          .with_columns((pl.col(column) % 360 - pl.col(f"{column}_min_angle")).alias(f"{column}_normalized"))
          .with_columns([(pl.col(f"{column}_normalized").list.sort().list.concat(pl.lit([360]))).alias(f"{column}_angle_sorted")])
          .with_columns([pl.col(f"{column}_angle_sorted").list.diff(null_behavior="drop").alias(f"{column}_arc_angle")])
          .with_columns([(pl.col(f"{column}_arc_angle").list.max() / 2 +
                          pl.col(f"{column}_angle_sorted").list.get(pl.col(f"{column}_arc_angle").list.arg_max()) +
                          pl.col(f"{column}_min_angle")).alias(f"{column}_max_separation_angle")])
    )
    return df[f"{column}_max_separation_angle"]


# -------------------------
# File Hash Utilities
# -------------------------
def hash_file(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a single file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(folder_path: str, chunk_size: int = 8192) -> str:
    """Compute combined MD5 hash of all .txt files in a folder."""
    hashes = [hash_file(os.path.join(folder_path, f), chunk_size) for f in sorted(os.listdir(folder_path)) if f.endswith(".txt")]
    return hashlib.md5("".join(hashes).encode()).hexdigest()


def hash_zip(zip_path: str, chunk_size: int = 8192) -> str:
    """Compute combined MD5 hash of all .txt files in a zip archive."""
    hashes = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for f in sorted(z.namelist()):
            if f.endswith(".txt"):
                h = hashlib.md5()
                with z.open(f) as file:
                    for chunk in iter(lambda: file.read(chunk_size), b""):
                        h.update(chunk)
                hashes.append(h.hexdigest())
    return hashlib.md5("".join(hashes).encode()).hexdigest()


def hash_path(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file, folder, or zip archive."""
    if os.path.isdir(path):
        return hash_folder(path, chunk_size)
    if zipfile.is_zipfile(path):
        return hash_zip(path, chunk_size)
    if os.path.isfile(path):
        return hash_file(path, chunk_size)
    raise ValueError(f"{path} is not a file, folder, or zip archive")


def compare_paths(path1: str, path2: str) -> bool:
    """Compare any two paths by content hash."""
    return hash_path(path1) == hash_path(path2)


# def geohash(
#     df: pl.DataFrame, lat_col: str, lon_col: str, precision: int = 7
# ) -> pl.DataFrame:
#     """
#     Add a geohash column to a Polars DataFrame based on latitude and longitude columns.

#     Parameters
#     ----------
#     df : pl.DataFrame
#         The input Polars DataFrame containing latitude and longitude data.
#     lat_col : str
#         The name of the latitude column in the DataFrame.
#     lon_col : str
#         The name of the longitude column in the DataFrame.
#     precision : int, optional
#         The precision (length) of the geohash string. Default is 7.

#     Returns
#     -------
#     pl.DataFrame
#         A new DataFrame with an additional 'geohash' column containing geohash strings.
#     """
#     df = df.with_columns(
#         pl.struct([lat_col, lon_col])
#         .map_elements(
#             lambda point: pygeohash.encode(
#                 point[lat_col], point[lon_col], precision=precision
#             ),
#             return_dtype=pl.String,
#         )
#         .alias("geohash")
#     )
#     return df
