import polars as pl
import unicodedata
import re
from typing import List, Optional
import os
from datetime import datetime, date, time
import requests
import pycountry
from difflib import get_close_matches
import warnings

import zipfile
import hashlib
import csv
import tempfile
import shutil
from collections import Counter
from itertools import islice
import logging 
import gtfs_checker 

# import pygeohash

"TODO: if filter_by_id_column ids come with file_number then filter by file_number too and if not dont and allow mix in the same list"

EPOCH = date(1970, 1, 1)

ID_COLS = ["trip_id", "service_id", "route_id", "stop_id", "shape_id", "parent_station"]
MANDATORY_COLS = ["trip_id", "service_id", "stop_id"]

# Configure logging for better output control
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def datetime_to_days_since_epoch(dt: datetime | date) -> int:
    if type(dt) is datetime:
        dt = dt.date()

    delta = dt - EPOCH
    return delta.days


def time_to_seconds(t: datetime | time) -> int:
    """Helper to convert a time object to seconds since midnight."""
    if type(t) is datetime:
        t = t.time()

    return t.hour * 3600 + t.minute * 60 + t.second


def get_city_geometry(city_name: str) -> gpd.GeoDataFrame:
    """
    Download city boundary geometry from OpenStreetMap.

    Parameters
    ----------
    city_name : str
        Name of the city (e.g., "Berlin, Germany").

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the city boundary polygon in EPSG:4326.
    """
    # Query OSM for place boundary
    gdf = ox.geocode_to_gdf(city_name)

    # Ensure CRS is WGS84
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def get_geographic_suggestions_from_string(
    query: str,
    user_agent: str = "MobilityDatabaseClient",
    max_results: int = 25
) -> Dict[str, List[str]]:
    """
    Suggests all possible country codes, subdivisions, and municipalities
    for a given string using OpenStreetMap's Nominatim service.
    
    This version collects all relevant fields without skipping any.
    Counties are always included in municipalities.
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=10)

    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    try:
        locations = geolocator.geocode(
            query,
            addressdetails=True,
            language='en',
            exactly_one=False,
            limit=max_results
        )
        if locations:
            for location in locations:
                address = location.raw.get('address', {})

                # Country code
                country_code = address.get('country_code')
                if country_code:
                    suggested_country_codes.add(country_code.upper())

                # Collect all possible subdivisions
                for key in ['state', 'province', 'region', 'county']:
                    value = address.get(key)
                    if value:
                        suggested_subdivision_names.add(value)

                # Collect all possible municipalities
                for key in ['city', 'town', 'village', 'county']:
                    value = address.get(key)
                    if value:
                        suggested_municipalities.add(value)

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return {
        'country_codes': sorted(suggested_country_codes),
        'subdivision_names': sorted(suggested_subdivision_names),
        'municipalities': sorted(suggested_municipalities),
    }

def get_geographic_suggestions_from_aoi(
    aoi: Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries],
    num_points: int = 1, # Number of points to sample for geocoding
    user_agent: str = "MobilityDatabaseClient" # Required by Nominatim
) -> Dict[str, List[str]]:
    """
    Suggests country codes, subdivisions, and municipalities for a given
    Area of Interest (AOI) by performing reverse geocoding.
    This simplified version generates sample points within the AOI's *bounding box*.

    This function uses OpenStreetMap's Nominatim service via geopy.
    Please be respectful of Nominatim's usage policy (one request per second).
    It's recommended to set a specific `user_agent` for your application.

    Args:
        aoi: An Area of Interest, which can be a shapely.geometry.Polygon,
             shapely.geometry.MultiPolygon, geopandas.GeoDataFrame, or
             geopandas.GeoSeries.
        num_points: The number of random points to sample within the AOI's
                    bounding box for reverse geocoding. More points can provide broader
                    coverage for large AOIs, but increases the number of Nominatim requests.
                    Defaults to 1 (representative point).
        user_agent: A unique user agent string for Nominatim requests. This is required.

    Returns:
        A dictionary containing lists of suggested 'country_codes',
        'subdivision_names', and 'municipalities'. Example:
        {
            'country_codes': ['US', 'CA'],
            'subdivision_names': ['California', 'Québec'],
            'municipalities': ['Los Angeles', 'Montreal']
        }
        Returns lists that might be empty if geocoding fails or no relevant info is found.

    Raises:
        ImportError: If geopandas or geopy are not installed.
        TypeError: If `aoi` is not a supported geospatial object.
        ValueError: If the AOI is empty or invalid.
    """
    if gpd is None or Nominatim is None or random is None:
        raise ImportError("geopandas, geopy, and random must be available to use get_geographic_suggestions_from_aoi. Please run 'pip install geopandas geopy'.")

    target_geometry: Union[Polygon, MultiPolygon]
    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if aoi.empty:
            raise ValueError("Provided GeoDataFrame/GeoSeries is empty.")
        target_geometry = aoi.to_crs(4326).union_all() # Combine all geometries
    elif isinstance(aoi, (Polygon, MultiPolygon)):
        target_geometry = aoi
    else:
        raise TypeError("aoi must be a shapely.geometry.Polygon, MultiPolygon, geopandas.GeoDataFrame, or geopandas.GeoSeries object.")

    if target_geometry.is_empty:
        raise ValueError("AOI geometry is empty.")

    geolocator = Nominatim(user_agent=user_agent, timeout=10)

    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    points_to_geocode: List[Point] = []
    if num_points <= 0:
        num_points = 1 # Ensure at least one point is sampled

    # Calculate bounding box once
    min_lon, min_lat, max_lon, max_lat = target_geometry.bounds

    if num_points == 1:
        # For a single point, representative_point is often more meaningful than bbox center
        points_to_geocode.append(target_geometry.representative_point())
    else:
        # Generate random points within the bounding box
        for _ in range(num_points):
            rand_lon = random.uniform(min_lon, max_lon)
            rand_lat = random.uniform(min_lat, max_lat)
            points_to_geocode.append(Point(rand_lon, rand_lat))

    for i, point in enumerate(points_to_geocode):
        lat, lon = point.y, point.x
        logger.debug(f"Geocoding point {i+1}/{len(points_to_geocode)}: ({lat}, {lon}) (sampled from bbox)")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location and location.raw:
                address = location.raw.get('address', {})
                
                # Country code (ISO 3166-1 alpha-2)
                country_code_long = address.get('country_code')
                if country_code_long:
                    suggested_country_codes.add(country_code_long.upper())

                # Subdivision name (state, province, region, etc.)
                # Ordered by common usage/specificity
                subdivision = address.get('state') or address.get('province') or address.get('region') or address.get('county')
                if subdivision:
                    suggested_subdivision_names.add(subdivision)

                # Municipality (city, town, village)
                municipality = address.get('city') or address.get('town') or address.get('village')
                if municipality:
                    suggested_municipalities.add(municipality)
            else:
                logger.warning(f"No location data found for point ({lat}, {lon}).")
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding failed for point ({lat}, {lon}): {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during geocoding for point ({lat}, {lon}): {e}")

    return {
        'country_codes': sorted(list(suggested_country_codes)), # Sort for consistent output
        'subdivision_names': sorted(list(suggested_subdivision_names)),
        'municipalities': sorted(list(suggested_municipalities))
    }


def read_csv_lazy(
    path: str, schema_overrides: dict | None = None, file_id: int | None = None, check_files:bool=True, mandatory_cols=MANDATORY_COLS, id_cols=ID_COLS
) -> pl.LazyFrame | None:
    """
    Lazily reads a CSV file into a Polars LazyFrame with optional schema overrides and column filtering.

    This is designed for reading GTFS files efficiently, allowing for partial schema enforcement
    and retaining metadata such as the GTFS directory name.

    Parameters:
        path : str
            Full path to the GTFS .txt file (CSV format).
        schema_overrides : dict, optional
            A dictionary mapping column names to Polars data types to override inferred types.
            Example: {'stop_lat': pl.Float64, 'stop_lon': pl.Float64}
        column_names : list of str, optional
            A list of column names to include. If None, all columns are read.

    Returns:
        pl.LazyFrame
            A lazily loaded Polars LazyFrame with applied schema overrides and a new column `gtfs_name`
            containing the GTFS directory name inferred from the path.
    """
    if (path is None) or (not os.path.isfile(path)):
        return None

    if check_files:
        infer_schema = False 
    else:
        infer_schema = True

    try:
        # Try lazy scan first
        lf = pl.scan_csv(
            path,
            infer_schema=infer_schema,
            raise_if_empty=False,
            truncate_ragged_lines=check_files
        )
    except Exception as e:
        # Show exception as a warning
        warnings.warn(f"scan_csv failed with error: {e}. Falling back to read_csv with ignore_errors.")
        
        # Fallback: read_csv with ignore_errors and convert to lazy
        try:
            lf = pl.read_csv(
                path,
                infer_schema=infer_schema,          # don’t try to infer types
                ignore_errors=check_files,          # skip parsing errors
                truncate_ragged_lines=check_files   # handle lines with missing columns
            ).lazy()
        except Exception as e:
            warnings.warn(f"Failed to load CSV {path}: {e}")
            return None

    if check_files:
        # Apply custom normalization (assuming normalize_df is defined elsewhere)
        lf = gtfs_checker.normalize_df(lf)

        # Apply schema overrides if specified
        if schema_overrides:
            for col, dtype in schema_overrides.items():
                if col in lf.collect_schema().names():
                    # Cast non-strictly so that parsing errors become null
                    lf = lf.with_columns(pl.col(col).cast(dtype, strict=False))

    # Extract GTFS directory name from path (e.g., 'gtfs_file' from '/home/xyz/gtfs_file/stops.txt')
    gtfs_name = os.path.basename(os.path.dirname(path))

    # Add gtfs_name as a literal column
    lf = lf.with_columns(
        pl.lit(gtfs_name).alias("gtfs_name"), pl.lit(file_id).alias("file_id")
    )

    columns = lf.collect_schema().names()
    if check_files:
        for col in id_cols:
            if col in columns:
                lf = lf.with_columns(
                    pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
                    .then(pl.lit(None))
                    .otherwise(
                        pl.concat_str([pl.col(col), pl.lit("_file_"), pl.col("file_id")])
                    )
                    .alias(col)
                )
                if col in mandatory_cols:
                    lf = lf.filter(pl.col(col).is_not_null())
    else:
        for col in id_cols:
            if col in columns:
                lf = lf.with_columns(
                    pl.col(col).cast(str).alias(col)
                )
                lf = lf.with_columns(
                    pl.when(pl.col(col).is_null())
                    .then(pl.lit(None))
                    .otherwise(
                        pl.concat_str([pl.col(col), pl.lit("_file_"), pl.col("file_id")])
                    )
                    .alias(col)
                )
    return lf


def read_csv_list(
    path_list: List[str], schema_overrides: Optional[dict] = None, search_files:bool=False, min_file_id:int=0, check_files:bool=True, mandatory_cols = MANDATORY_COLS, id_cols=ID_COLS
) -> pl.LazyFrame:
    """
    Lazily reads a list of CSV (GTFS) files into a single concatenated Polars LazyFrame.

    This function uses `read_csv_lazy()` to read each individual file lazily,
    optionally applying schema overrides and selecting specific columns, and then
    concatenates all resulting LazyFrames.

    Parameters:
        path_list : list of str
            List of file paths to GTFS CSV (.txt) files.
        schema_overrides : dict, optional
            A dictionary mapping column names to Polars data types to override inferred types.
            Example: {'stop_lat': pl.Float64, 'stop_lon': pl.Float64}
        column_names : list of str, optional
            A list of columns to read from each file. If None, all columns are read.

    Returns:
        pl.LazyFrame
            A concatenated LazyFrame of all input files, using `how='diagonal_relaxed'` to handle differing schemas.
    """
    if search_files:
        new_path_list = []
        for path in path_list:
            if path is None:
                new_path_list.append(None)

            folder, file = os.path.split(path)
            new_path = gtfs_checker.search_file(folder,file)
            if new_path is None:
                print(f"File {file} not found in path {folder}")
            else:
                new_path_list.append(new_path)

        path_list = new_path_list

    file_lfs = [
        res
        for i in range(len(path_list))
        if (
            res := read_csv_lazy(
                path_list[i], schema_overrides=schema_overrides, file_id=i+min_file_id, check_files=check_files, mandatory_cols=mandatory_cols, id_cols=id_cols
            )
        )
        is not None
    ]

    if len(file_lfs) == 0:
        return None

    return pl.concat(
        file_lfs,
        how="diagonal_relaxed",
    )


def get_country_region(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    headers = {"User-Agent": "pyGTFSHandler/0.1.0 (https://blogs.upm.es/aga/en/)"}
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
            subdivision_names = [subdiv.name for subdiv in subdivisions]

            # Exact match (case-insensitive)
            for subdiv in subdivisions:
                if subdiv.name.lower() == region_name.lower():
                    subdivision_code = subdiv.code
                    break

            # If no exact match, try fuzzy matching on subdivision names
            if subdivision_code is None:
                close_matches = get_close_matches(
                    region_name, subdivision_names, n=3, cutoff=0.6
                )
                if close_matches:
                    for match_name in close_matches:
                        for subdiv in subdivisions:
                            if subdiv.name == match_name:
                                subdivision_code = subdiv.code
                                break
                        if subdivision_code:
                            break
                    if subdivision_code:
                        warnings.warn(
                            f"Fuzzy match used for region '{region_name}'. Matched with '{match_name}'.",
                            UserWarning,
                        )

            # fallback: pycountry's search_fuzzy
            if subdivision_code is None:
                fuzzy_matches = pycountry.subdivisions.search_fuzzy(region_name)
                for match in fuzzy_matches:
                    if match.country_code == country_code:
                        subdivision_code = match.code
                        warnings.warn(
                            f"Fuzzy match used via pycountry.search_fuzzy for region '{region_name}'. Matched with '{match.name}'.",
                            UserWarning,
                        )
                        break

        except LookupError:
            subdivision_code = None

    return country_code, subdivision_code


def get_holidays(year, country_code, subdivision_code=None):
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"

    # Define a consistent empty schema
    empty_df = pl.DataFrame(
        schema={
            "date": pl.Int32,
            "localName": pl.Utf8,
            "name": pl.Utf8,
            "countryCode": pl.Utf8,
            "fixed": pl.Boolean,
            "global": pl.Boolean,
            "counties": pl.List(pl.Utf8),
            "launchYear": pl.Int32,
            "types": pl.List(pl.Utf8),
        }
    )

    try:
        response = requests.get(url, timeout=10)

        # Case: 204 No Content
        if response.status_code == 204:
            warnings.warn(
                f"Holiday API returned no content for {country_code}-{year} (204)."
            )
            return empty_df

        # Case: other non-200 status codes
        if response.status_code != 200:
            warnings.warn(
                f"Holiday API request failed for {country_code}-{year}. "
                f"Status code: {response.status_code}"
            )
            return empty_df

        # Ensure response is JSON
        if "application/json" not in response.headers.get("Content-Type", ""):
            warnings.warn(
                f"Holiday API returned non-JSON response for {country_code}-{year}. "
                f"Content-Type: {response.headers.get('Content-Type')}"
            )
            return empty_df

        holidays = response.json()

        if not holidays:  # Empty JSON list
            warnings.warn(
                f"Holiday API returned empty result for {country_code}-{year}."
            )
            return empty_df

    except requests.RequestException as e:
        warnings.warn(
            f"Failure in the holidays URL request for {country_code}-{year}. "
            f"Request error: {e}"
        )
        return empty_df
    except ValueError as e:
        warnings.warn(
            f"Failed to parse JSON for {country_code}-{year}. Error: {e}"
        )
        return empty_df

    # Apply subdivision filter if provided
    if subdivision_code:
        holidays = [
            h for h in holidays
            if h.get("counties") is None or subdivision_code in h.get("counties", [])
        ]

    # Convert to DataFrame
    df = pl.DataFrame(holidays)

    # Convert date to days since 1970-01-01
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .dt.epoch(time_unit="d")
            .alias("date")
        )

    return df

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


def mean_angle(column, over=None):
    if over is None:
        mean_cos = pl.col(column).radians().cos().mean()
        mean_sin = pl.col(column).radians().sin().mean()
    else:
        mean_cos = pl.col(column).radians().cos().mean().over(over)
        mean_sin = pl.col(column).radians().sin().mean().over(over)

    return pl.arctan2(mean_sin, mean_cos).degrees().mod(360)


def max_separation_angle(df, column):
    df = (
        df.with_columns(pl.col(column).list.concat(pl.col(column) + 180).alias(column))
        .with_columns(
            [
                (pl.col(column) % 360).list.min().alias(name=f"{column}_min_angle"),
            ]
        )
        .with_columns(
            (pl.col(column) % 360 - pl.col(f"{column}_min_angle")).alias(
                f"{column}_normalized"
            ),
        )
        .with_columns(
            [
                # collect, sort, append 360
                (
                    pl.col(f"{column}_normalized")
                    .list.sort()
                    .list.concat(pl.lit([360]))
                ).alias(f"{column}_angle_sorted"),
            ]
        )
        .with_columns(
            [
                # arc differences
                pl.col(f"{column}_angle_sorted")
                .list.diff(null_behavior="drop")
                .alias(f"{column}_arc_angle")
            ]
        )
        .with_columns(
            [
                # max separation = max arc/2 + angle at max
                (
                    pl.col(f"{column}_arc_angle").list.max() / 2
                    + pl.col(f"{column}_angle_sorted").list.get(
                        pl.col(f"{column}_arc_angle").list.arg_max()
                    )
                    + pl.col(f"{column}_min_angle")
                ).alias(f"{column}_max_separation_angle")
            ]
        )
    )
    return df[f"{column}_max_separation_angle"]


def hash_file(path, chunk_size=8192):
    """Compute MD5 hash of a single file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(folder_path, chunk_size=8192):
    """Compute a combined MD5 hash of all .txt files in a folder (order-independent)."""
    hashes = []
    for f in os.listdir(folder_path):
        if f.endswith(".txt"):
            file_path = os.path.join(folder_path, f)
            hashes.append(hash_file(file_path, chunk_size))
    return hashlib.md5("".join(sorted(hashes)).encode()).hexdigest()


def hash_zip(zip_path, chunk_size=8192):
    """Compute a combined MD5 hash of all .txt files in a zip archive (order-independent)."""
    hashes = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for f in z.namelist():
            if f.endswith(".txt"):
                h = hashlib.md5()
                with z.open(f) as file:
                    for chunk in iter(lambda: file.read(chunk_size), b""):
                        h.update(chunk)
                hashes.append(h.hexdigest())
    return hashlib.md5("".join(sorted(hashes)).encode()).hexdigest()


def hash_path(path, chunk_size=8192):
    """Compute hash of a path: file, folder, or zip."""
    if os.path.isdir(path):
        return hash_folder(path, chunk_size)
    elif zipfile.is_zipfile(path):
        return hash_zip(path, chunk_size)
    elif os.path.isfile(path):
        return hash_file(path, chunk_size)
    else:
        raise ValueError(f"{path} is not a file, folder, or zip archive")


def compare_paths(path1, path2):
    """Compare any two paths by content."""
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
