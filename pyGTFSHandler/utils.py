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
import pygeohash

EPOCH = date(1970, 1, 1)


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


def get_df_schema_dict(path) -> dict:
    """
    Returns a dictionary specifying the expected data types for mandatory columns
    in a GTFS (General Transit Feed Specification) file.

    This is useful for consistent schema enforcement when reading GTFS .txt files.

    Parameters:
    -----------
    path : str
        The file path or file name of a GTFS component (e.g., 'stops.txt').

    Returns:
    --------
    dict
        A dictionary mapping mandatory column names to their expected data types.
    """
    if "stops.txt" in str(path):
        schema_dict = {
            "stop_id": str,
            "stop_name": str,
            "stop_lat": float,
            "stop_lon": float,
        }
    elif "trips.txt" in str(path):
        schema_dict = {"route_id": str, "service_id": str, "trip_id": str}
    elif "stop_times.txt" in str(path):
        schema_dict = {
            "trip_id": str,
            "arrival_time": str,
            "departure_time": str,
            "stop_id": str,
            "stop_sequence": int,
        }
    elif "routes.txt" in str(path):
        schema_dict = {
            "route_id": str,
            "route_short_name": str,
            "route_long_name": str,
            "route_type": int,
        }
    elif "calendar.txt" in str(path):
        schema_dict = {
            "service_id": str,
            "monday": int,
            "tuesday": int,
            "wednesday": int,
            "thursday": int,
            "friday": int,
            "saturday": int,
            "sunday": int,
            "start_date": int,
            "end_date": int,
        }
    elif "calendar_dates.txt" in str(path):
        schema_dict = {"service_id": str, "date": int, "exception_type": int}
    elif "frequencies.txt" in str(path):
        schema_dict = {
            "trip_id": str,
            "start_time": str,
            "end_time": str,
            "headway_secs": int,
        }

    else:
        raise Exception(f"File {path} not implemented.")

    return schema_dict


def read_csv_lazy(path: str, schema_overrides: dict = None) -> pl.LazyFrame:
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
    # Lazily scan CSV with optional column selection
    lf = pl.scan_csv(path, infer_schema=False)

    # Apply custom normalization (assuming normalize_df is defined elsewhere)
    lf = normalize_df(lf)

    # Apply schema overrides if specified
    if schema_overrides:
        for col, dtype in schema_overrides.items():
            if col in lf.collect_schema().names():
                lf = lf.with_columns(pl.col(col).cast(dtype))

    # Extract GTFS directory name from path (e.g., 'gtfs_file' from '/home/xyz/gtfs_file/stops.txt')
    gtfs_name = os.path.basename(os.path.dirname(path))

    # Add gtfs_name as a literal column
    lf = lf.with_columns(pl.lit(gtfs_name).alias("gtfs_name"))

    return lf


def read_csv_list(
    path_list: List[str], schema_overrides: Optional[dict] = None
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
    return pl.concat(
        [read_csv_lazy(path, schema_overrides=schema_overrides) for path in path_list],
        how="diagonal_relaxed",
    )


def normalize_string(s: str) -> str:
    """
    Normalize a string by:
    - Converting accented characters to their ASCII equivalents.
    - Lowercasing all characters.
    - Removing whitespace.
    - Removing all non-alphanumeric characters except underscores.

    This function is used to normalize column names.

    Parameters:
        s (str): The input string.

    Returns:
        str: The normalized string.
    """
    s = unicodedata.normalize("NFKD", s)  # Decompose Unicode characters
    s = s.encode("ascii", "ignore").decode("ascii")  # Remove non-ASCII characters
    s = s.lower()  # Convert to lowercase
    s = re.sub(r"\s+", "", s)  # Remove all whitespace
    s = re.sub(r"[^a-z0-9_]", "", s)  # Keep only a-z, 0-9, and underscore
    return s


def normalize_df(lf: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """
    Normalize both column names and string values in a Polars LazyFrame or DataFrame.

    Column names:
    - Lowercased, stripped of spaces and special characters.

    String columns (pl.Utf8):
    - Strip leading/trailing whitespace.
    - Remove diacritical marks (accents) from characters.
    - Optionally preserve uppercase (used here to keep valid URL casing).
    - Retain all valid URL characters.
    - Remove invalid characters.

    Parameters:
        lf (pl.LazyFrame | pl.DataFrame): Input LazyFrame or DataFrame.

    Returns:
        pl.LazyFrame | pl.DataFrame: The normalized LazyFrame or DataFrame.
    """
    # Get the schema to avoid triggering expensive computation in lazy mode
    schema = lf.collect_schema()
    column_names = schema.names()

    # Normalize all column names
    normalized_column_names = [normalize_string(col) for col in column_names]
    rename_map = dict(zip(column_names, normalized_column_names))
    lf = lf.rename(rename_map)

    # Normalize string column values (for columns with Utf8 type)
    for old_name, new_name in zip(column_names, normalized_column_names):
        dtype = schema.get(old_name)
        if dtype == pl.Utf8:
            expr = (
                pl.col(new_name)
                .str.strip_chars()  # Trim leading/trailing whitespace
                # .str.to_lowercase()  # Uncomment to force lowercase values
                # Remove diacritics from characters (accented letters)
                .str.replace_all(r"[áàãâäåāÁÀÃÂÄÅĀ]", "a")
                .str.replace_all(r"[éèêëēėęÉÈÊËĒĖĘ]", "e")
                .str.replace_all(r"[íìîïīįıÍÌÎÏĪĮ]", "i")
                .str.replace_all(r"[óòõôöøōÓÒÕÔÖØŌ]", "o")
                .str.replace_all(r"[úùûüūÚÙÛÜŪ]", "u")
                .str.replace_all(r"[çćčÇĆČ]", "c")
                .str.replace_all(r"[ñńÑŃ]", "n")
                .str.replace_all(r"[ß]", "ss")
                .str.replace_all(r"[ÿŸ]", "y")
                .str.replace_all(r"[žźżŽŹŻ]", "z")
                .str.replace_all(
                    r"\s+", "_"
                )  # Replace internal spaces with underscores
                # Keep only valid URL characters
                .str.replace_all(r"[^a-zA-Z0-9\-_.~:/?#\[\]@!$&'()*+,;=]", "")
            )
            lf = lf.with_columns(expr.alias(new_name))

    return lf


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
    holidays = requests.get(url).json()
    if subdivision_code:
        df = pl.DataFrame(
            [
                h
                for h in holidays
                if h["counties"] is None or subdivision_code in h["counties"]
            ]
        )
    else:
        df = pl.DataFrame(holidays)

    df = df.with_columns(
        [
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d")
            .dt.epoch(time_unit="d")  # days since 1970-01-01 (int)
            .alias("date"),
        ]
    )

    return df


def geohash(
    df: pl.DataFrame, lat_col: str, lon_col: str, precision: int = 7
) -> pl.DataFrame:
    """
    Add a geohash column to a Polars DataFrame based on latitude and longitude columns.

    Parameters
    ----------
    df : pl.DataFrame
        The input Polars DataFrame containing latitude and longitude data.
    lat_col : str
        The name of the latitude column in the DataFrame.
    lon_col : str
        The name of the longitude column in the DataFrame.
    precision : int, optional
        The precision (length) of the geohash string. Default is 7.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with an additional 'geohash' column containing geohash strings.
    """
    df = df.with_columns(
        pl.struct([lat_col, lon_col])
        .map_elements(
            lambda point: pygeohash.encode(
                point[lat_col], point[lon_col], precision=precision
            ),
            return_dtype=pl.String,
        )
        .alias("geohash")
    )
    return df
