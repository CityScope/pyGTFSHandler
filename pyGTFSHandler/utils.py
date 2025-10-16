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
import shutil

# import pygeohash

"TODO: if filter_by_id_column ids come with file_number then filter by file_number too and if not dont and allow mix in the same list"

EPOCH = date(1970, 1, 1)

id_cols = ["trip_id", "service_id", "route_id", "stop_id", "shape_id", "parent_station"]
mandatory_cols = ["trip_id", "service_id", "stop_id"]

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


def sanitize_csv_quotes(path: str) -> str:
    """
    Cleans malformed CSV quote usage in a GTFS file (e.g., stops.txt).
    Fixes unescaped quotes and ensures each line has balanced quotes.

    The cleaned data is written back to the same file, with a backup created as `filename.bak`.

    Args:
        path (str): Path to the GTFS .txt file.

    Returns:
        str: The original file path (after cleaning).
    """
    backup_path = path + ".bak"
    shutil.copy(path, backup_path)

    cleaned_lines = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Remove BOM or nulls
            line = line.replace("\x00", "").replace("\ufeff", "").strip("\r\n")

            # If the line has unbalanced quotes, fix them
            quote_count = line.count('"')
            if quote_count % 2 != 0:
                # Try to fix lines with unescaped internal quotes
                line = re.sub(r'(?<!")"(?![";,])', '""', line)

            # Handle stray leading/trailing quotes that break fields
            # e.g. ""Sixth Street Garage" → "Sixth Street Garage"
            line = re.sub(r'^"+', '"', line)
            line = re.sub(r'"+$', '"', line)

            # Clean common broken GTFS patterns like ""Stop Name or Stop Name""
            line = re.sub(r'""+', '"', line)

            cleaned_lines.append(line)

    # Write cleaned data back to the same file
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(cleaned_lines))

    return path


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
        schema_dict = {
            "route_id": str,
            "service_id": str,
            "trip_id": str,
        }
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
            "agency_id": str,
            "route_short_name": str,
            "route_long_name": str,
            "route_type": str,
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
        schema_dict = {"service_id": str, "date": int, "exception_type": str}
    elif "frequencies.txt" in str(path):
        schema_dict = {
            "trip_id": str,
            "start_time": str,
            "end_time": str,
            "headway_secs": int,
        }
    elif "shapes.txt" in str(path):
        schema_dict = {
            "shape_id": str,
            "shape_pt_sequence": int,
            "shape_pt_lat": float,
            "shape_pt_lon": float,
            "shape_dist_traveled": float,
        }
    elif "agency.txt" in str(path):
        schema_dict = {
            "agency_id": str,
        }

    else:
        raise Exception(f"File {path} not implemented.")

    return schema_dict


def read_csv_lazy(
    path: str, schema_overrides: dict | None = None, file_id: int | None = None, check_files:bool=False
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
    if not os.path.isfile(path):
        return None

    if check_files:
        # Open the CSV file and create a CSV reader object
        with open(path, 'r') as file:
            row_count = len(list(csv.reader(file)))

        if row_count > 0:
            row_count -= 1

    # Lazily scan CSV with optional column selection
    if check_files:
        clean_path = sanitize_csv_quotes(path)
        lf = pl.scan_csv(clean_path, infer_schema=False, raise_if_empty=False, truncate_ragged_lines=True)
    else:
        lf = pl.scan_csv(path, infer_schema=False, raise_if_empty=False, truncate_ragged_lines=True)

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
    lf = lf.with_columns(
        pl.lit(gtfs_name).alias("gtfs_name"), pl.lit(file_id).alias("file_id")
    )

    columns = lf.collect_schema().names()
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

    if check_files:
        lf = lf.collect()
        if row_count != len(lf):
            warnings.warn(f"{row_count - len(lf)} rows of the file {path} are invalid and have been skipped.")
        
        lf = lf.lazy()

    return lf


def read_csv_list(
    path_list: List[str], schema_overrides: Optional[dict] = None, check_files:bool=False, search_files:bool=False
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
            folder, file = os.path.split(path)
            new_path = search_file(folder,file)
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
                path_list[i], schema_overrides=schema_overrides, file_id=i, check_files=check_files
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


def search_file(path, file):
    """
    Recursively searches for the first file that matches the given filename
    in the directory and its subdirectories.

    Args:
        path (str): The root directory to start searching from.
        file (str): The filename to search for (case-sensitive).

    Returns:
        str | None: The full path of the first matching file, or None if not found.
    """
    for root, dirs, files in os.walk(path):
        if file in files:
            return os.path.join(root, file)
    return None


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


def normalize_route_type(route_type):
    if isinstance(route_type, int):
        return route_type
    elif isinstance(route_type, str):
        if route_type == "tram":
            route_type = 0

        elif route_type == "subway":
            route_type = 1

        elif route_type == "rail":
            route_type = 2

        elif route_type == "bus":
            route_type = 3

        elif route_type == "ferry":
            route_type = 4

        elif (
            (route_type == "cable car")
            or (route_type == "cable_car")
            or (route_type == "cable-car")
            or (route_type == "cablecar")
        ):
            route_type = 5

        elif route_type == "gondola":
            route_type = 6

        elif route_type == "funicular":
            route_type = 7
        else:
            raise Exception(
                f"Got route_type {route_type} but accepted values are tram, subway, rail, bus, ferry, cable car, gondola and funicular"
            )

    else:
        raise Exception(
            f"Route type {route_type} with dtype {type(route_type)} not implemented"
        )

    return route_type


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
