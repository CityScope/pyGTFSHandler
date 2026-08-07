# -*- coding: utf-8 -*-
"""GTFS file discovery, per-field parsing/normalization, and structural
validation (missing files, malformed rows, folder/zip unpacking).

Why this module exists and how it's organized:
-----------------------------------------------
- **Structural checks** (`preprocess_gtfs`, `MANDATORY_FILES`/
  `FILE_PAIRS`): before any file is parsed at all, this confirms the
  mandatory GTFS files (or an acceptable alternative, e.g. `calendar.txt`
  *or* `calendar_dates.txt`) are present, and that a zip/folder is laid out
  as expected (`unzip` transparently handles zips with all files at the
  archive root vs. nested one level under a single subfolder).
- **Row-level parsing** (`validate_and_load_csv`, `try_parse_line`,
  `normalize_df`): a slower, Python-level per-row parser used only when
  `check_files=True`, which tolerates ragged/malformed rows (wrong column
  count, garbage values in a typed column) by dropping just the offending
  row (with a warning) rather than failing the whole file, and can write a
  sibling `_errors.txt` alongside the source file with the excluded rows for
  inspection. The fast path (`check_files=False`, used internally once a
  feed's structure is already known-good) skips straight to `pl.scan_csv`.
- **Field-level normalization** (`parse_date`, `parse_time`,
  `normalize_route_type`/`extended_to_standard_route_type`,
  `normalize_string`): the actual per-value parsing/cleaning rules (date
  formats, GTFS standard vs. "extended"/Transmodel route_type codes, accent
  and whitespace normalization for id-like strings) that both the row-level
  parser above and the fast polars-native loaders in `utils.py` rely on for
  a single, shared definition of "what counts as valid."

This module intentionally has no notion of GTFS *semantics* beyond
individual files (e.g. it doesn't check that a `stop_times.txt` `trip_id`
exists in `trips.txt` -- that's a cross-file referential-integrity concern
handled where both files are already loaded, e.g. in
`models/stop_times.py`/`models/stops.py`).
"""

import polars as pl
import pandas as pd
import geopandas as gpd
from datetime import datetime
import os
import csv
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
import zipfile
import shutil
import unicodedata
import copy

def parse_date(d):
    """
    Parse a date from various formats and return an int in YYYYMMDD format.
    """
    if isinstance(d, int):
        d = str(d)

    date_formats = [
        "%Y%m%d", "%Y-%m-%d", "%Y%m%d", "%Y%m-%d",
        "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y",
        "%m-%d-%Y", "%d.%m.%Y",
    ]

    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(d, fmt)
            return str(parsed_date.strftime("%Y%m%d"))
        except ValueError:
            continue

    raise ValueError(f"Could not parse date: {d}")

# ------------------------------
# TIME PARSER
# ------------------------------
def parse_time(t: str) -> str:
    """
    Parse a time string and return HH:MM:SS.
    Rules:
      - With colons (:): parse flexibly (e.g., "7:1" -> "07:01:00").
      - Without colons: only 4 or 6 digits are allowed:
            4 digits (HHMM) -> HH:MM:00
            6 digits (HHMMSS) -> HH:MM:SS
      Supports hours up to 47.
    """
    if isinstance(t, int):
        t = str(t)
    t = t.strip().lower().replace(',', '.')

    # Handle AM/PM (unchanged)
    ampm_match = re.match(r'(\d{1,2}):?(\d{1,2})?:?(\d{1,2})?\s*(am|pm)', t)
    if ampm_match:
        h, m, s, meridiem = ampm_match.groups()
        h, m, s = int(h), int(m or 0), int(s or 0)
        if meridiem == 'pm' and h < 12:
            h += 12
        if meridiem == 'am' and h == 12:
            h = 0
        if h > 47:
            raise ValueError(f"Hour value over 47: {h}")
        return f"{h:02}:{m:02}:{s:02}"

    # Case 1: String has colons → flexible parsing
    if ':' in t:
        parts = t.split(':')
        parts = [p.zfill(2) if p else '00' for p in parts]
        while len(parts) < 3:
            parts.append('00')

        h, m, s = map(int, parts[:3])
        if m > 59 or s > 59:
            raise ValueError(f"Invalid time value: {t}")
        if h > 47:
            raise ValueError(f"Invalid hour value: {t} is over 47 hours")

        return f"{h:02}:{m:02}:{s:02}"

    # Case 2: No colons → must be 4 or 6 digits
    digits = ''.join(c for c in t if c.isdigit())
    if not digits:
        raise ValueError(f"Could not parse time: {t}")

    if len(digits) == 4:  # HHMM
        h, m, s = int(digits[:2]), int(digits[2:4]), 0
    elif len(digits) == 6:  # HHMMSS
        h, m, s = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
    else:
        raise ValueError(f"Invalid time format (must be 4 or 6 digits): {t}")

    if m > 59 or s > 59:
        raise ValueError(f"Invalid time value: {t}")
    if h > 47:
        raise ValueError(f"Invalid hour value: {t} is over 47 hours")

    return f"{h:02}:{m:02}:{s:02}"

# ------------------------------
# SCHEMA DEFINITION
# ------------------------------
def get_df_schema_dict(path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Returns the expected Polars dtype schema and mandatory columns for a GTFS file.

    Used throughout `models/*.py` to pass `schema_overrides` to
    `utils.io.read_csv_list`/`read_csv_lazy` and to know which columns
    `check_files` must validate as present/non-null.

    Args:
        path: A GTFS filename or path (only the basename, e.g.
            `"stops.txt"`, is inspected -- any directory portion or
            extension besides `.txt` is normalized away).

    Returns:
        Tuple[Dict[str, Any], List[str]]: `(schema_dict, mandatory_cols)`
        where `schema_dict` maps column name to Python/Polars type and
        `mandatory_cols` lists the GTFS-required columns for that file.
    """
    path = os.path.splitext(path)[0]
    path += ".txt"
    if "stops.txt" in str(path):
        schema_dict = {"stop_id": str, "stop_name": str, "stop_lat": float, "stop_lon": float}
        mandatory_cols = ["stop_id", "stop_lat", "stop_lon"]
    elif "trips.txt" in str(path):
        schema_dict = {"route_id": str, "service_id": str, "trip_id": str, "direction_id": int}
        mandatory_cols = ["route_id", "service_id", "trip_id"]
    elif "stop_times.txt" in str(path):
        schema_dict = {
            "trip_id": str,
            "arrival_time": "time|None",
            "departure_time": "time|None",
            "stop_id": str,
            "stop_sequence": int
        }
        mandatory_cols = ["trip_id", "arrival_time", "departure_time", "stop_id"]
    elif "routes.txt" in str(path):
        schema_dict = {"route_id": str, "agency_id": str, "route_short_name": str,
                       "route_long_name": str, "route_type": "route_type"}
        mandatory_cols = ["route_id"]
    elif "calendar.txt" in str(path):
        schema_dict = {
            "service_id": str, "monday": "int|bool", "tuesday": "int|bool", "wednesday": "int|bool",
            "thursday": "int|bool", "friday": "int|bool", "saturday": "int|bool", "sunday": "int|bool",
            "start_date": "date", "end_date": "date"
        }
        mandatory_cols = ["service_id", "monday", "tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"]
    elif "calendar_dates.txt" in str(path):
        schema_dict = {"service_id": str, "date": "date", "exception_type": "exception_type"}
        mandatory_cols = ["service_id","date","exception_type"]
    elif "frequencies.txt" in str(path):
        schema_dict = {
            "trip_id": str,
            "start_time": "time",
            "end_time": "time",
            "headway_secs": "seconds",
        }
        mandatory_cols = ["trip_id","start_time","end_time","headway_secs"]
    elif "shapes.txt" in str(path):
        schema_dict = {
            "shape_id": str,
            "shape_pt_sequence": int,
            "shape_pt_lat": float,
            "shape_pt_lon": float,
            "shape_dist_traveled": float,
        }
        mandatory_cols = ["shape_id","shape_pt_lat","shape_pt_lon"]
    elif "agency.txt" in str(path):
        schema_dict = {
            "agency_id": str,
        }
        mandatory_cols = ["agency_id"]
    else:
        raise Exception(f"File {path} not implemented.")
    return schema_dict, mandatory_cols

# ------------------------------
# CSV FORMAT DETECTION
# ------------------------------
def normalize_string(s: str, *, strict: bool = True) -> str:
    """
    Normalize a string:

    - Converts accented characters to ASCII.
    - Strips leading/trailing whitespace in all cases.

    strict=True:
        - Lowercase
        - Replace spaces (and multiple consecutive spaces) with a single underscore
        - Keep only a-z, 0-9, and underscores
        - Discard all other symbols

    strict=False:
        - Preserve case and spaces
        - Keep URL-safe characters: a-zA-Z0-9-_.~:/?#[]@!$&'()*+,;=
        - Collapse multiple spaces into a single space

    Parameters:
        s (str): Input string
        strict (bool): Whether to apply strict rules

    Returns:
        str: Normalized string
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")

    # Normalize accented characters
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.strip()  # Strip whitespace in both cases

    if strict:
        s = s.lower()
        # Replace one or more spaces with a single underscore
        s = re.sub(r"\s+", "_", s)
        # Keep only a-z, 0-9, and underscores
        s = re.sub(r"[^a-z0-9_]", "", s)
    else:
        # Preserve case and spaces, keep URL-safe characters
        s = re.sub(r"[^a-zA-Z0-9\-_.~:/?#\[\]@!$&'()*+,;=\s]", "", s)
        # Collapse multiple spaces into a single space
        s = re.sub(r"\s{2,}", " ", s)

    # Final strip to remove any leading/trailing underscores or spaces
    return s.strip()

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


def _try_normalize_route_type_name(route_type):
    """Like `normalize_route_type`, but returns None instead of raising for
    a string that doesn't match any known route_type name (used when
    bulk-resolving a routes.txt column, where an unrecognized value should
    be reported once as a warning rather than aborting the whole load)."""
    try:
        return normalize_route_type(route_type)
    except Exception:
        return None


def normalize_route_type(route_type: Union[int, str]) -> Optional[int]:
    """Normalizes a route type (int code or name string) to its integer GTFS code.

    Accepts either the standard/extended numeric `route_type` code
    (returned as-is) or a human-friendly name (e.g. `"bus"`, `"subway"`),
    which is mapped to the corresponding standard code. Used by `Feed`
    and `Routes.load` to normalize the caller-supplied `route_types`
    filter before comparing it against `routes.txt`.

    Args:
        route_type: An `int` GTFS route type code, or a `str` that is
            either an integer-like code or a recognized route type name.

    Returns:
        int | None: The normalized integer route type code, or `None` if
        `route_type` could not be parsed as an int and did not match a
        known name.
    """
    if isinstance(route_type,int):
        return route_type 

    try:
        return int(route_type)
    except:
        None 
    
    if isinstance(route_type, str):
        if route_type == "tram":
            route_type = 0

        elif (route_type == "subway") or (route_type == "metro") or (route_type == "underground"):
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

def extended_to_standard_route_type(route_type: int) -> int | None:
    """
    Convert any GTFS route_type or Extended GTFS (Transmodel) route_type
    into a standard GTFS route_type (0–7).

    Returns:
        int in {0..7} or None if not mappable.
    """

    # --- If already a standard GTFS type, return unchanged ---
    if route_type in {-1, 0, 1, 2, 3, 4, 5, 6, 7}:
        return route_type

    # --- Extra custom values you asked to convert ---
    if route_type == 11:   # Trolleybus
        return 3           # Bus
    if route_type == 12:   # Monorail
        return 2           # Rail

    # --- Rail services (100–117) ---
    if 100 <= route_type <= 117:
        return 2

    # --- Coach (200–209) → Bus ---
    if 200 <= route_type <= 209:
        return 3

    # --- Urban Rail, Metro, Underground, Monorail ---
    urban_rail_map = {
        400: 2,  # urban rail → rail
        401: 1,  # metro
        402: 1,  # underground/subway
        403: 2,  # urban rail
        404: 2,  # all urban rail
        405: 2,  # monorail → rail (per your instruction)
    }
    if route_type in urban_rail_map:
        return urban_rail_map[route_type]

    # --- Bus (700–716) & Trolleybus (800) ---
    if 700 <= route_type <= 716:
        return 3
    if route_type == 800:
        return 3  # electric bus

    # --- Tram / Light rail (900–906) ---
    if 900 <= route_type <= 906:
        return 0

    # --- Water / Ferry ---
    if route_type in {1000, 1200}:
        return 4

    # --- Air (no GTFS mapping) ---
    if route_type == 1100:
        return None

    # --- Aerial lift services (1300–1307) ---
    lift_map = {
        1300: 6,  # aerial lift, gondola-like
        1301: 6,
        1302: 5,  # cable car
        1303: 5,  # elevator → closest = cable car
        1304: 6,  # chair lift
        1305: 6,  # drag lift
        1306: 6,
        1307: 6,
    }
    if route_type in lift_map:
        return lift_map[route_type]

    # --- Funicular ---
    if route_type == 1400:
        return 7

    # --- Taxi (1500–1507) or Misc (1700, 1702): no GTFS mapping ---
    if 1500 <= route_type <= 1507 or route_type in {1700, 1702}:
        return None

    # --- Unknown ---
    return None


def route_type_to_str(route_type: int) -> str:
    """Converts a standard GTFS `route_type` integer code to its lowercase name.

    Inverse of the numeric side of `normalize_route_type`.

    Args:
        route_type: A standard GTFS route type code (`int`), e.g. `3`.

    Returns:
        str: The lowercase route type name, e.g. `"bus"`.

    Raises:
        Exception: If `route_type` is not an `int`.
    """
    if not isinstance(route_type, int):
        raise Exception(f"route_type must be an int, got {type(route_type)}")

    mapping = {
        -1: "None",
        0: "tram",
        1: "subway",
        2: "rail",
        3: "bus",
        4: "ferry",
        5: "cable car",
        6: "gondola",
        7: "funicular",
    }

    if route_type not in mapping:
        raise Exception(f"route_type {route_type} is not in range 0–7")

    return mapping[route_type]

