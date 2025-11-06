import polars as pl
import pandas as pd
import geopandas as gpd
from datetime import datetime
import os
import csv
import re
from typing import List, Dict, Any, Tuple
import warnings 
import zipfile 
import shutil 
import unicodedata 

MANDATORY_FILES = ["stops.txt","trips.txt","stop_times.txt",["calendar.txt","calendar_dates.txt"]]

# ------------------------------
# DATE PARSER
# ------------------------------
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
    if "stops.txt" in str(path):
        schema_dict = {"stop_id": str, "stop_name": str, "stop_lat": float, "stop_lon": float}
        mandatory_cols = ["stop_id", "stop_lat", "stop_lon"]
    elif "trips.txt" in str(path):
        schema_dict = {"route_id": str, "service_id": str, "trip_id": str}
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
def detect_csv_format(sample_text: str, max_lines: int = 1) -> Dict[str, Any]:
    lines = sample_text.strip().splitlines()[:max_lines]
    sample = "\n".join(lines)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
        delimiter, quotechar, doublequote = dialect.delimiter, dialect.quotechar, dialect.doublequote
    except Exception:
        possible_delims = [',', ';', '\t', '|']
        delim_scores = {}
        for d in possible_delims:
            counts = [ln.count(d) for ln in lines if ln.strip()]
            if counts:
                variance = max(counts) - min(counts)
                delim_scores[d] = (sum(counts)/len(counts), variance)
        delimiter = min(delim_scores, key=lambda k: delim_scores[k][1]) if delim_scores else ','
        quote_candidates = ['"', "'"]
        qcounts = {q: sample.count(q) for q in quote_candidates}
        quotechar = max(qcounts, key=qcounts.get) if max(qcounts.values()) > 0 else '"'
        doublequote = (quotechar*2) in sample
    dot_nums = len(re.findall(r'\d+\.\d+', sample))
    comma_nums = len(re.findall(r'\d+,\d+', sample))
    float_point = '.' if dot_nums >= comma_nums else ','
    return {"delimiter": delimiter, "quotechar": quotechar, "doublequote": doublequote, "float_point": float_point}


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

# ------------------------------
# TRY PARSE SINGLE LINE
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


def normalize_route_type(route_type):
    if route_type in [str(i) for i in range(0, 8)]:
        return int(route_type)
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


def try_parse_line(line: str, config: Dict[str, Any], expected_cols: int|None = None, header:list|None=None, schema:dict|None=None, mandatory_columns:list=[]) -> Tuple[List[str]|None, str|None, str|None, bool]:
    parsed = None
    error = ""
    fix = ""
    try:
        parsed = next(csv.reader([line], delimiter=config["delimiter"], quotechar=config["quotechar"], doublequote=config["doublequote"]))
    except Exception as e:
        fixed_line = re.sub(r'(?<=\w)"(?=\w)', "'", line)
        try:
            parsed = next(csv.reader([fixed_line], delimiter=config["delimiter"], quotechar=config["quotechar"], doublequote=config["doublequote"]))
            error += "Quotation error "
            fix += "Replaced embedded \" with ' "
        except Exception:
            return None, f"Quotation error: {e}", "excluded", True
        
    detected_cols = len(parsed)
    if expected_cols is not None and detected_cols != expected_cols:
        fixed_line = re.sub(r'(?<=\w)"(?=\w)', "'", line)
        try:
            parsed = next(csv.reader([fixed_line], delimiter=config["delimiter"], quotechar=config["quotechar"], doublequote=config["doublequote"]))
            if len(parsed) == expected_cols:
                error += f"Expected {expected_cols} cols, got {detected_cols} "
                fix += "Replaced embedded \" with ' "
        except Exception:
            error += f"Expected {expected_cols} cols, got {detected_cols} "
            fix += "Excluded "
            return None, error, fix, True
    
    if schema is not None and header is not None and parsed is not None:
        for col_idx, col_name in enumerate(header):
            if col_name not in schema:
                continue

            dtype = schema[col_name]

            val = parsed[col_idx]
            original = val
            parsed_val = normalize_string(val,strict=False)

            # Skip empty values
            if val is None or str(val).strip() == '': 
                if col_name in mandatory_columns:
                    val = None 
                    original = None
                    if isinstance(dtype, str):
                        if "None" not in dtype:
                            error += f"None value in mandatory column {col_name} "
                            fix += "Excluded "
                            return None, error, fix, True
                else:
                    parsed[col_idx] = None
                    if isinstance(dtype, str):
                        if "None" not in dtype:
                            error += f"None value in column {col_name} "
                            fix += "Using default value. "
                            
                    continue
                
            try:
                if dtype == "date":
                    parsed_val = parse_date(str(val))
                elif dtype == "time":
                    parsed_val = parse_time(str(val))
                elif dtype == "time|None":
                    if val is None:
                        parsed_val = None 
                    else:
                        parsed_val = parse_time(str(val))
                elif dtype == "int|bool":
                    sval = str(val).strip().lower()
                    if sval in ("true", "1"):
                        parsed_val = 1
                    elif sval in ("false", "0"):
                        parsed_val = 0
                    else:
                        parsed_val = int(float(sval))
                elif dtype == int:
                    parsed_val = int(float(val))
                elif dtype == float:
                    parsed_val = float(val)
                elif dtype == "route_type":
                    parsed_val = normalize_route_type(val)
                elif dtype == "exception_type":
                    if val == "added":
                        parsed_val = 1 
                    elif val == "removed":
                        parsed_val = 2 
        
                    parsed_val = int(float(val))
                    if parsed_val != 1 and parsed_val != 2:
                        parsed_val = None 
                        original = None
                        error += f"{parsed_val} not valid for {col_name}. Only [1,2] are valid. "
                        fix += f"Set {col_name} to None. "
                else:
                    parsed_val = str(val)
            except Exception as e:
                error += f"Parse failed for column '{col_name}' value '{original}': {e} "
                if col_name in mandatory_columns:
                    fix += "Excluded "
                    return None, error, fix, True
                else:
                    parsed_val = None 
                    original = None
                    fix += f"Replaced {original} with None "

            # Track modifications
            if str(parsed_val) != str(original):
                error += f"Value in column '{col_name}' modified after parsing "
                fix += f"{original} -> {parsed_val} "

            # Apply modification directly to parsed list
            if parsed_val is None:
                parsed[col_idx] = None 
            else:
                parsed[col_idx] = str(parsed_val)

    if error == "":
        error = None 

    if fix == "":
        fix = None 

    return parsed, error, fix, False


def validate_and_load_csv(path: str, header: bool = True, csv_text=None):
    # Get schema info
    schema_dict, mandatory_cols = get_df_schema_dict(path)

    # Read CSV text if not provided
    if csv_text is None:
        if not os.path.isfile(path):
            raise Exception(f"File {path} does not exist")
        
        folder, file = os.path.split(path)
        path = search_file(folder,file)
        if path is None:
            return None
        
        with open(path, encoding="utf-8") as f:
            csv_text = f.read()

    lines = csv_text.splitlines()
    config = detect_csv_format(csv_text)

    expected_cols = None
    colum_names = None

    if header:
        header_line = lines[0]
        lines = lines[1:]
        orig_colum_names, error_msg, fix, error = try_parse_line(header_line, config)
        if error or (orig_colum_names is None):
            raise Exception(f"Error parsing header of file {path}: {error_msg} {fix}")
        elif error_msg is not None:
            warnings.warn(f"Warning parsing header of file {path}: {error_msg} {fix}")
        else:
            error_msg = ""

        colum_names = []
        for i in orig_colum_names:
            new_col = normalize_string(i)
            if new_col != i:
                error_msg += f"Column name {i} changed to {new_col}"
            
            colum_names.append(new_col)

        if mandatory_cols is not None:
            for i in mandatory_cols:
                if i not in colum_names:
                    raise Exception(f"Column {i} not in file {path}")
            
        expected_cols = len(colum_names)
        header_error = error_msg 

    # Build initial Polars DataFrame with line content
    lines_df = pl.DataFrame({
        "line_number": range(1, len(lines) + 1),
        "content": lines
    })

    # Parse each line into structured columns
    lines_df = lines_df.with_columns(
        pl.col("content").map_elements(
            lambda line: {
                "parsed": try_parse_line(line, config, expected_cols, colum_names, schema_dict,mandatory_cols)[0],
                "error": try_parse_line(line, config, expected_cols, colum_names, schema_dict,mandatory_cols)[1],
                "fix": try_parse_line(line, config, expected_cols, colum_names, schema_dict,mandatory_cols)[2],
                "excluded": try_parse_line(line, config, expected_cols, colum_names, schema_dict,mandatory_cols)[3],
            },
            return_dtype=pl.Struct({
                "parsed": pl.List(pl.Utf8),
                "error": pl.Utf8,
                "fix": pl.Utf8,
                "excluded": pl.Boolean,
            })
        ).alias("parsed_struct")
    ).unnest("parsed_struct")

    # Build final DataFrame with parsed columns and df_cols as column names
    parsed_cols_df = lines_df.filter(~pl.col("excluded")).select(['line_number','parsed'])
    if colum_names is not None:
        parsed_cols_df = lines_df.select(['line_number','parsed'])
        for i, col_name in enumerate(colum_names):
            parsed_cols_df = parsed_cols_df.with_columns(
                pl.Series(col_name, lines_df["parsed"].list.get(i).cast(pl.Utf8))
            )
        
        parsed_cols_df = parsed_cols_df.drop('parsed')

    errors_df = lines_df.select(['line_number','error','fix','excluded']).drop_nulls("error")
    if len(errors_df.filter("excluded")) > 0:
        warnings.warn(f"{len(errors_df.filter("excluded"))} rows of file {path} have failed while parsing.")

    if header_error != "":
        errors_df = pl.concat([
            pl.DataFrame({'line_number':[0],'error':['Error parsing header'],'fix':[header_error],'excluded':[False]}),
            errors_df
        ])

    return parsed_cols_df, errors_df


def preprocess_gtfs(path,output_folder, mandatory_files = MANDATORY_FILES):
    log = ""
    delete_path = None
    if os.path.isfile(path):
        # Extract the ZIP
        extraction_folder, ext = os.path.splitext(path)
        os.makedirs(extraction_folder, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)

        logs += f"Extracted file {path} to {extraction_folder} \n"
        delete_path = extraction_folder
        path = extraction_folder

    # elif os.path.isdir(path):
    #     found = False
    #     for file_name in os.listdir(path):
    #         file_path = os.path.join(path, file_name)
    #         if os.path.isfile(file_path) and file_name.startswith(file_name):
    #             found = True
    #             break

    #     if not found:
    #         shutil.make_archive(base_name=path, format='zip', root_dir=path)

    else:
        raise Exception(f"Path {path} does not exist")
    
    for file in MANDATORY_FILES:
        if search_file(path,file) is None:
            raise Exception(f"File {file} not found in folder path. This GTFS might be broken.")
        
    gtfs = {}
    gtfs_errors = {}
    for root, dirs, files in os.walk(path):
        for file_name in files:
            file_name, ext = os.path.splitext(file_name)
            if ext == ".txt" or ext == ".csv":  
                file_path = os.path.join(root, file_name + ext)
            else:
                log += f"Can't read file {file_path}. Not a txt file. \n"
                continue

            try:
                content, errors = validate_and_load_csv(file_name,header=True)
                if (len(content) == 0) and (file_name + ".txt" in mandatory_files):
                    warnings.warn(f"File {os.path.join(path,file_path+ext)} is empty")
                    log += f"File {os.path.join(path,file_path+ext)} is empty. \n"

                gtfs[file_name] = content 
                gtfs_errors[file_name] = errors 
            except Exception as e:
                log += f"Error reading file {file_path}. {e}. \n"
                if (file_name + ".txt") in mandatory_files:
                    raise Exception(f"Error reading file {os.path.join(path,file_path+ext)}. {e}. \n")

    check_id_files_list = [["stops","stop_times"],["trips","stop_times"],["trips","routes"],["agency","routes"],["trips","shapes"]]
    check_id_col_list = ["stop_id","trip_id","route_id","agency_id","shape_id"]

    for i in range(len(check_id_col_list)):
        check_id_files = check_id_files_list[i]
        check_id_files_mandatory = []
        check_id_files_final = []
        for check_id_file in check_id_files:
            if check_id_file in gtfs.keys():
                check_id_files_final.append(check_id_file) 

        if len(check_id_files_final) <= 1:
            continue

        check_id_files = check_id_files_final
        for check_id_file in check_id_files:
            if check_id_file in mandatory_files:
                check_id_files_mandatory.append(check_id_file)

        if len(check_id_files_mandatory) == 0:
            check_id_files_mandatory = check_id_files 


        check_id_col = check_id_col_list[i]
        # Initialize as a set
        ids = None  # start with None to handle first intersection

        for check_id_file in check_id_files_mandatory:
            # Get the column values as a set of strings
            current_ids = set(map(str, gtfs[check_id_file].select(check_id_col)[check_id_col].to_list()))
            if ids is None:
                # For the first file, just initialize ids
                ids = current_ids
            else:
                # Intersect with the existing ids
                ids = ids.intersection(current_ids)

        if ids is not None:
            for check_id_file in check_id_files:
                not_in_id = set(gtfs[check_id_file].select(check_id_col)[check_id_col].to_list()) - ids
                if len(not_in_id) > 0:
                    log += f"The id column {check_id_col} in {os.path.join(path,check_id_file)} has some ids not used elsewere. Excluding the following {check_id_col} values: \n" 
                    log += f"{not_in_id} \n" 
                    warnings.warn(f"len{not_in_id} {check_id_col} ids in {os.path.join(path,check_id_file)} are not used elsewere and are being excluded.")
                    gtfs[check_id_file] = gtfs[check_id_file].filter(pl.col(check_id_col).is_in(ids))
                    gtfs_errors[check_id_file] = gtfs_errors[check_id_file].join(gtfs[check_id_file],on="line_number",how="left")

    basename = os.path.basename(path)
    # Ensure the output folder exists
    os.makedirs(os.path.join(output_folder, basename), exist_ok=True)


    for file_name, df in gtfs.items():
        base_path = os.path.join(output_folder, basename, f"{file_name}.txt")
        df.write_csv(
            base_path,
            separator=",",          # use comma as field separator
            quote_char='"',         # wrap strings with double quotes
            decimal_separator=".",  # use '.' for decimals
            include_header=True
        )
        logs += f"Created file {base_path} \n"

    for file_name, df in gtfs_errors.items():
        error_path = os.path.join(output_folder, basename, f"{file_name}_errors.txt")
        df.write_csv(
            error_path,
            separator=",",
            quote_char='"',
            decimal_separator=".",
            include_header=True
        )
        logs += f"Created file {error_path} \n"

    if delete_path is not None:
        shutil.rmtree(delete_path)
        logs += f"Deleted file {delete_path} \n"

    with open(os.path.join(output_folder, basename, "logs.txt"), "w", encoding="utf-8") as f:
        f.write(logs)

    return os.path.join(output_folder, basename)
    