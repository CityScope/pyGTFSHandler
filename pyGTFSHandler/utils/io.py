# -*- coding: utf-8 -*-
"""File-system-level GTFS IO: locating files inside a folder/zip, unzipping,
lazily scanning CSVs into polars, and the slower row-tolerant CSV parser
used when `check_files=True`.

Why this module exists and how it's organized:
-----------------------------------------------
- **Discovery/unzipping** (`search_file`, `unzip`, `preprocess_gtfs`,
  `MANDATORY_FILES`/`PREFERENTIAL_FILES`/`FILE_PAIRS`): finds a named GTFS
  file inside a folder (case-sensitive match first, falling back to a
  case-insensitive one), transparently unzips a `.zip` feed (handling both
  "files at the archive root" and "files nested one level under a single
  subfolder" layouts), and confirms a feed's mandatory files (or an
  acceptable alternative, e.g. `calendar.txt` *or* `calendar_dates.txt`) are
  present before anything is parsed.
- **Fast path** (`read_csv_lazy`/`read_csv_list`, in this same module):
  the common case (`check_files=False`, used once a feed's structure is
  already known-good) -- straight to `pl.scan_csv`, no per-row Python
  parsing.
- **Tolerant path** (`validate_and_load_csv`, `try_parse_line`,
  `detect_csv_format`): a slower, Python-level per-row parser used only when
  `check_files=True`, which tolerates ragged/malformed rows (wrong column
  count, garbage values in a typed column) by dropping just the offending
  row (with a warning) rather than failing the whole file, and can write a
  sibling `_errors.txt` alongside the source file with the excluded rows for
  inspection. It delegates per-value parsing rules (dates, times, route
  types, id-string normalization) to `gtfs_checker.py`, which has no
  awareness of files/paths at all -- keeping "how do I read a file" and
  "what does a valid value look like" as two separate concerns.
"""

import os
import csv
import re
import shutil
import zipfile
import unicodedata
import copy
import warnings
from typing import Any, Dict, List, Tuple, Optional

import polars as pl

from . import gtfs_checker

ID_COLS = ["trip_id", "service_id", "route_id", "stop_id", "shape_id", "parent_station", "agency_id"]
MANDATORY_COLS = ["trip_id", "service_id", "stop_id"]

MANDATORY_FILES = ["stops.txt","trips.txt","stop_times.txt",["calendar.txt","calendar_dates.txt"]]

PREFERENTIAL_FILES = ["routes.txt","frequencies.txt","calendar.txt","calendar_dates.txt"]

IMPLEMENTED_PARSERS = ["stops.txt","trips.txt","stop_times.txt","calendar.txt","calendar_dates.txt","agency.txt","routes.txt","frequencies.txt","shapes.txt"]

FILE_PAIRS = [
    {'files':["stops.txt","stop_times.txt"],'ids':["stop_id"]},
    {'files':["trips.txt","stop_times.txt", "frequencies.txt"],'ids':["trip_id"]},
    {'files':["calendar.txt","trips.txt","calendar_dates.txt"],'ids':["service_id"]},
    {'files':["trips.txt","routes.txt"],'ids':["route_id"]},
    {'files':["agency.txt","routes.txt"],'ids':["agency_id"]},
    {'files':["trips.txt","shapes.txt"],'ids':["shape_id"]},
]
# ------------------------------
# DATE PARSER
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
        for f in files:
            if os.path.splitext(file)[0] == os.path.splitext(f)[0]:
                return os.path.join(root, f)
            
    return None

# ------------------------------
# TRY PARSE SINGLE LINE
# ------------------------------

def try_parse_line(line: str, config: Dict[str, Any], expected_cols: int|None = None, header:list|None=None, schema:dict|None=None, mandatory_columns:list=[]) -> Tuple[List[str]|None, str|None, str|None, bool]:
    line = line.strip()
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
        if len(header) != len(parsed):
            error += f"Len of parsed row {parsed} mismatch with len of header {header}. parsed: {parsed} header: {header}. "
            fix += "Excluded "
            return None, error, fix, True
        
        for col_idx, col_name in enumerate(header):
            if col_name not in schema:
                continue

            dtype = schema[col_name]

            val = parsed[col_idx]
            if val is not None:
                val = str(val).strip()

            original = val
            parsed_val = gtfs_checker.normalize_string(val,strict=False)

            # Skip empty values
            if val is None or val == '': 
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
                    parsed_val = gtfs_checker.parse_date(parsed_val)
                elif dtype == "time":
                    parsed_val = gtfs_checker.parse_time(parsed_val)
                elif dtype == "time|None":
                    if parsed_val is None:
                        parsed_val = None 
                    else:
                        parsed_val = gtfs_checker.parse_time(parsed_val)
                elif dtype == "int|bool":
                    if parsed_val in ("true", "1"):
                        parsed_val = 1
                    elif parsed_val in ("false", "0"):
                        parsed_val = 0
                    else:
                        parsed_val = int(float(parsed_val))
                elif dtype == int:
                    parsed_val = int(float(parsed_val))
                elif dtype == float:
                    parsed_val = float(parsed_val)
                elif dtype == "route_type":
                    parsed_val = gtfs_checker.normalize_route_type(parsed_val)
                elif dtype == "exception_type":
                    if parsed_val == "added":
                        parsed_val = 1 
                    elif parsed_val == "removed":
                        parsed_val = 2 
        
                    parsed_val = int(float(parsed_val))
                    if parsed_val != 1 and parsed_val != 2:
                        parsed_val = None 
                        original = None
                        error += f"{parsed_val} not valid for {col_name}. Only [1,2] are valid. "
                        fix += f"Set {col_name} to None. "
                else:
                    parsed_val = parsed_val
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
        header_line = lines[0].strip()
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
            new_col = gtfs_checker.normalize_string(i)
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
    if colum_names is not None:
        parsed_cols_df = lines_df.select(['line_number','parsed','excluded'])
        for i, col_name in enumerate(colum_names):
            parsed_cols_df = parsed_cols_df.with_columns(
                pl.Series(col_name, lines_df["parsed"].list.get(i).cast(pl.Utf8))
            )
        
        parsed_cols_df = parsed_cols_df.drop('parsed')

    else:
        parsed_cols_df = lines_df.select(['line_number','parsed','excluded'])

    parsed_cols_df = parsed_cols_df.filter(~pl.col('excluded')).drop('excluded')
    errors_df = lines_df.select(['line_number','error','fix','excluded']).drop_nulls("error")

    if len(errors_df.filter("excluded")) > 0:
        warnings.warn(f"{len(errors_df.filter("excluded"))} rows of file {path} have failed while parsing.")

    if header_error != "":
        errors_df = pl.concat([
            pl.DataFrame({'line_number':[0],'error':['Error parsing header'],'fix':[header_error],'excluded':[False]}),
            errors_df
        ])

    return parsed_cols_df, errors_df



def unzip(file,output_path=None, delete:bool=False, overwrite:bool=True):
    if os.path.isfile(file):
        # Extract the ZIP
        extraction_folder, ext = os.path.splitext(file)
        if output_path is not None:
            basename = os.path.basename(extraction_folder)
            extraction_folder = os.path.join(output_path,basename)

        if os.path.exists(extraction_folder):
            if overwrite:
                shutil.rmtree(extraction_folder)
            else:
                return extraction_folder
            
        os.makedirs(extraction_folder, exist_ok=True)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)

        if delete:
            shutil.rmtree(file)

        return extraction_folder
    else:
        raise Exception(f"Invalid zip file {file}")
    
    

def preprocess_gtfs(path,output_folder, mandatory_files = MANDATORY_FILES, file_preferences = PREFERENTIAL_FILES, file_pairs = FILE_PAIRS):
    log = ""
    delete_path = None
    if os.path.isfile(path):
        orig_path = copy.copy(path)
        path = unzip(path)
        log += f"Extracted file {orig_path} to {path} \n"
        delete_path = path
    elif not os.path.isdir(path):
        raise Exception(f"Path {path} does not exist")
    
    mandatory_file_list = []
    mandatory_file_groups = {}
    for file in mandatory_files:
        if isinstance(file,list):
            file_group = []
            for f in file:
                if search_file(path,f) is not None:
                    mandatory_file_list.append(f) 
                    file_group.append(f)

            if len(file_group) > 1:
                for f in file_group:
                    mandatory_file_groups[f] = file_group

            elif len(file_group) == 0:
                print(log)
                raise Exception(f"None of the files {file} not found in folder path {path}. At least one file should exist. This GTFS might be broken.")
        else:
            if search_file(path,file) is None:
                print(log)
                raise Exception(f"File {file} not found in folder path {path}. This GTFS might be broken.")
            else:
                mandatory_file_list.append(file)
    
    gtfs = {}
    gtfs_errors = {}
    for root, dirs, files in os.walk(path):
        for file_name in files:
            file_name, ext = os.path.splitext(file_name)
            if ext == ".txt" or ext == ".csv":  
                file_path = os.path.join(root, file_name + ext)
            else:
                log += f"Can't read file {os.path.join(root, file_name)}. Not a text file. \n"
                continue

            if (file_name + ".txt") not in IMPLEMENTED_PARSERS:
                basename = os.path.basename(path)
                # Ensure the output folder exists
                os.makedirs(os.path.join(output_folder, basename), exist_ok=True)
                shutil.copy2(file_path, os.path.join(output_folder,basename,file_name + '.txt'))
                log += f"File {file_name + ext} has no parser. Copying directly to the output folder. \n"
                continue
            
            try:
                content, errors = validate_and_load_csv(file_path,header=True)
            except Exception as e:
                log += f"Error reading file {file_path}. {e}. \n"
                if (file_name + ".txt") in mandatory_file_list:
                    print(log)
                    raise Exception(f"Error reading file {os.path.join(path,file_path)}. {e}. \n")
                
                continue

            if (len(content) == 0) and (file_name + ".txt" in mandatory_file_list):
                excluded_df = errors.filter(pl.col('excluded'))
                if len(excluded_df) > 0:
                    basename = os.path.basename(path)
                    # Ensure the output folder exists
                    os.makedirs(os.path.join(output_folder, basename), exist_ok=True)
                    error_path = os.path.join(output_folder, basename, f"{file_name}_errors.txt")
                    excluded_df.write_csv(
                        error_path,
                        separator=",",
                        quote_char='"',
                        decimal_comma=False,   # Use '.' for decimals
                        include_header=True
                    )
                    log += f"Created file {error_path} \n"
                    
                print(log)
                if ((file_name+".txt") in mandatory_file_groups.keys()) and (len(mandatory_file_groups[(file_name+".txt")]) > 1):
                    mandatory_file_groups.pop(file_name+".txt", None)
                    for k in mandatory_file_groups.keys():
                        if (file_name+".txt") in mandatory_file_groups[k]: 
                            mandatory_file_groups[k].remove(file_name+".txt")

                    if (file_name+".txt") in mandatory_file_list:
                        mandatory_file_list.remove(file_name+".txt")

                    continue
                elif (file_name+".txt") in mandatory_file_list:
                    raise Exception(f"File {os.path.join(path,file_path)} is empty")
                else:
                    mandatory_file_groups.pop(file_name+".txt", None)
                    for k in mandatory_file_groups.keys():
                        if (file_name+".txt") in mandatory_file_groups[k]: 
                            mandatory_file_groups[k].remove(file_name+".txt")

                    continue

                # warnings.warn(f"File {os.path.join(path,file_path)} is empty")
                # log += f"File {os.path.join(path,file_path)} is empty. \n"

            gtfs[file_name] = content 
            gtfs_errors[file_name] = errors 

    for file in mandatory_files:
        if isinstance(file,list):
            file_group = []
            for f in file:
                if f in mandatory_file_list:
                    file_group.append(f)

            if len(file_group) == 0:
                print(log)
                raise Exception(f"None of the files {file} not found in folder path {path}. At least one file should exist. This GTFS might be broken.")
        else:
            if file not in mandatory_file_list:
                print(log)
                raise Exception(f"File {file} not found in folder path {path}. This GTFS might be broken.")

    for i in range(len(file_pairs)):
        file_list_i = set(file_pairs[i]['files'])
        file_list_i = file_list_i.intersection(set(f + ".txt" for f in gtfs.keys()))
        id_cols = set(file_pairs[i]['ids'])
        preferential_files_i = (
            set(mandatory_file_list).union(file_preferences)
        ).intersection(file_list_i)
        mandatory_files_i = (
            set(mandatory_file_list)
        ).intersection(file_list_i)
        for id_col in id_cols:
            id_vals = None 
            file_without_col = None

            if len(mandatory_files_i) == 0:
                check_file_list = preferential_files_i 
            else:
                check_file_list = mandatory_files_i

            if len(check_file_list) == 0:
                check_file_list = file_list_i

            for file in (set(check_file_list) - set(mandatory_file_groups.keys())):
                if id_col in gtfs[file.removesuffix(".txt")].columns:
                    if len(file_list_i) > 1:
                        # Get the column values as a set of strings
                        current_id_vals = set(
                            map(
                                str, 
                                gtfs[file.removesuffix(".txt")].select(id_col)[id_col].to_list()
                            )
                        )
                        if id_vals is None:
                            # For the first file, just initialize ids
                            id_vals = current_id_vals
                        else:
                            # Intersect with the existing ids
                            id_vals = id_vals.intersection(current_id_vals)  
                else:
                    file_without_col = file 
                    break 

            if file_without_col is None:
                unique_groups = []
                for f in check_file_list:
                    if f in mandatory_file_groups.keys():
                        unique_groups.append(mandatory_file_groups[f])
                
                unique_groups = {tuple(v) for v in unique_groups}
                for file_group in unique_groups:
                    current_id_vals = None
                    has_col = False
                    for file in file_group: 
                        if id_col in gtfs[file.removesuffix(".txt")].columns:
                            has_col = True
                            if len(file_list_i) > 1:
                                # Get the column values as a set of strings
                                current_id_vals_i = set(
                                    map(
                                        str, 
                                        gtfs[
                                            file.removesuffix(".txt")
                                        ].select(id_col)[id_col].to_list()
                                    )
                                )
                                if current_id_vals is None:
                                    # For the first file, just initialize ids
                                    current_id_vals = current_id_vals_i
                                else:
                                    # Intersect with the existing ids
                                    current_id_vals = current_id_vals.union(current_id_vals_i)  

                        else:
                            file_without_col = file

                    if has_col:
                        file_without_col = None 
                        if id_vals is None:
                            # For the first file, just initialize ids
                            id_vals = current_id_vals
                        elif current_id_vals is not None:
                            # Intersect with the existing ids
                            id_vals = id_vals.intersection(current_id_vals) 
                    else:
                        break

            if file_without_col is None:
                if (id_vals is not None) and (len(file_list_i) > 1):
                    for file in file_list_i:
                        not_in_id = set(gtfs[file.removesuffix(".txt")].select(id_col)[id_col].to_list()) - id_vals
                        if len(not_in_id) > 0:
                            log += f"The id column {id_col} in {os.path.join(path,file)} has some ids not used elsewere. Excluding the following {id_col} values: \n" 
                            log += f"{not_in_id} \n" 
                            # warnings.warn(f"len{not_in_id} {check_id_col} ids in {os.path.join(path,check_id_file)} are not used elsewere and are being excluded.")
                            gtfs[file.removesuffix(".txt")] = gtfs[file.removesuffix(".txt")].filter(
                                pl.col(id_col).is_in(id_vals)
                            )
                            gtfs_errors[file.removesuffix(".txt")] = gtfs_errors[file.removesuffix(".txt")].join(
                                gtfs[file.removesuffix(".txt")],on="line_number",how="left"
                            )

            else:
                if len(mandatory_files_i) >= 2: 
                    raise Exception(f"Mandatory column {id_col} not in file {file_without_col}.")
                else:
                    if len(mandatory_files_i) == 0:
                        delete_files_i = [f for f in file_list_i if f not in preferential_files_i]
                    else:
                        delete_files_i = [f for f in file_list_i if f not in mandatory_files_i]

                    if len(delete_files_i) > 0:
                        log += f"Excluding files {[fi for fi in delete_files_i]} as file {file_without_col} does not contain necesary column {id_col}. \n"
                        for key in delete_files_i:
                            key = str(key).removesuffix(".txt")
                            gtfs.pop(key, None)         # remove key if it exists
                            gtfs_errors.pop(key, None)  # remove key if it exists

    basename = os.path.basename(path)
    # Ensure the output folder exists
    os.makedirs(os.path.join(output_folder, basename), exist_ok=True)


    for file_name, df in gtfs.items():
        base_path = os.path.join(output_folder, basename, f"{file_name}.txt")
        df.write_csv(
            base_path,
            separator=",",         # use comma as field separator
            quote_char='"',        # wrap strings with double quotes
            decimal_comma=False,   # False means use '.' as decimal separator
            include_header=True
        )
        log += f"Created file {base_path} \n"

    for file_name, df in gtfs_errors.items():
        if len(df) > 0:
            excluded_df = df.filter(pl.col('excluded'))
            if len(excluded_df) > 0:
                error_path = os.path.join(output_folder, basename, f"{file_name}_errors.txt")
                excluded_df.write_csv(
                    error_path,
                    separator=",",
                    quote_char='"',
                    decimal_comma=False,   # Use '.' for decimals
                    include_header=True
                )
                log += f"Created file {error_path} \n"

            file_log_path = os.path.join(output_folder, basename, f"{file_name}_logs.txt")
            df.write_csv(
                file_log_path,
                separator=",",
                quote_char='"',
                decimal_comma=False,   # Use '.' for decimals
                include_header=True
            )
            log += f"Created file {file_log_path} \n"

    if delete_path is not None:
        shutil.rmtree(delete_path)
        log += f"Deleted file {delete_path} \n"

    with open(os.path.join(output_folder, basename, "logs.txt"), "w", encoding="utf-8") as f:
        f.write(log)

    return os.path.join(output_folder, basename)
    



# ------------------------------
# LAZY CSV READING
# ------------------------------

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
                # Kept as a string here (not cast straight to int): GTFS also
                # permits named route_type values ("bus", "tram", "cable
                # car", ...), which `Routes.__read_routes` resolves via
                # `gtfs_checker.normalize_route_type` before casting to int
                # itself. Casting to int here would silently null out every
                # named value before that logic ever saw it.
                dtype = str
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
        path_list = [search_file(os.path.dirname(p), os.path.basename(p)) or p for p in path_list]

    file_lfs = [
        lf for i, p in enumerate(path_list)
        if (lf := read_csv_lazy(p, schema_overrides=schema_overrides, file_id=i+min_file_id, check_files=check_files, mandatory_cols=mandatory_cols, id_cols=id_cols)) is not None
    ]

    if not file_lfs:
        return None

    return pl.concat(file_lfs, how="diagonal_relaxed")


