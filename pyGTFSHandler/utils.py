import polars as pl
from pathlib import Path
import unicodedata
import re


def load_lazyframe_from_file(dir_path: Path, filename: str) -> pl.LazyFrame:
    """
    Load a LazyFrame from a CSV file on disk.
    """
    file_path = dir_path / filename
    if not file_path.exists():
        raise FileNotFoundError(f"{filename} not found in {dir_path}")
    return pl.scan_csv(str(file_path))


def _clean_string(string):
    # Replace special characters and multiple spaces/underscores with a single underscore
    string = _normalize_string(string)
    string = re.sub(
        r"[^a-zA-Z0-9]", "_", string
    )  # Replace non-alphanumeric characters with underscores
    string = re.sub(
        r"_+", "_", string
    )  # Replace multiple underscores with a single one
    string = string.strip("_")  # Remove leading/trailing underscores

    return string


def read_csv_lazy(path, schema_overrides: dict = None, columns: list = None):
    # Use scan_csv for lazy loading
    lf = pl.scan_csv(path, has_header=True, separator=",")

    # Get column names without triggering expensive computation
    column_names = lf.collect_schema().names()
    rename_dict = {name: _clean_string(name) for name in column_names}
    lf = lf.rename(rename_dict)

    # Get schema to check column types
    schema = lf.collect_schema()

    # Apply string operations only to string-type columns
    for col_name in column_names:
        dtype = schema.get(col_name)
        if dtype == pl.Utf8:  # Check if the column is of string type
            lf = lf.with_columns(
                pl.col(col_name)
                .str.strip_chars()
                .str.replace_all(r"[áàãâä]", "a")
                .str.replace_all(r"[éèêë]", "e")
                .str.replace_all(r"[íìîï]", "i")
                .str.replace_all(r"[óòõôö]", "o")
                .str.replace_all(r"[úùûü]", "u")
                .str.replace_all(r"[ñ]", "n")
                .str.replace_all(r"[`’]", "")
            )

    # Apply schema overrides if provided
    if schema_overrides:
        for col, dtype in schema_overrides.items():
            if col in column_names:
                lf = lf.with_columns(pl.col(col).cast(dtype))

    return lf


def _normalize_string(s: str) -> str:
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
    normalized_column_names = [_normalize_string(col) for col in column_names]
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
