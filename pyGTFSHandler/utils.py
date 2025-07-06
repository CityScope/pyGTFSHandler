import polars as pl
import unicodedata
import re


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
