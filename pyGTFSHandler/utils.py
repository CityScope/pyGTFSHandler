import polars as pl
from pathlib import Path


def load_lazyframe_from_file(dir_path: Path, filename: str) -> pl.LazyFrame:
    """
    Load a LazyFrame from a CSV file on disk.
    """
    file_path = dir_path / filename
    if not file_path.exists():
        raise FileNotFoundError(f"{filename} not found in {dir_path}")
    return pl.scan_csv(str(file_path))
