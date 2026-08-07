# -*- coding: utf-8 -*-
"""GTFS trips.txt handling: loading and `service_id`/`trip_id`/`route_id`
filtering.

Why this module exists and how it's organized:
-----------------------------------------------
Kept deliberately thin: `_read_trips` reads the file, applies the three id
filters via `geo_polars.filter_by_id_column`, and backfills three optional GTFS
columns this codebase relies on elsewhere with an all-null default when the
feed omits them -- `direction_id` (cast to a nullable int), and
`trip_headsign`/`shape_id` (used respectively by frequency-template
deduplication in `models/frequencies.py` and by the real-shape-geometry
lookup in `models/shapes.py`/`Feed.load_shapes`). Without these defaults,
either of those would raise a missing-column error the moment a feed simply
didn't bother to include an optional field.
"""

from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from ..utils import geo_polars
from ..utils import gtfs_checker
from ..utils import io
import os
import warnings

class Trips:
    """
    A class to load and optionally filter GTFS trips data from one or more `trips.txt` files using Polars LazyFrame.

    Attributes:
        paths (List[Path]): List of directory paths containing `trips.txt` files.
        lf (pl.LazyFrame): A Polars LazyFrame containing the (optionally filtered) trips data.
    """

    def __init__(self, lf: Optional[pl.LazyFrame] = None, trip_ids: Optional[List[str]] = None) -> None:
        """Wraps an already-loaded trips LazyFrame, or leaves it empty for `load` to fill in.

        Args:
            lf: Optional pre-loaded `trips.txt` LazyFrame.
            trip_ids: Optional pre-computed list of in-scope `trip_id`s.
        """
        self.lf = lf
        self.trip_ids = trip_ids

    def load(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        service_ids: Optional[List[str]] = None,
        trip_ids: Optional[List[str]] = None,
        route_ids: Optional[List[str] | pl.LazyFrame | pl.DataFrame] = None,
        check_files:bool=False,
        min_file_id=0
    ):
        """
        Initializes the Trips class by reading and filtering the trips data.

        Args:
            path (str | Path | list[str | Path]): One or more paths to directories containing `trips.txt` files.
            service_ids (list[str], optional): List of service IDs to filter by.
            trip_ids (list[str], optional): List of trip IDs to filter by.
            route_ids (list[str], optional): List of route IDs to filter by.
        """
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        self.lf = self._read_trips(paths,service_ids, trip_ids, route_ids, check_files=check_files, min_file_id=min_file_id)
        if (service_ids is not None) or (route_ids is not None):
            self.trip_ids = (
                self.lf.select("trip_id").unique().collect()["trip_id"].to_list()
            )
        else:
            self.trip_ids = trip_ids

        if (self.trip_ids is not None) and (len(self.trip_ids) > 0) and (self.trip_ids[0] is None):
            self.trip_ids = []

    def _read_trips(
        self,
        paths,
        service_ids: Optional[List[str]],
        trip_ids: Optional[List[str]],
        route_ids: Optional[List[str]],
        check_files=False,
        min_file_id=0
    ) -> pl.LazyFrame:
        """
        Reads the trips data from one or more `trips.txt` files and applies optional filters.

        Args:
            service_ids (list[str], optional): List of service IDs to filter by.
            trip_ids (list[str], optional): List of trip IDs to filter by.
            route_ids (list[str], optional): List of route IDs to filter by.

        Returns:
            pl.LazyFrame: Filtered trips data as a LazyFrame including duplicated night trips.
        """
        trip_paths: List[Path] = []
        file = "trips.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                trip_paths.append(None)
                warnings.warn(f"File {file} does not exist in {p}", UserWarning)
            else:
                trip_paths.append(new_p)


        schema_dict, _ = gtfs_checker.get_df_schema_dict("trips.txt")
        trips = io.read_csv_list(trip_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id)
        if (trips is None) or (trips.select(pl.len()).collect().item() == 0):
            raise Exception(f"No trips.txt file found for any {paths}")
        
        trips = geo_polars.filter_by_id_column(trips, "service_id", service_ids)
        trips = geo_polars.filter_by_id_column(trips, "trip_id", trip_ids)
        trips = geo_polars.filter_by_id_column(trips, "route_id", route_ids)
        existing_columns = trips.collect_schema().names()
        if "direction_id" not in existing_columns:
            trips = trips.with_columns(pl.lit(None).alias("direction_id"))

        trips = trips.with_columns(pl.col("direction_id").cast(int).alias("direction_id"))

        # `trip_headsign` and `shape_id` are optional per the GTFS spec, but
        # downstream code (frequency-template deduplication, shape lookups)
        # expects them to always be present, even as an all-null column.
        for optional_column in ("trip_headsign", "shape_id"):
            if optional_column not in existing_columns:
                trips = trips.with_columns(pl.lit(None, dtype=pl.Utf8).alias(optional_column))

        return trips
