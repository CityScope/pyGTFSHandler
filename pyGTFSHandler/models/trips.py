from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from .. import utils, gtfs_checker
import os
import warnings

class Trips:
    """
    A class to load and optionally filter GTFS trips data from one or more `trips.txt` files using Polars LazyFrame.

    Attributes:
        paths (List[Path]): List of directory paths containing `trips.txt` files.
        lf (pl.LazyFrame): A Polars LazyFrame containing the (optionally filtered) trips data.
    """

    def __init__(self,lf=None,trip_ids=None) -> None:
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

        if service_ids is not None:
            service_ids = [
                sid[:-6] if (sid is not None and sid.endswith("_night")) else sid
                for sid in service_ids
            ]

        self.lf = self.__read_trips(paths,service_ids, trip_ids, route_ids, check_files=check_files, min_file_id=min_file_id)
        if (service_ids is not None) or (route_ids is not None):
            self.trip_ids = (
                self.lf.select("trip_id").unique().collect()["trip_id"].to_list()
            )
        else:
            self.trip_ids = trip_ids

        if (self.trip_ids is not None) and (len(self.trip_ids) > 0) and (self.trip_ids[0] is None):
            self.trip_ids = []

    def __read_trips(
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
            new_p = gtfs_checker.search_file(p, file=file)
            if new_p is None:
                trip_paths.append(None)
                warnings.warn(f"File {file} does not exist in {p}", UserWarning)
            else:
                trip_paths.append(new_p)


        schema_dict, _ = gtfs_checker.get_df_schema_dict("trips.txt")
        trips = utils.read_csv_list(trip_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id)
        if (trips is None) or (trips.select(pl.len()).collect().item() == 0):
            raise Exception(f"No trips.txt file found for any {paths}")
        
        trips = utils.filter_by_id_column(trips, "service_id", service_ids)
        trips = utils.filter_by_id_column(trips, "trip_id", trip_ids)
        trips = utils.filter_by_id_column(trips, "route_id", route_ids)
        if "direction_id" not in trips.collect_schema().names():
            trips = trips.with_columns(pl.lit(None).alias("direction_id"))

        trips = trips.with_columns(pl.col("direction_id").cast(int).alias("direction_id"))

        return trips
