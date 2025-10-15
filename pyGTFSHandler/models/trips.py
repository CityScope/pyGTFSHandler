from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from .. import utils
import os


class Trips:
    """
    A class to load and optionally filter GTFS trips data from one or more `trips.txt` files using Polars LazyFrame.

    Attributes:
        paths (List[Path]): List of directory paths containing `trips.txt` files.
        lf (pl.LazyFrame): A Polars LazyFrame containing the (optionally filtered) trips data.
    """

    def __init__(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        service_ids: Optional[List[str]] = None,
        trip_ids: Optional[List[str]] = None,
        route_ids: Optional[List[str] | pl.LazyFrame | pl.DataFrame] = None,
        check_files:bool=False
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
            self.paths = [Path(path)]
        else:
            self.paths = [Path(p) for p in path]

        if service_ids is not None:
            service_ids = [
                sid[:-6] if (sid is not None and sid.endswith("_night")) else sid
                for sid in service_ids
            ]

        self.lf = self.__read_trips(service_ids, trip_ids, route_ids, check_files=check_files)
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
        service_ids: Optional[List[str]],
        trip_ids: Optional[List[str]],
        route_ids: Optional[List[str]],
        check_files=False
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
        trip_paths = [p / "trips.txt" for p in self.paths]
        for p in trip_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File {p} does not exist")

        schema_dict = utils.get_df_schema_dict(trip_paths[0])
        trips = utils.read_csv_list(trip_paths, schema_overrides=schema_dict, check_files=check_files)

        trips = utils.filter_by_id_column(trips, "service_id", service_ids)
        trips = utils.filter_by_id_column(trips, "trip_id", trip_ids)
        trips = utils.filter_by_id_column(trips, "route_id", route_ids)

        return trips
