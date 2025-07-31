from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from utils import read_csv_list, get_df_schema_dict


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
        route_ids: Optional[List[str]] = None,
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

        if service_ids:
            service_ids = [
                sid[:-6] if sid.endswith("_night") else sid for sid in service_ids
            ]

        self.lf = self.__read_trips(service_ids, trip_ids, route_ids)
        if service_ids or route_ids:
            self.trip_ids = (
                self.lf.select("trip_id").unique().collect()["trip_id"].to_list()
            )
        else:
            self.trip_ids = trip_ids

    def __read_trips(
        self,
        service_ids: Optional[List[str]],
        trip_ids: Optional[List[str]],
        route_ids: Optional[List[str]],
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
        trip_paths = [p / "trips.txt" for p in self.paths if (p / "trips.txt").exists()]
        if not trip_paths:
            raise FileNotFoundError("No trips.txt files found in the provided path(s).")

        schema_dict = get_df_schema_dict(trip_paths[0])
        trips = read_csv_list(trip_paths, schema_overrides=schema_dict)

        if service_ids:
            service_ids_df = pl.DataFrame({"service_id": service_ids})
            trips = trips.join(service_ids_df.lazy(), on="service_id", how="semi")

        if trip_ids:
            trip_ids_df = pl.DataFrame({"trip_id": trip_ids})
            trips = trips.join(trip_ids_df.lazy(), on="trip_id", how="semi")

        if route_ids:
            route_ids_df = pl.DataFrame({"route_id": route_ids})
            trips = trips.join(route_ids_df.lazy(), on="route_id", how="semi")

        return trips
