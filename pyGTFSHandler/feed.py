from .models import StopTimes, Stops, Trips, Calendar

from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import geopandas as gpd
import polars as pl


class Feed:
    """
    Represents a GTFS feed composed of various GTFS components (calendar, trips, stops, etc.).

    Attributes:
        gtfs_dir (list[Path]): List of directories containing GTFS data.
        calendar (Calendar): Calendar object handling service date ranges.
        trips (Trips): Trips object containing trip information.
        stops (Stops): Stops object filtered optionally by an AOI.
        stop_times (StopTimes): StopTimes object linking trips and stops.
    """

    def __init__(
        self,
        gtfs_dirs: Union[list[Union[str, Path]], str, Path],
        aoi: Union[gpd.GeoDataFrame, gpd.GeoSeries, None] = None,
        service_ids: Optional[list[str]] = None,
        trip_ids: Optional[list[str]] = None,
        stop_ids: Optional[list[str]] = None,
        route_ids: Optional[list[str]] = None,
    ):
        """
        Initializes a Feed instance by loading and validating GTFS directories
        and constructing calendar, trip, stop, and stop_time objects.

        Args:
            gtfs_dirs (list[str | Path] | str | Path): One or more GTFS directories.
            aoi (GeoDataFrame | GeoSeries, optional): Area of interest to filter stops.
            service_ids (list[str], optional): List of service IDs to filter.
            trip_ids (list[str], optional): List of trip IDs to filter.
            stop_ids (list[str], optional): List of stop IDs to filter.
            route_ids (list[str], optional): List of route IDs to filter.
        """
        if not isinstance(gtfs_dirs, list):
            gtfs_dirs = [gtfs_dirs]

        self.gtfs_dir = [Path(p) for p in gtfs_dirs]

        for p in self.gtfs_dir:
            if not p.is_dir():
                raise ValueError(f"{p} is not a valid directory.")

        self.calendar = Calendar(self.gtfs_dir, service_ids=service_ids)
        self.trips = Trips(
            self.gtfs_dir,
            service_ids=self.calendar.service_ids,
            trip_ids=trip_ids,
            route_ids=route_ids,
        )
        self.stops = Stops(self.gtfs_dir, aoi=aoi, stop_ids=stop_ids)
        self.stop_times = StopTimes(
            self.gtfs_dir, stop_ids=self.stops.stop_ids, trip_ids=self.trips.trip_ids
        )

    def get_service_intensity_in_date_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Calculates the number of scheduled stop times per date within a given date range.

        Args:
            start_date (datetime, optional): Start of date range to filter. If None, uses earliest date available.
            end_date (datetime, optional): End of date range to filter. If None, uses latest date available.

        Returns:
            pl.DataFrame: DataFrame with columns ['date', 'service_intensity', 'weekday'] representing service intensity.
        """
        date_df = self.calendar.get_services_in_date_range(start_date, end_date)

        stop_times = self.stop_times.lf.select("trip_id")
        trips = self.trips.lf.select(["trip_id", "service_id"])

        if self.stop_times.frequencies is not None:
            frequencies = self.stop_times.frequencies

            # Join frequencies with service_id
            frequencies = frequencies.join(trips, on="trip_id", how="left")

            # Compute number of trips per frequency block
            frequencies = frequencies.with_columns(
                (
                    (pl.col("end_time_secs") - pl.col("start_time_secs"))
                    / pl.col("headway_secs")
                )
                .floor()
                .cast(pl.UInt32)
                .alias("n_trips")
            )

            # Sum n_trips per trip_id (if multiple frequency blocks per trip)
            trip_n_trips = frequencies.group_by("trip_id").agg(
                pl.col("service_id").first(), pl.col("n_trips").sum()
            )

            stop_times = stop_times.join(trip_n_trips, on="trip_id", how="left")
        else:
            stop_times = stop_times.join(trips, on="trip_id", how="left")
            stop_times = stop_times.with_columns(pl.lit(1).alias("n_trips"))

        # Compute stop time counts per service
        stop_time_counts_df = (
            stop_times.group_by("service_id")
            .agg(pl.col("n_trips").sum().alias("num_stop_times"))
            .collect()
        )

        # Explode date-service pairs
        exploded = date_df.explode("service_ids").rename({"service_ids": "service_id"})

        # Join with stop time counts
        joined = exploded.join(stop_time_counts_df, on="service_id", how="left")

        # Fill missing counts with zero
        joined = joined.with_columns(pl.col("num_stop_times").fill_null(0))

        # Group by date to compute total service intensity
        total_by_date = joined.group_by("date").agg(
            pl.col("weekday").first(),
            pl.col("num_stop_times").sum().alias("service_intensity"),
        )

        return total_by_date.sort("date")
