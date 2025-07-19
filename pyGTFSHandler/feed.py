from .models import StopTimes, Stops, Trips, Calendar, Routes

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
        self.routes = Routes(self.gtfs_dir, route_ids=route_ids)
        self.trips = Trips(
            self.gtfs_dir,
            service_ids=self.calendar.service_ids,
            trip_ids=trip_ids,
            route_ids=self.routes.route_ids,
        )
        self.stops = Stops(self.gtfs_dir, aoi=aoi, stop_ids=stop_ids)
        self.stop_times = StopTimes(
            self.gtfs_dir, stop_ids=self.stops.stop_ids, trip_ids=self.trips.trip_ids
        )

        # Select relevant columns from stop_times
        self.lf = self.stop_times.lf.select(
            [
                "trip_id",
                "stop_id",
                "departure_time",
                "arrival_time",
                "stop_sequence",
                "shape_dist_traveled",
                "shape_time_traveled",
                "shape_total_travel_time",
                "next_day",
            ]
        )

        if self.stop_times.frequencies is not None:
            # Join frequencies
            self.lf = (
                self.lf.join(
                    self.stop_times.frequencies.select(
                        [
                            "trip_id",
                            "start_time",
                            "end_time",
                            "headway_secs",
                            "next_day",
                            "n_trips",
                        ]
                    ),
                    on="trip_id",
                    how="left",
                )
                .with_columns(
                    [
                        # Combine both `next_day` columns into one
                        (
                            pl.col("next_day")
                            | pl.col("next_day_right").fill_null(False)
                        ).alias("next_day"),
                        # Set n_trips = 1 where it's null (assuming n_trips may already exist)
                        pl.when(pl.col("n_trips").is_null())
                        .then(pl.lit(1))
                        .otherwise(pl.col("n_trips"))
                        .alias("n_trips"),
                    ]
                )
                .drop(["next_day_right"])
            )

        else:
            # No frequencies: create n_trips = 1
            self.lf = self.lf.with_columns(pl.lit(1).alias("n_trips"))

        # Merge with trips
        self.lf = self.lf.join(
            self.trips.lf.select(["trip_id", "service_id", "route_id", "shape_id"]),
            on="trip_id",
            how="left",
        )

        # Merge with stops
        self.lf = self.lf.join(
            self.stops.lf.select(["stop_id", "parent_station"]),
            on="stop_id",
            how="left",
        )

        # Merge with routes
        self.lf = self.lf.join(
            self.routes.lf.select(["route_id", "route_type"]), on="route_id", how="left"
        )

        # Apply next_day transformation to service_id
        self.lf = self.lf.with_columns(
            [
                pl.when(pl.col("next_day"))
                .then(pl.col("service_id") + "_night")
                .otherwise(pl.col("service_id"))
                .alias("service_id")
            ]
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

        gtfs_lf = self.lf.select("trip_id", "service_id", "n_trips")

        # Compute stop time counts per service
        stop_time_counts_df = (
            gtfs_lf.group_by("service_id")
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
