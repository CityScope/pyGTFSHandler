# -*- coding: utf-8 -*-
"""
GTFS Feed Orchestration Module

This module provides the `Feed` class, which serves as the main entry point for
loading, integrating, and analyzing a complete General Transit Feed Specification (GTFS) dataset.
It orchestrates the loading and processing of individual GTFS files (`stops.txt`,
`trips.txt`, `routes.txt`, `calendar.txt`, `stop_times.txt`, `shapes.txt`, etc.)
by using dedicated model classes for each component.

What this file does:
-------------------
The `Feed` class acts as a high-level container that:
1.  **Loads Data**: Initializes and loads data from multiple GTFS files into
    specialized objects (`Stops`, `Trips`, `Routes`, `Calendar`, `StopTimes`, `Shapes`).
2.  **Applies Filters**: Allows for initial filtering of the entire feed based on
    various criteria such as an Area of Interest (AOI), date ranges, time ranges, specific `service_ids`,
    `trip_ids`, `stop_ids`, or `route_ids`.
3.  **Integrates Components**: Joins the data from all individual components into a
    single, unified Polars LazyFrame (`self.lf`). This master frame contains a
    denormalized view of the schedule, linking stops, times, trips, routes,
    and services.
4.  **Handles Data Inconsistencies**:
    -   It ensures that trips generated from `frequencies.txt` (especially those
        crossing midnight) are correctly added to the main trips table.
    -   It performs advanced, shape-based interpolation for stop times that
        could not be fixed with simple methods, using distance-traveled data to
        estimate arrival/departure times accurately.
    -   For all trips and stops that happen after midnight the service_id will have "_night"
        added starting in the first stop after midnight.
5.  **Provides Analysis Methods**: Offers high-level methods for common GTFS
    analyses, such as:
    -   Calculating service intensity (number of trips x stops) over a date range.
    -   Calculating the mean interval (headway) between services at stops.
    -   Filtering the integrated data by specific date or time ranges.

The final result is a powerful `Feed` object that holds a clean, integrated, and
analysis-ready representation of the entire GTFS schedule.

TODOs:
- TODO avoid performing non necesary checks at the begining. Do first all the filters and then the checks.
- TODO check that all trips have a route and if not generate it.
- TODO check that trips from stop_times have a trip in trips and same for stops
"""

from models import StopTimes, Stops, Trips, Calendar, Routes, Shapes
import utils

from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List
import geopandas as gpd
import polars as pl


class Feed:
    """
    Represents and orchestrates a complete GTFS feed.

    This class integrates various GTFS components (calendar, trips, stops, etc.)
    into a single, queryable data structure. It handles the loading, filtering,
    and joining of all relevant GTFS files.

    Attributes:
        gtfs_dir (List[Path]): List of Path objects for the directories containing GTFS data.
        calendar (Calendar): An object handling `calendar.txt` and `calendar_dates.txt`.
        routes (Routes): An object handling `routes.txt`.
        trips (Trips): An object handling `trips.txt`, filtered by calendar and routes.
        stops (Stops): An object handling `stops.txt`, optionally filtered by an Area of Interest.
        stop_times (StopTimes): An object handling `stop_times.txt`, linking trips and stops.
        trip_shape_ids_lf (pl.LazyFrame): A LazyFrame mapping generated shape IDs to trip IDs.
        shapes (Shapes): An object handling `shapes.txt` data.
        lf (pl.LazyFrame): The main, integrated LazyFrame containing denormalized schedule data.
    """

    def __init__(
        self,
        gtfs_dirs: Union[List[Union[str, Path]], str, Path],
        aoi: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
        service_ids: Optional[List[str]] = None,
        trip_ids: Optional[List[str]] = None,
        stop_ids: Optional[List[str]] = None,
        route_ids: Optional[List[str]] = None,
    ):
        """
        Initializes a Feed instance by loading, filtering, and integrating GTFS data.

        The constructor orchestrates the entire data loading pipeline, from reading
        individual files to joining them into a final, unified LazyFrame.

        Args:
            gtfs_dirs (Union[List[Union[str, Path]], str, Path]):
                One or more paths to directories containing GTFS files.
            aoi (Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]]):
                An Area of Interest (GeoDataFrame or GeoSeries) used to filter stops
                geospatially. Only stops within this AOI will be included. Defaults to None.
            service_ids (Optional[List[str]]):
                A list of `service_id`s to filter the data. Only data related to these
                services will be loaded. Defaults to None.
            trip_ids (Optional[List[str]]):
                A list of `trip_id`s to filter the data. Defaults to None.
            stop_ids (Optional[List[str]]):
                A list of `stop_id`s to filter the data. Defaults to None.
            route_ids (Optional[List[str]]):
                A list of `route_id`s to filter the data. Defaults to None.

        Raises:
            ValueError: If any of the provided `gtfs_dirs` is not a valid directory.
        """
        # --- 1. Initialization and Validation ---
        if not isinstance(gtfs_dirs, list):
            gtfs_dirs = [gtfs_dirs]

        self.gtfs_dir: List[Path] = [Path(p) for p in gtfs_dirs]

        for p in self.gtfs_dir:
            if not p.is_dir():
                raise ValueError(f"{p} is not a valid directory.")

        # --- 2. Load Individual GTFS Components with Filtering ---
        # The loading is done in a logical order to allow for cascading filters.
        # e.g., Calendar is loaded first, and its service_ids are used to filter Trips.
        self.calendar: Calendar = Calendar(self.gtfs_dir, service_ids=service_ids)
        self.routes: Routes = Routes(self.gtfs_dir, route_ids=route_ids)
        self.trips: Trips = Trips(
            self.gtfs_dir,
            service_ids=self.calendar.service_ids,
            trip_ids=trip_ids,
            route_ids=self.routes.route_ids,
        )
        self.stops: Stops = Stops(self.gtfs_dir, aoi=aoi, stop_ids=stop_ids)

        self.stop_times: StopTimes = StopTimes(
            self.gtfs_dir, stop_ids=self.stops.stop_ids, trip_ids=self.trips.trip_ids
        )

        # --- 3. Integrate Generated Trips from Frequencies ---
        # If StopTimes generated new trips from frequencies.txt, we need to add them
        # to the main trips table.
        if "orig_trip_id" in self.stop_times.lf.collect_schema().names():
            # Get the mapping of new trip_ids to original template trip_ids.
            unique_trip_ids: pl.LazyFrame = self.stop_times.lf.select(
                ["trip_id", "orig_trip_id"]
            ).unique()

            # Join the trips table with this mapping. This duplicates the original
            # trip's data (route_id, service_id, etc.) for each new generated trip_id.
            self.trips.lf = (
                self.trips.lf.join(
                    unique_trip_ids,
                    left_on="trip_id",
                    right_on="orig_trip_id",
                    how="right",  # Right join ensures all new trip_ids from stop_times are included.
                )
                .with_columns(
                    # The 'trip_id' from the original trips table becomes the new `trip_id` (e.g., trip_A_123)
                    # while the original `trip_id` column is now from the right side (`orig_trip_id`).
                    # We rename the right-side `trip_id` column to be the definitive `trip_id`.
                    pl.col("trip_id_right").alias("trip_id")
                )
                .drop("trip_id_right")
            )

        # --- 4. Load Shapes and Perform Advanced Time Interpolation ---
        self.trip_shape_ids_lf: pl.LazyFrame = self.stop_times.generate_shape_ids()
        self.shapes: Shapes = Shapes(
            self.gtfs_dir, self.trip_shape_ids_lf, self.stops.lf
        )

        # --- 5. Build the Main Integrated LazyFrame (`lf`) ---
        # Start with the core stop_times data.
        self.lf: pl.LazyFrame = self.stop_times.lf.select(
            [
                "trip_id",
                "stop_id",
                "departure_time",
                "arrival_time",
                "stop_sequence",
                "shape_time_traveled",
                "shape_total_travel_time",
                "next_day",
                "fixed_time",  # Keep this flag for advanced interpolation
            ]
        )

        # Join with frequency data if it exists.
        if self.stop_times.frequencies is not None:
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
                        # Combine the `next_day` column from stop_times and frequencies.
                        pl.col("next_day_right")
                        .fill_null(pl.col("next_day"))
                        .alias("next_day"),
                        # Trips not from frequencies have 1 trip.
                        pl.col("n_trips").fill_null(1),
                    ]
                )
                .drop(["next_day_right"])
            )
        else:
            # If no frequencies file, all trips are individual trips.
            self.lf = self.lf.with_columns(pl.lit(1, dtype=pl.UInt32).alias("n_trips"))

        # Merge with trips, stops, routes, and shapes data to create the full view.
        self.lf = self.lf.join(
            self.trips.lf.select(["trip_id", "service_id", "route_id", "shape_id"]),
            on="trip_id",
            how="left",
        )
        self.lf = self.lf.join(
            self.stops.lf.select(["stop_id", "parent_station"]),
            on="stop_id",
            how="left",
        )
        self.lf = self.lf.join(
            self.routes.lf.select(["route_id", "route_type"]), on="route_id", how="left"
        )
        self.lf = self.lf.join(
            self.shapes.lf.filter(pl.col("stop_sequence").is_not_null()).select(
                [
                    "shape_id",
                    "stop_sequence",
                    "shape_dist_traveled",
                    "shape_total_distance",
                ]
            ),
            on=["shape_id", "stop_sequence"],
            how="left",
        )

        # --- 6. Perform Final Data Cleaning and Transformation ---
        # If any times were fixed with the simple method, run the advanced,
        # shape-based interpolation now that shape_dist_traveled is available.
        if self.stop_times.fixed_times:
            self.lf = self.__fix_null_times(self.lf)

        # For services running past midnight, create a unique "night" service_id
        # to distinguish them from the same service on the previous day.
        self.lf = self.lf.with_columns(
            [
                pl.when(pl.col("next_day"))
                .then(pl.col("service_id") + "_night")
                .otherwise(pl.col("service_id"))
                .alias("service_id")
            ]
        )

    def __fix_null_times(self, stop_times: pl.LazyFrame) -> pl.LazyFrame:
        """
        Performs advanced, shape-based interpolation for missing stop times.

        This method uses linear interpolation based on `shape_dist_traveled` to
        estimate missing `departure_time` values for stops between two stops
        with known times. It correctly handles trips that cross midnight.

        Args:
            stop_times (pl.LazyFrame): The LazyFrame of stop times, which must
                                       include `shape_dist_traveled`.

        Returns:
            pl.LazyFrame: A LazyFrame with null times interpolated.
        """
        stop_times = stop_times.sort("trip_id", "stop_sequence")

        # Temporarily nullify times that were fixed with the simple forward-fill,
        # so they can be re-interpolated more accurately.
        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("fixed_time"))
                .then(None)
                .otherwise(pl.col("departure_time"))
                .alias("departure_time")
            ]
        )

        # Create context columns: the next and previous known time/distance points.
        stop_times = stop_times.with_columns(
            [
                pl.col("departure_time")
                .forward_fill()
                .over("trip_id")
                .alias("dep_time_fwd"),
                pl.col("shape_dist_traveled")
                .forward_fill()
                .over("trip_id")
                .alias("dist_fwd"),
                pl.col("departure_time")
                .backward_fill()
                .over("trip_id")
                .alias("dep_time_bwd"),
                pl.col("shape_dist_traveled")
                .backward_fill()
                .over("trip_id")
                .alias("dist_bwd"),
            ]
        )

        # Apply linear interpolation for rows where departure_time is null.
        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("departure_time").is_null())
                .then(
                    # Handle midnight crossing case (backward time is smaller than forward time)
                    pl.when(pl.col("dep_time_bwd") < pl.col("dep_time_fwd"))
                    .then(
                        pl.col("dep_time_fwd")
                        + (
                            (pl.col("shape_dist_traveled") - pl.col("dist_fwd"))
                            / (pl.col("dist_bwd") - pl.col("dist_fwd"))
                        )
                        * (
                            (pl.col("dep_time_bwd") + 86400) - pl.col("dep_time_fwd")
                        )  # Add 24h to backward time
                    )
                    .otherwise(
                        pl.col("dep_time_fwd")
                        + (
                            (pl.col("shape_dist_traveled") - pl.col("dist_fwd"))
                            / (pl.col("dist_bwd") - pl.col("dist_fwd"))
                        )
                        * (pl.col("dep_time_bwd") - pl.col("dep_time_fwd"))
                    )
                )
                .otherwise(pl.col("departure_time"))
                .alias(
                    "departure_time"
                )  # Overwrite departure_time with interpolated value
            ]
        )

        # Clean up temporary helper columns.
        stop_times = stop_times.drop(
            ["dep_time_fwd", "dep_time_bwd", "dist_fwd", "dist_bwd"]
        )

        # Recalculate travel times now that nulls are filled.
        stop_times = self.stop_times._add_shape_time_and_midnight_crossing(stop_times)

        return stop_times

    def filter_by_date_range(
        self, data: pl.LazyFrame, start_date: datetime = None, end_date: datetime = None
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a date range.

        It uses the `Calendar` object to find all `service_id`s active within
        the specified date range and then inner-joins the input data with these
        service IDs.

        Args:
            data (pl.LazyFrame): The LazyFrame to be filtered. Must contain a `service_id` column.
            start_date (datetime): The start of the date range (inclusive).
            end_date (datetime): The end of the date range (inclusive).

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """
        date_df: pl.DataFrame = self.calendar.get_services_in_date_range(
            start_date, end_date
        )
        date_df = date_df.explode("service_ids").rename({"service_ids": "service_id"})
        data = data.join(
            date_df.select("service_id").lazy(), on="service_id", how="inner"
        )
        return data

    def filter_by_time_range(
        self,
        data: pl.LazyFrame,
        start_time: datetime = datetime.min,
        end_time: datetime = datetime.max,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a time-of-day range.

        It handles trips defined by `frequencies.txt` differently from those with
        explicit schedules.

        Args:
            data (pl.LazyFrame): The LazyFrame to filter. Must contain time-related columns.
            start_time (datetime): The start of the time range. Defaults to 00:00:00.
            end_time (datetime): The end of the time range. Defaults to 23:59:59.

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """
        start_time: int = utils.time_to_seconds(start_time.time())
        end_time: int = utils.time_to_seconds(end_time.time())

        # If frequencies are present, filter based on both frequency windows and explicit times.
        if self.stop_times.frequencies is not None:
            # Keep rows if their frequency window overlaps the filter time.
            data = data.filter(
                pl.col("start_time").is_null()
                | (
                    (pl.col("start_time") >= start_time)
                    & (pl.col("end_time") <= end_time)
                )
            )
            # For non-frequency trips, filter by explicit departure/arrival times.
            data = data.filter(
                (pl.col("start_time").is_not_null())
                | (
                    (pl.col("departure_time") >= start_time)
                    & (pl.col("arrival_time") <= end_time)
                )
            )
        else:
            # If no frequencies, just filter by explicit departure/arrival times.
            data = data.filter(
                (pl.col("departure_time") >= start_time)
                & (pl.col("arrival_time") <= end_time)
            )

        return data

    def get_service_intensity_in_date_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Calculates the number of scheduled stop times per date within a given date range.

        This provides a measure of how much service is running each day.

        Args:
            start_date (Optional[datetime]): Start of the date range to analyze.
                                              If None, uses the earliest date in the feed.
            end_date (Optional[datetime]): End of the date range to analyze.
                                            If None, uses the latest date in the feed.

        Returns:
            pl.DataFrame: A DataFrame with columns ['date', 'weekday', 'service_intensity'], where
                          `service_intensity` is the total count of stop-time events.
        """
        # Get all active services for each day in the date range.
        date_df: pl.DataFrame = self.calendar.get_services_in_date_range(
            start_date, end_date
        )
        gtfs_lf: pl.LazyFrame = self.lf.select("trip_id", "service_id", "n_trips")

        # Count the number of trips associated with each service_id.
        stop_time_counts_df: pl.DataFrame = (
            gtfs_lf.group_by("service_id")
            .agg(pl.col("n_trips").sum().alias("num_stop_times"))
            .collect()
        )

        # Explode the date_df so each row is a (date, service_id) pair.
        exploded: pl.DataFrame = date_df.explode("service_ids").rename(
            {"service_ids": "service_id"}
        )

        # Join the daily services with their trip counts.
        joined: pl.DataFrame = exploded.join(
            stop_time_counts_df, on="service_id", how="left"
        )
        joined = joined.with_columns(pl.col("num_stop_times").fill_null(0))

        # Group by date to sum up the trip counts for total daily intensity.
        total_by_date: pl.DataFrame = joined.group_by("date").agg(
            pl.col("weekday").first(),
            pl.col("num_stop_times").sum().alias("service_intensity"),
        )

        return total_by_date.sort("date")

    def get_mean_intervall_at_stops(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        start_time: datetime = datetime.min,
        end_time: datetime = datetime.max,
        method: str = "route",
    ) -> pl.LazyFrame:
        """
        Calculates the mean interval (headway) between services at stops.

        This method uses the harmonic mean, which is appropriate for averaging
        frequencies or rates. It can calculate intervals in different ways based on
        the `method` parameter.

        Args:
            start_date (Optional[datetime]): The start date for filtering data.
            end_date (Optional[datetime]): The end date for filtering data.
            start_time (datetime): The start time of day for filtering. Defaults to 00:00:00.
            end_time (datetime): The end time of day for filtering. Defaults to 23:59:59.
            method (str): The aggregation method. One of ['route', 'max', 'agg'].
                          - 'route': Calculates mean interval per route and direction at each stop. (Default)
                          - 'max': For each stop, finds the route with the best (minimum) interval.
                          - 'agg': Aggregates all services at a stop to calculate one combined interval with the best direction per route.

        Returns:
            pl.LazyFrame: A LazyFrame containing mean interval calculations. The schema
                          depends on the chosen method.

        Raises:
            Exception: If an unimplemented `method` is provided.
        """
        # Filter the data first by date and time.
        gtfs_lf: pl.LazyFrame = self.filter_by_date_range(self.lf, start_date, end_date)
        gtfs_lf = self.filter_by_time_range(gtfs_lf, start_time, end_time)

        gtfs_lf = gtfs_lf.collect().lazy()

        end_time = utils.time_to_seconds(end_time.time())
        start_time = utils.time_to_seconds(start_time.time())

        # Group by stop, route, and direction to calculate intervals for each unique service path.
        gtfs_lf = (
            gtfs_lf.sort("departure_time")
            .group_by(["stop_id", "route_id", "direction_id"])
            .agg(
                [
                    pl.col("departure_time").alias("departure_times"),
                    # Calculate all intervals: time between departures, plus time from start/end of window.
                    (
                        # Interval from start_time to first departure and from last departure to end_time.
                        (pl.col("departure_times").list.get(0) - end_time)
                        .list.concat(pl.col("departure_times").list.diff().drop_nans())
                        .concat(
                            (end_time - pl.col("departure_times").list.get(-1)).list
                        )
                    ).alias("intervals"),
                ]
            )
            .with_columns(
                # Harmonic Mean: N / (sum of 1/x_i). Robust for averaging rates.
                (
                    pl.col("intervals").list.len()
                    / pl.col("intervals").list.eval(1 / pl.element()).list.sum()
                ).alias("mean_interval")
            )
            .collect()
            .lazy()
        )

        if method == "max":
            # For each stop, find the service (route/direction) with the minimum mean interval.
            gtfs_lf = gtfs_lf.group_by("stop_id").agg(
                [
                    pl.col("mean_interval").min().alias("mean_interval"),
                    pl.col("route_id")
                    .take(pl.col("mean_interval").arg_min())
                    .alias("route_id"),
                    pl.col("direction_id")
                    .take(pl.col("mean_interval").arg_min())
                    .alias("direction_id"),
                    pl.col("departure_times")
                    .take(pl.col("mean_interval").arg_min())
                    .alias("departure_times"),
                ]
            )

        elif method == "agg":
            # First, find the best interval per route at a stop (handling bidirectional routes).
            gtfs_lf = gtfs_lf.group_by("stop_id", "route_id").agg(
                [
                    pl.col("mean_interval").min().alias("mean_interval"),
                    pl.col("direction_id")
                    .take(pl.col("mean_interval").arg_min())
                    .alias("direction_id"),
                    pl.col("departure_times")
                    .take(pl.col("mean_interval").arg_min())
                    .alias("departure_times"),
                ]
            )
            # Then, aggregate all routes at the stop to get a combined interval.
            gtfs_lf = (
                gtfs_lf.group_by("stop_id")
                .agg(
                    [
                        pl.col("route_id").alias("route_ids"),
                        pl.col("direction_id").alias("direction_ids"),
                        pl.col("departure_times")
                        .flatten()
                        .sort()
                        .alias("departure_times"),
                    ]
                )
                .with_columns(
                    # Recalculate intervals and harmonic mean for the combined departure times.
                    (
                        (pl.col("departure_times").list.get(0) - start_time)
                        .list.concat(pl.col("departure_times").list.diff().drop_nans())
                        .concat(
                            (end_time - pl.col("departure_times").list.get(-1)).list
                        )
                    ).alias("intervals")
                )
                .with_columns(
                    (
                        pl.col("intervals").list.len()
                        / pl.col("intervals").list.eval(1 / pl.element()).list.sum()
                    ).alias("mean_interval")
                )
            )

        elif method != "route":
            raise Exception(
                f"method '{method}' not implemented. Choose from 'route', 'max', or 'agg'."
            )

        return gtfs_lf
