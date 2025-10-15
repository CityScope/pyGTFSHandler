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


Columns that might have duplicates between gtfs files: stop_id, route_id
Columns that autoresolve duplicates between gtfs files: trip_id, service_id, shape_id
TODOs:
- TODO when grouping by parent_station shape_direction should be what it is for the direction with the most remaining stops and the oposite for
the direction with less remaining stops
- TODO avoid performing non necesary checks at the begining. Do first all the filters and then the checks.
- TODO check that all trips have a route and if not generate it.
- TODO check that trips from stop_times have a trip in trips and same for stops
- TODO add direction_id to routes based on shape direction per shape_id
- TODO finish dealing with shape_id and shapes.txt
- TODO If in a trip_id a stop_id is repeated divide it into 2 trip ids check if this is really needed
"""

from .models import StopTimes, Stops, Trips, Calendar, Routes, Shapes
from . import utils

from pathlib import Path
from datetime import datetime, time, date
from typing import Optional, Union, List
import geopandas as gpd
import polars as pl

SECS_PER_DAY: int = 86400


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
        stop_group_distance: float = 0,
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        date_type: Optional[list[str] | str] = None,
        start_time: Optional[datetime | time] = None,
        end_time: Optional[datetime | time] = None,
        route_types: Optional[list[int] | list[str] | int | str] = None,
        service_ids: Optional[List[str]] = None,
        trip_ids: Optional[List[str]] = None,
        stop_ids: Optional[List[str]] = None,
        route_ids: Optional[List[str]] = None,
        check_files:bool=True
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

        route_types_print = f"route types {route_types}" if route_types is not None else ""
        time_range_print = (
            f"time range {start_date} - {end_date}"
            if (start_date is not None or end_date is not None)
            else ""
        )
        aoi_print = f"aoi {aoi.geometry.union_all()}" if aoi is not None else ""
        error_msg = f"No trips with your id filters and filters {route_types_print} {time_range_print} {aoi_print}".strip()

        self.stops = Stops(
            self.gtfs_dir,
            aoi=aoi,
            stop_group_distance=stop_group_distance,
            stop_ids=stop_ids,
            check_files=check_files
        )

        if (self.stops.stop_ids is not None) and (len(self.stops.stop_ids) == 0):
            raise Exception(f"No stops found inside your aoi")

        self.calendar = Calendar(
            self.gtfs_dir,
            start_date=start_date,
            end_date=end_date,
            date_type=date_type,
            service_ids=service_ids,
            lon=self.stops.mean_lon,
            lat=self.stops.mean_lat,
            check_files=check_files
        )

        if route_types is not None:
            if route_types is list:
                route_types = [utils.normalize_route_type(i) for i in route_types]
            else:
                route_types = [utils.normalize_route_type(route_types)]

        self.routes = Routes(
            self.gtfs_dir, route_ids=route_ids, route_types=route_types, check_files=check_files
        )

        if (self.routes.route_ids is not None) and (len(self.routes.route_ids) == 0):
            raise Exception(f"No routes found with filter {route_types}")

        if (self.calendar.service_ids is not None) and (len(self.calendar.service_ids) == 0):
            raise Exception(f"No trips found in time range {start_date} - {end_date}")

        self.trips = Trips(
            self.gtfs_dir,
            service_ids=self.calendar.service_ids,
            trip_ids=trip_ids,
            route_ids=self.routes.route_ids,
            check_files=check_files
        )

        if (self.trips.trip_ids is not None) and (len(self.trips.trip_ids) == 0):
            raise Exception(error_msg)


        self.stop_times = StopTimes(
            self.gtfs_dir,
            trips=self.trips.lf,
            start_time=start_time,
            end_time=end_time,
            stop_ids=self.stops.stop_ids,
            trip_ids=self.trips.trip_ids,
            check_files=check_files
        )

        if self.stop_times.lf.select(pl.count()).collect().item() == 0:
            raise Exception(error_msg)

        self.trips.lf = self.stop_times.trips_lf

        # Reload stops_lf so that at least in the lf the next stop of bordering trips is loaded

        self.stops.reload_stops_lf(self.gtfs_dir, self.stop_times.lf.select("stop_id"))

        # --- 3. Integrate Generated Trips from Frequencies ---
        # If StopTimes generated new trips from frequencies.txt, we need to add them
        # to the main trips table.

        # --- 4. Load Shapes and Perform Advanced Time Interpolation ---
        self.trip_shape_ids_lf: pl.LazyFrame = (
            self.stop_times.generate_shape_ids().collect().lazy()
        )
        self.shapes = Shapes(self.gtfs_dir, self.trip_shape_ids_lf, self.stops.lf, check_files=check_files)

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
                "gtfs_name",
                "file_id",
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
            self.trips.lf.filter(~pl.col("next_day")).select(
                ["trip_id", "service_id", "route_id"]
            ),
            on="trip_id",
            how="left",
        )
        self.lf = self.lf.join(
            self.trip_shape_ids_lf.select(["trip_ids", "shape_id"])
            .explode("trip_ids")
            .rename({"trip_ids": "trip_id"}),
            on="trip_id",
            how="left",
        )

        self.lf = self.lf.join(
            self.stops.lf.select(["stop_id", "parent_station"]),
            on=["stop_id"],
            how="left",
        )

        # Ensure that every trip does not stop twice at the parent_station

        # Sort by trip_id and stop_sequence
        self.lf = self.lf.sort(
            ["trip_id", "service_id", "route_id", "shape_id", "stop_sequence"]
        )

        # Create a new column with shifted parent_station per trip_id group
        self.lf = self.lf.with_columns(
            [
                pl.col("parent_station")
                .shift(1)
                .over("trip_id")
                .alias("prev_parent_station")
            ]
        )

        # Replace duplicate consecutive parent_station with None
        self.lf = self.lf.with_columns(
            [
                pl.when(pl.col("parent_station") == pl.col("prev_parent_station"))
                .then(None)
                .otherwise(pl.col("parent_station"))
                .alias("parent_station")
            ]
        )

        # Drop helper column if you don't want to keep it
        self.lf = self.lf.drop("prev_parent_station")

        self.lf = self.lf.join(
            self.routes.lf.select(["route_id", "route_type"]),
            on=["route_id"],
            how="left",
        )

        self.lf = self.lf.join(
            self.shapes.stop_shapes.select(
                [
                    "shape_id",
                    "stop_id",
                    "stop_sequence",
                    "shape_dist_traveled",
                    "shape_total_distance",
                    "shape_direction",
                    "shape_direction_backwards",
                ]
            ),
            on=["stop_id", "shape_id", "stop_sequence"],
            how="left",
        )

        # --- Perform Final Data Cleaning and Transformation ---
        # If any times were fixed with the simple method, run the advanced,
        # shape-based interpolation now that shape_dist_traveled is available.
        if self.stop_times.fixed_times:
            self.lf = self.__fix_null_times(self.lf)

        self.lf = self.lf.drop("fixed_time")

        # For services running past midnight, create a unique "night" service_id
        # to distinguish them from the same service on the previous day.
        self.lf = self.lf.with_columns(
            [
                pl.when(pl.col("next_day"))
                .then(pl.concat_str(pl.col("service_id"), pl.lit("_night")))
                .otherwise(pl.col("service_id"))
                .alias("service_id")
            ]
        ).drop("next_day")

        self.lf = self.lf.unique()

        self.lf = self.lf.join(
            self.stops.lf.select(["stop_id"]).with_columns(pl.lit(True).alias("isin_aoi")),
            on="stop_id",
            how="left"
        )
        self.lf = self.lf.with_columns(pl.col("isin_aoi").fill_null(False))

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
                .alias("departure_time"),
                pl.when(pl.col("fixed_time"))
                .then(None)
                .otherwise(pl.col("shape_dist_traveled"))
                .alias("shape_dist_traveled_copy"),
            ]
        )

        # Create context columns: the next and previous known time/distance points.
        stop_times = stop_times.with_columns(
            [
                pl.col("departure_time")
                .forward_fill()
                .over("trip_id")
                .alias("dep_time_fwd"),
                pl.col("shape_dist_traveled_copy")
                .forward_fill()
                .over("trip_id")
                .alias("dist_fwd"),
                pl.col("departure_time")
                .backward_fill()
                .over("trip_id")
                .alias("dep_time_bwd"),
                pl.col("shape_dist_traveled_copy")
                .backward_fill()
                .over("trip_id")
                .alias("dist_bwd"),
            ]
        )

        # Apply linear interpolation for rows where departure_time is null.
        stop_times = stop_times.with_columns(
            [
                pl.when(
                    pl.col("departure_time").is_null()
                    | pl.col("departure_time").is_nan()
                )
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
                            (pl.col("dep_time_bwd") + SECS_PER_DAY)
                            - pl.col("dep_time_fwd")
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

        stop_times = stop_times.with_columns(
            pl.col("departure_time").round(0).cast(int).alias("departure_time"),
            pl.col("arrival_time").round(0).cast(int).alias("arrival_time"),
        )

        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("fixed_time"))
                .then(pl.col("departure_time"))
                .otherwise(pl.col("arrival_time"))
                .alias("arrival_time")
            ]
        )

        # Clean up temporary helper columns.
        stop_times = stop_times.drop(
            [
                "dep_time_fwd",
                "dep_time_bwd",
                "dist_fwd",
                "dist_bwd",
                "shape_dist_traveled_copy",
            ]
        )

        # Recalculate travel times now that nulls are filled.
        stop_times = self.stop_times._add_shape_time_and_midnight_crossing(stop_times)

        return stop_times

    def _frequencies_to_stop_times(self, gtfs_lf):
        gtfs_lf = gtfs_lf.collect()
        frequencies_exist = (self.stop_times.frequencies is not None) and (
            gtfs_lf.select(
                (
                    (~pl.col("start_time").is_null()) & (~pl.col("start_time").is_nan())
                ).any()
            ).item()
        )
        gtfs_lf = gtfs_lf.lazy()

        if frequencies_exist:
            gtfs_lf_frequencies = (
                gtfs_lf.filter(
                    (~pl.col("start_time").is_null()) & (~pl.col("start_time").is_nan())
                )
                .with_columns(
                    (
                        (
                            (
                                pl.col("start_time")
                                - pl.col("departure_time")
                                + pl.col("shape_time_traveled")
                            )
                            / pl.col("headway_secs")
                        ).ceil()
                        * pl.col("headway_secs")
                        + pl.col("departure_time")
                    ).alias("aligned_start"),
                )
                .with_columns(
                    [
                        pl.int_ranges(
                            pl.col("aligned_start"),
                            pl.col("end_time") + 1 + pl.col("shape_time_traveled"),
                            pl.col("headway_secs"),
                        ).alias("new_departure_time")
                    ]
                )
                .explode("new_departure_time")
                .with_columns(
                    pl.concat_str(
                        [
                            pl.col("trip_id"),
                            pl.lit("_"),
                            (
                                (
                                    pl.col("new_departure_time")
                                    - pl.col("start_time")
                                    - pl.col("shape_time_traveled")
                                )
                                / pl.col("headway_secs")
                            )
                            .ceil()
                            .cast(int),
                        ]
                    ).alias("trip_id")
                )
            ).drop("aligned_start")

            gtfs_lf_times = gtfs_lf.filter(
                (pl.col("start_time").is_null()) | (pl.col("start_time").is_nan())
            ).with_columns(pl.col("departure_time").alias("new_departure_time"))

            gtfs_lf = pl.concat([gtfs_lf_frequencies, gtfs_lf_times])

            gtfs_lf = gtfs_lf.with_columns(
                (
                    pl.col("arrival_time")
                    - pl.col("departure_time")
                    + pl.col("new_departure_time")
                ).alias("arrival_time"),
                (pl.col("new_departure_time")).alias("departure_time"),
            ).drop("new_departure_time")

            gtfs_lf = gtfs_lf.with_columns(
                pl.lit(None).alias("start_time"),
                pl.lit(None).alias("end_time"),
                pl.lit(None).alias("headway_secs"),
                pl.lit(1).alias("n_trips"),
            )

        return gtfs_lf

    def calendar_new_end_date(self, new_end_date: datetime | date, file_id=None,gtfs_name=None):
        end_date = int(utils.datetime_to_days_since_epoch(new_end_date))
        if self.calendar.lf is not None:
            if file_id is not None:
                self.calendar.lf = self.calendar.lf.with_columns(
                    pl.when(
                        pl.col("file_id") == pl.lit(file_id)
                    ).then(
                    pl.lit(end_date)
                    ).otherwise(pl.col("end_date")
                    ).alias("end_date")
                )
            elif gtfs_name is not None:
                self.calendar.lf = self.calendar.lf.with_columns(
                    pl.when(
                        pl.col("gtfs_name") == pl.lit(gtfs_name)
                    ).then(
                    pl.lit(end_date)
                    ).otherwise(pl.col("end_date")
                    ).alias("end_date")
                )
            else:
                self.calendar.lf = self.calendar.lf.with_columns(
                    pl.lit(end_date).alias("end_date")
                )

    def calendar_new_start_date(self, new_start_date: datetime | date, file_id=None,gtfs_name=None):
        start_date = int(utils.datetime_to_days_since_epoch(new_start_date))
        if self.calendar.lf is not None:
            if file_id is not None:
                self.calendar.lf = self.calendar.lf.with_columns(
                    pl.when(
                        pl.col("file_id") == pl.lit(file_id)
                    ).then(
                    pl.lit(start_date)
                    ).otherwise(pl.col("start_date")
                    ).alias("start_date")
                )
            elif gtfs_name is not None:
                self.calendar.lf = self.calendar.lf.with_columns(
                    pl.when(
                        pl.col("gtfs_name") == pl.lit(gtfs_name)
                    ).then(
                    pl.lit(start_date)
                    ).otherwise(pl.col("start_date")
                    ).alias("start_date")
                )
            else:
                self.calendar.lf = self.calendar.lf.with_columns(
                    pl.lit(start_date).alias("start_date")
                )
                
    def filter_by_date_range(
        self,
        data: pl.LazyFrame,
        start_date: datetime | date = None,
        end_date: datetime | date = None,
        date_type: str | list[str] = None,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a date range.

        It uses the `Calendar` object to find all `service_id`s active within
        the specified date range and then semi-joins the input data with these
        service IDs.

        Args:
            data (pl.LazyFrame): The LazyFrame to be filtered. Must contain a `service_id` column.
            start_date (datetime): The start of the date range (inclusive).
            end_date (datetime): The end of the date range (inclusive).

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """
        date_df: pl.DataFrame = self.calendar.get_services_in_date_range(
            start_date,
            end_date,
            date_type=date_type,
            lon=self.stops.mean_lon,
            lat=self.stops.mean_lat,
        )
        date_df = date_df.explode("service_ids").rename({"service_ids": "service_id"})
        data = data.join(
            date_df.select("service_id").lazy(), on="service_id", how="semi"
        )
        return data

    def filter_by_date(
        self,
        data: pl.LazyFrame,
        date: datetime | date,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a date.

        It uses the `Calendar` object to find all `service_id`s active within
        the specified date range and then semi-joins the input data with these
        service IDs.

        Args:
            data (pl.LazyFrame): The LazyFrame to be filtered. Must contain a `service_id` column.
            date (datetime): The desired date.

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """

        service_ids = self.calendar.get_services_in_date(date)
        if len(service_ids) == 0:
            raise Exception(f"No services in date {date}")

        service_ids_df = pl.LazyFrame({"service_id": service_ids})
        data = data.join(
            service_ids_df.select("service_id"), on="service_id", how="semi"
        )
        return data

    def filter_by_time_range(
        self,
        data: pl.LazyFrame,
        start_time: datetime | time = datetime.min,
        end_time: datetime | time = datetime.max,
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
        start_time: int = utils.time_to_seconds(start_time)
        end_time: int = utils.time_to_seconds(end_time)

        # If frequencies are present, filter based on both frequency windows and explicit times.
        if self.stop_times.frequencies is not None:
            # Keep rows if their frequency window overlaps the filter time.
            data = data.filter(
                pl.col("start_time").is_null()
                | pl.col("start_time").is_nan()
                | (
                    (pl.col("end_time") > start_time)
                    & (pl.col("start_time") < end_time)
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

            data = (
                data.with_columns(
                    [
                        # Clip start_time to be no earlier than global start_time
                        pl.when(
                            pl.col("start_time").is_null()
                            | pl.col("start_time").is_nan()
                        )
                        .then(pl.col("start_time"))
                        .otherwise(
                            pl.when(pl.col("start_time") < start_time)
                            .then(start_time)
                            .otherwise(pl.col("start_time"))
                        )
                        .alias("start_time"),
                        # Clip end_time to be no later than global end_time
                        pl.when(
                            pl.col("end_time").is_null() | pl.col("end_time").is_nan()
                        )
                        .then(pl.col("end_time"))
                        .otherwise(
                            pl.when(pl.col("end_time") > end_time)
                            .then(end_time)
                            .otherwise(pl.col("end_time"))
                        )
                        .alias("end_time"),
                    ]
                )
                .with_columns(
                    [
                        # Compute number of trips
                        pl.when(
                            pl.col("start_time").is_null()
                            | pl.col("start_time").is_nan()
                        )
                        .then(pl.lit(1))
                        .otherwise(
                            (
                                (pl.col("end_time") - pl.col("start_time"))
                                / pl.col("headway_secs")
                            )
                            .ceil()
                            .cast(pl.UInt32)
                        )
                        .alias("n_trips")
                    ]
                )
                .filter(pl.col("n_trips") > 0)
            )
        else:
            # If no frequencies, just filter by explicit departure/arrival times.
            data = data.filter(
                (pl.col("departure_time") >= start_time)
                & (pl.col("arrival_time") <= end_time)
            )

        return data

    def filter_by_route_type(
        self, data: pl.LazyFrame, route_types: list | int | str
    ) -> pl.LazyFrame:
        if isinstance(route_types, list):
            route_types = [utils.normalize_route_type(i) for i in route_types]
        else:
            route_types = [utils.normalize_route_type(route_types)]

        route_types_df = pl.DataFrame({"route_type": route_types})
        data = data.join(route_types_df.lazy(), on="route_type", how="semi")
        return data

    def get_service_intensity_in_date_range(
        self,
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        date_type: Optional[str | list[str]] = None,
        route_types: Optional[str | int | list[str] | list[int]] = None,
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
            start_date,
            end_date,
            date_type=date_type,
            lon=self.stops.mean_lon,
            lat=self.stops.mean_lat,
        )

        gtfs_lf = self.lf
        gtfs_lf = gtfs_lf.filter(pl.col("isin_aoi"))
        if route_types is not None:
            gtfs_lf = self.filter_by_route_type(gtfs_lf, route_types=route_types)

        if self.stop_times.frequencies is None:
            gtfs_lf: pl.LazyFrame = gtfs_lf.unique(
                ["service_id", "stop_id", "departure_time"]
            ).select("trip_id", "service_id", "n_trips")
        else:
            gtfs_lf: pl.LazyFrame = gtfs_lf.unique(
                ["service_id", "stop_id", "departure_time", "start_time", "end_time"]
            ).select("trip_id", "service_id", "n_trips")

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

        total_by_date = self.calendar.add_holidays_and_weekends(
            total_by_date, lon=self.stops.mean_lon, lat=self.stops.mean_lat
        )

        return total_by_date.sort("date")

    def get_mean_intervall_at_stops(
        self,
        date: datetime | date,
        start_time: datetime | time = time.min,
        end_time: datetime | time = time.max,
        route_types: list | int | str | None = None,
        on: str = "parent_station",
        method: str = "route",
        n_divisions: int = 1,
    ) -> pl.LazyFrame:
        """
        Calculates the mean interval (headway) between services at stops.

        This method computes the headway for a given time window using the harmonic
        mean, which is the appropriate method for averaging rates like service
        frequencies. The calculation can be performed in several ways depending on
        the `method` parameter.

        Args:
            date: The specific date to filter the data for. If None, data for all
                dates in the feed will be used.
            start_time: The start of the time window for the analysis. Defaults to
                        the beginning of the day (00:00:00).
            end_time: The end of the time window for the analysis. Defaults to
                    the end of the day (23:59:59.999999).
            method: The aggregation method. Must be one of ['route', 'max', 'agg'].
                    - 'route': (Default) Calculates mean interval per route and
                    direction at each stop.
                    - 'max': For each stop, finds the route with the best (i.e.,
                    minimum) interval and returns only that result for the stop.
                    - 'agg': Aggregates all services at a stop to calculate a
                    single combined interval. It groups services by direction,
                    finds the best service per direction, and then combines the
                    headways of the different directions.
            n_divisions: The number of direction pairs (e.g., N/S, E/W) to
                        consider when `method` is 'agg'. For example, `n_divisions=2`
                        would create four directional sectors (e.g., N, E, S, W).
                        Only used when `method` is 'agg'. Defaults to 1.

        Returns:
            A Polars LazyFrame with mean interval calculations. The schema depends
            on the chosen `method`:
            - 'route': ["parent_station", "route_id", "direction_id", "mean_interval", ...]
            - 'max': ["parent_station", "route_id", "direction_id", "mean_interval", ...]
            - 'agg': ["parent_station", "mean_interval", "shape_directions", "shape_ids", "route_ids"]

        Raises:
            ValueError: If an unimplemented `method` is provided.
        """
        # --- 1. Pre-computation and Filtering ---

        if on == "stop_id":
            on = ["stop_id", "gtfs_name"]  # to avoid duplicated stop ids
        else:
            on = [on]

        # Fail early for invalid methods.
        valid_methods = ["route", "max", "agg"]
        if method not in valid_methods:
            raise ValueError(
                f"Method '{method}' is not implemented. Choose from {valid_methods}."
            )

        gtfs_lf = self.lf

        gtfs_lf = gtfs_lf.filter(pl.col("isin_aoi"))

        gtfs_lf = gtfs_lf.filter(
            pl.col("stop_sequence") != pl.col("stop_sequence").max().over("trip_id")
        )

        if route_types is not None:
            gtfs_lf = self.filter_by_route_type(gtfs_lf, route_types=route_types)

        # Filter the GTFS data by the specified date and time window.
        gtfs_lf = self.filter_by_date(gtfs_lf, date)

        gtfs_lf = self.filter_by_time_range(gtfs_lf, start_time, end_time)

        gtfs_lf = self._frequencies_to_stop_times(gtfs_lf)

        gtfs_lf = self.filter_by_time_range(gtfs_lf, start_time, end_time)

        # Convert time objects to total seconds from midnight for interval calculations.
        start_time_sec = utils.time_to_seconds(start_time)
        end_time_sec = utils.time_to_seconds(end_time)

        gtfs_lf = gtfs_lf.collect().lazy()
        # Per-Route Interval Calculation ('route' and 'max' methods)

        if method in ("max", "route"):
            # This block calculates the mean interval for each distinct service,
            # defined by a unique combination of stop, route, and direction.
            gtfs_lf = (
                gtfs_lf.sort("departure_time")
                .group_by([*on, "route_id", "direction_id"])
                .agg(
                    [
                        # Collect all departure times into a list for each service group.
                        pl.col("departure_time").sort().alias("departure_times"),
                        # Get a representative direction (in degrees) for the service.
                        utils.mean_angle("shape_direction").alias("shape_direction"),
                        utils.mean_angle("shape_direction_backwards").alias(
                            "shape_direction_backwards"
                        ),
                        (
                            (pl.col("departure_time").min() - start_time_sec)
                            + (end_time_sec - pl.col("departure_time").max())
                        ).alias("initial_interval"),
                    ]
                )
            )

            gtfs_lf = (
                gtfs_lf.with_columns(
                    # Calculate the harmonic mean of the intervals.
                    # Harmonic Mean = N / sum(1/x_i), where x_i are the intervals.
                    # This is the correct way to average rates (like service frequency).
                    (
                        (
                            (pl.col("departure_times").list.diff(null_behavior="drop"))
                            .list.eval(pl.element().pow(2))
                            .list.sum()
                            + pl.col("initial_interval") ** 2
                        )
                        / (end_time_sec - start_time_sec)
                    ).alias("mean_interval"),
                )
                .drop("initial_interval")
                # Another optimization fence after the first major aggregation.
                .collect()
                .lazy()
            )

        # Method-Specific Aggregation

        if method == "max":
            # For each stop, find the single service (route/direction) that has
            # the minimum mean interval (i.e., the highest frequency).
            gtfs_lf = gtfs_lf.group_by(on).agg(
                [
                    # Find the minimum interval at the stop.
                    pl.col("route_id")
                    .sort_by("mean_interval")
                    .first()
                    .alias("route_id"),
                    pl.col("direction_id")
                    .sort_by("mean_interval")
                    .first()
                    .alias("direction_id"),
                    pl.col("departure_times")
                    .sort_by("mean_interval")
                    .first()
                    .alias("departure_times"),
                    pl.col("mean_interval").min().alias("mean_interval"),
                ]
            )

        elif method == "agg":
            # This method provides a single, combined headway for all services at a stop.
            # It works by grouping services into directional corridors, finding the
            # best headway in each corridor, and then combining these headways.

            # Create sectors for each direction and its opposite (e.g., N/S are one pair).
            # E.g., n_divisions=2 -> 4 sectors (N, E, S, W).
            n_sectors = n_divisions * 2
            # Bin services into directional sectors.shape_direction_backwards

            gtfs_lf = (
                gtfs_lf.with_columns(  # This ensures a separation between both shape directions of exactly 180º
                    (
                        pl.when(
                            (
                                (
                                    pl.col("shape_direction")
                                    + 360
                                    - pl.col("shape_direction_backwards")
                                )
                                % 360
                            )
                            > (
                                (
                                    pl.col("shape_direction_backwards")
                                    + 360
                                    - pl.col("shape_direction")
                                )
                                % 360
                            )
                        )
                        .then(
                            -1
                            * (
                                180
                                - (
                                    (
                                        pl.col("shape_direction_backwards")
                                        + 360
                                        - pl.col("shape_direction")
                                    )
                                    % 360
                                )
                            )
                            / 2
                        )
                        .otherwise(
                            (
                                180
                                - (
                                    (
                                        pl.col("shape_direction")
                                        + 360
                                        - pl.col("shape_direction_backwards")
                                    )
                                    % 360
                                )
                            )
                            / 2
                        )
                    ).alias("shape_diff")
                )
                .with_columns(
                    (
                        pl.when(
                            pl.col("shape_diff").is_null()
                            | pl.col("shape_diff").is_nan()
                        )
                        .then(pl.lit(0))
                        .otherwise(pl.col("shape_diff"))
                    ).alias("shape_diff")
                )
                .with_columns(
                    (
                        (pl.col("shape_direction") + 360 + pl.col("shape_diff")) % 360
                    ).alias("shape_direction"),
                )
                .drop("shape_diff")
            )
            gtfs_lf = gtfs_lf.group_by(on).agg(pl.all())

            gtfs_lf = gtfs_lf.collect()
            gtfs_lf = gtfs_lf.with_columns(
                utils.max_separation_angle(gtfs_lf, "shape_direction").alias(
                    "shape_split_direction"
                )
            ).explode(pl.exclude([*on, "shape_split_direction"]))

            gtfs_lf = gtfs_lf.lazy()

            if n_divisions % 2 == 0:
                gtfs_lf = gtfs_lf.with_columns(
                    pl.col("shape_split_direction") + 90 / n_divisions
                )

            gtfs_lf = gtfs_lf.with_columns(pl.col("shape_split_direction") % 360)

            gtfs_lf = (
                gtfs_lf.with_columns(
                    (
                        (
                            pl.col("shape_direction")
                            - pl.col("shape_split_direction")
                            + 360
                        )
                        % 360
                    ).alias("angle")
                )
                .with_columns(
                    ((pl.col("angle") * n_sectors / 360).floor().cast(pl.Int32)).alias(
                        "shape_direction_id"
                    )
                )
                .drop("angle")
            )

            # Assign each service to a directional sector ID (0 to n_sectors-1)
            # based on its angle relative to the stop's mean direction.
            # Build chained when/then expression

            # Calculate headway for each directional bin.
            gtfs_lf = (
                gtfs_lf.sort("departure_time")
                .group_by([*on, "shape_direction_id"])
                .agg(
                    [
                        pl.col("departure_time").sort().alias("departure_times"),
                        utils.mean_angle("shape_direction").alias("shape_direction"),
                        pl.col("route_id").unique().alias("route_ids"),
                        pl.col("shape_id").unique().alias("shape_ids"),
                        (
                            (pl.col("departure_time").min() - start_time_sec)
                            + (end_time_sec - pl.col("departure_time").max())
                        ).alias("initial_interval"),
                    ]
                )
            )

            gtfs_lf = gtfs_lf.with_columns(
                [
                    (
                        (
                            (pl.col("departure_times").list.diff(null_behavior="drop"))
                            .list.eval(pl.element().pow(2))
                            .list.sum()
                            + pl.col("initial_interval") ** 2
                        )
                        / (end_time_sec - start_time_sec)
                    ).alias("mean_interval"),
                    (pl.col("shape_direction_id") % n_divisions).alias(
                        "shape_direction_group_id"
                    ),
                ]
            )

            # For each direction *group* (e.g., N/S), find the best headway.
            # This selects the best-performing service from a set of parallel services.
            gtfs_lf = gtfs_lf.group_by([*on, "shape_direction_group_id"]).agg(
                [
                    # Keep the details of the service with the minimum interval.
                    pl.col("shape_direction")
                    .sort_by("mean_interval")
                    .first()
                    .alias("shape_direction"),
                    pl.col("shape_ids")
                    .sort_by("mean_interval")
                    .first()
                    .alias("shape_ids"),
                    pl.col("route_ids")
                    .sort_by("mean_interval")
                    .first()
                    .alias("route_ids"),
                    pl.col("departure_times")
                    .sort_by("mean_interval")
                    .first()
                    .alias("departure_times"),
                    # The minimum interval for this direction group.
                    pl.col("mean_interval").min().alias("mean_interval"),
                ]
            )

            # Combine the best headways from all direction groups for each stop.
            gtfs_lf = gtfs_lf.group_by(on).agg(
                [
                    # The combined headway is the harmonic mean of the individual best headways.
                    # Formula: 1 / (1/H_1 + 1/H_2 + ...), where H_i is the headway.
                    (1 / (1 / pl.col("mean_interval")).sum()).alias("mean_interval"),
                    # Collect metadata from all contributing direction groups.
                    pl.col("shape_direction").alias("shape_directions"),
                    # Flatten list[list[str]] -> list[str]
                    (pl.col("shape_ids").flatten().unique()).alias("shape_ids"),
                    (pl.col("departure_times").flatten().unique().sort()).alias(
                        "departure_times"
                    ),
                    (pl.col("route_ids").flatten().unique()).alias("route_ids"),
                ]
            )

        return gtfs_lf.collect()
