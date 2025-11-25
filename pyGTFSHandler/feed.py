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

"""TODO: revise filter_by_time_range with frequencies
TODO: revise interval func to work with all possible by, at and hows especially 'shape_direction' 'all' instead of 'max'
"""

from .models import StopTimes, Stops, Trips, Calendar, Routes, Shapes
from . import utils
from . import gtfs_checker

from pathlib import Path
from datetime import datetime, time, date
from typing import Optional, Union, List
import geopandas as gpd
import polars as pl
import pandas as pd 
import warnings 
import numpy as np 

SECS_PER_DAY: int = 86400


def concat_feeds(
    feeds,
    stop_group_distance=0
):  
    if isinstance(feeds,Feed):
        return feeds

    result = feeds[0]
    if len(feeds) == 0:
        return result
    
    calendar_lf = []
    calendar_exceptions_lf = []
    routes_lf = []
    #shapes_lf = []
    #shapes_stop_shapes = []
    #shapes_gdf = []
    stop_times_lf = []
    stop_times_frequencies = []
    stops_lf = []
    stops_gdf = []
    trips_lf = [] 
    #lf = []
    for feed in feeds:
        calendar_lf.append(feed.calendar.lf)
        calendar_exceptions_lf.append(feed.calendar.exceptions_lf)
        routes_lf.append(feed.routes.lf)
        #shapes_lf.append(feed.shapes.lf)
        #shapes_stop_shapes.append(feed.shapes.stop_shapes)
        #shapes_gdf.append(feed.shapes.gdf)
        stop_times_lf.append(feed.stop_times.lf)
        stop_times_frequencies.append(feed.stop_times.frequencies)
        stops_lf.append(feed.stops.lf)
        stops_gdf.append(feed.stops.gdf)
        trips_lf.append(feed.trips.lf)
        #lf.append(feed.lf)

    calendar_lf = pl.concat(calendar_lf)
    calendar_exceptions_lf = pl.concat(calendar_exceptions_lf)
    calendar = Calendar(lf=calendar_lf,exceptions_lf=calendar_exceptions_lf)

    routes_lf = pl.concat(routes_lf)
    routes = Routes(lf=routes_lf)

    #shapes_lf = pl.concat(shapes_lf)
    #shapes_stop_shapes = pl.concat(shapes_stop_shapes)
    #shapes_gdf = pd.concat(shapes_gdf)
    #shapes = Shapes(lf=shapes_lf,stop_shapes=shapes_stop_shapes,gdf=shapes_gdf)

    stop_times_lf = pl.concat(stop_times_lf)
    stop_times_frequencies = pl.concat(stop_times_frequencies)
    stop_times = StopTimes(lf=stop_times_lf,frequencies=stop_times_frequencies,fixed_times=False)

    stops_lf = pl.concat(stops_lf)
    stops_gdf = pd.concat(stops_gdf)
    stops = Stops(lf=stops_lf,gdf=stops_gdf)

    trips_lf = pl.concat(trips_lf)
    trips = Trips(lf=trips_lf)

    #lf = pl.concat(lf)

    result.calendar = calendar 
    result.routes = routes 
    result.stop_times = stop_times 
    result.stops = stops 
    result.trips = trips 

    if stop_group_distance > 0:
        result.stops.lf, result.stops.gdf = result.stops.group_stops(stop_group_distance)

    result.shapes, result.trip_shape_ids_lf = result.load_shapes(result.stops,result.stop_times)
 
    result.lf = result.build_lf(result.calendar, result.routes, result.shapes, result.stop_times, result.stops, result.trips, result.trip_shape_ids_lf)

    return result

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
        check_files:bool=True,
        min_file_id=0
    ):
        self.calendar, self.routes, _, self.stop_times, self.stops, self.trips = self.load(
            gtfs_dirs=gtfs_dirs,
            aoi=aoi,
            stop_group_distance=stop_group_distance,
            start_date=start_date,
            end_date=end_date,
            date_type=date_type,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            service_ids=service_ids,
            trip_ids=trip_ids,
            stop_ids=stop_ids,
            route_ids=route_ids,
            check_files=check_files,
            min_file_id=min_file_id
        )

        self.shapes, self.trip_shape_ids_lf = self.load_shapes(self.stops,self.stop_times)

        self.lf = self.build_lf(
            self.calendar, 
            self.routes, 
            self.shapes, 
            self.stop_times, 
            self.stops, 
            self.trips,
            self.trip_shape_ids_lf
        )

    def load(
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
        check_files:bool=True,
        min_file_id:int=0
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

        gtfs_dir: List[Path] = [Path(p) for p in gtfs_dirs]

        for i in range(len(gtfs_dir)):
            if gtfs_dir[i].is_file():
                orig_file = gtfs_dir[i]
                gtfs_dir[i] = Path(gtfs_checker.unzip(gtfs_dir[i],delete=False))
                warnings.warn(f"Extracting {orig_file} to {gtfs_dir[i]}")

            if not gtfs_dir[i].is_dir():
                raise ValueError(f"{gtfs_dir[i]} is not a valid directory.")

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

        stops = Stops()
        stops.load(
            gtfs_dir,
            aoi=aoi,
            stop_group_distance=stop_group_distance,
            stop_ids=stop_ids,
            check_files=check_files,
            min_file_id=min_file_id
        )

        if (stops.stop_ids is not None) and (len(stops.stop_ids) == 0):
            raise Exception(f"No stops found inside your aoi")

        calendar = Calendar()
        calendar.load(
            gtfs_dir,
            start_date=start_date,
            end_date=end_date,
            date_type=date_type,
            service_ids=service_ids,
            lon=stops.mean_lon,
            lat=stops.mean_lat,
            check_files=check_files,
            min_file_id=min_file_id
        )

        if (route_types == 'all') or (route_types is None) or ('all' in route_types) or (None in route_types):
            route_types = None 

        if route_types is not None:
            if isinstance(route_types, list):
                route_types = [gtfs_checker.normalize_route_type(i) for i in route_types]
            else:
                route_types = [gtfs_checker.normalize_route_type(route_types)]

        routes = Routes()
        routes.load(
            gtfs_dir, route_ids=route_ids, route_types=route_types, check_files=check_files, min_file_id=min_file_id
        )

        if (routes.route_ids is not None) and (len(routes.route_ids) == 0):
            raise Exception(f"No routes found with filter {route_types}")

        if (calendar.service_ids is not None) and (len(calendar.service_ids) == 0):
            raise Exception(f"No trips found in time range {start_date} - {end_date}")

        trips = Trips()
        trips.load(
            gtfs_dir,
            service_ids=calendar.service_ids,
            trip_ids=trip_ids,
            route_ids=routes.route_ids,
            check_files=check_files,
            min_file_id=min_file_id
        )

        if (trips.trip_ids is not None) and (len(trips.trip_ids) == 0):
            raise Exception(error_msg)


        stop_times = StopTimes()
        stop_times.load(
            gtfs_dir,
            trips=trips.lf,
            start_time=start_time,
            end_time=end_time,
            stop_ids=stops.stop_ids,
            trip_ids=trips.trip_ids,
            check_files=check_files,
            min_file_id=min_file_id
        )

        if stop_times.lf.select(pl.count()).collect().item() == 0:
            raise Exception(error_msg)

        trips.lf = stop_times.trips_lf

        # Reload stops_lf so that at least in the lf the next stop of bordering trips is loaded

        self.stop_ids_in_aoi = (
            stops.lf.select(pl.col("stop_id").unique())
            .collect()
            .to_series()
            .to_list()
        )
        stops.reload_stops_lf(gtfs_dir, stop_times.lf.select("stop_id"))

        # --- 3. Integrate Generated Trips from Frequencies ---
        # If StopTimes generated new trips from frequencies.txt, we need to add them
        # to the main trips table.

        return calendar, routes, None, stop_times, stops, trips
    
    def load_shapes(self,stops,stop_times):
        # --- 4. Load Shapes and Perform Advanced Time Interpolation ---
        trip_shape_ids_lf: pl.LazyFrame = (
            stop_times.generate_shape_ids().collect().lazy()
        )
        shapes = Shapes()
        shapes.load(None, trip_shape_ids_lf, stops.lf, check_files=False, min_file_id=0)
        return shapes, trip_shape_ids_lf
    
    def build_lf(self, calendar, routes, shapes, stop_times, stops, trips, trip_shape_ids_lf):
        # --- 5. Build the Main Integrated LazyFrame (`lf`) ---
        # Start with the core stop_times data.
        lf: pl.LazyFrame = stop_times.lf.select(
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
        if stop_times.frequencies is not None:
            lf = (
                lf.join(
                    stop_times.frequencies.select(
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
            lf = lf.with_columns(pl.lit(1, dtype=pl.UInt32).alias("n_trips"))

        # Merge with trips, stops, routes, and shapes data to create the full view.
        lf = lf.join(
            trips.lf.filter(~pl.col("next_day")).select(
                ["trip_id", "service_id", "route_id", "direction_id"]
            ),
            on="trip_id",
            how="left",
        )
        lf = lf.join(
            trip_shape_ids_lf.select(["trip_ids", "shape_id"])
            .explode("trip_ids")
            .rename({"trip_ids": "trip_id"}),
            on="trip_id",
            how="left",
        )

        lf = lf.join(
            stops.lf.select(["stop_id", "parent_station"]),
            on=["stop_id"],
            how="left",
        ).with_columns(
            pl.when(
                pl.col("parent_station").is_null()
            ).then(
                pl.col("stop_id")
            ).otherwise(
                pl.col("parent_station")
            ).alias("parent_station")
        )
        # Ensure that every trip does not stop twice at the parent_station

        # Sort by trip_id and stop_sequence
        lf = lf.sort(
            ["trip_id", "service_id", "route_id", "shape_id", "stop_sequence"]
        )

        # Create a new column with shifted parent_station per trip_id group
        lf = lf.with_columns(
            [
                pl.col("parent_station")
                .shift(1)
                .over("trip_id")
                .alias("prev_parent_station")
            ]
        )

        # Replace duplicate consecutive parent_station with None
        lf = lf.with_columns(
            [
                pl.when(pl.col("parent_station") == pl.col("prev_parent_station"))
                .then(None)
                .otherwise(pl.col("parent_station"))
                .alias("parent_station")
            ]
        )

        # Drop helper column if you don't want to keep it
        lf = lf.drop("prev_parent_station")

        lf = lf.join(
            routes.lf.select(["route_id", "route_type"]),
            on=["route_id"],
            how="left",
        )

        lf = lf.join(
            shapes.stop_shapes.select(
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
        if stop_times.fixed_times:
            lf = self.__fix_null_times(lf)

        lf = lf.drop("fixed_time")

        # For services running past midnight, create a unique "night" service_id
        # to distinguish them from the same service on the previous day.
        lf = lf.with_columns(
            [
                pl.when(pl.col("next_day"))
                .then(pl.concat_str(pl.col("service_id"), pl.lit("_night")))
                .otherwise(pl.col("service_id"))
                .alias("service_id")
            ]
        ).drop("next_day")

        lf = lf.unique()

        lf = lf.join(
            stops.lf.select(["stop_id"]),
            on="stop_id",
            how="left"
        )

        lf = lf.join(
            stops.lf.select(["stop_id"]),
            on="stop_id",
            how="left"
        )
        stop_ids_in_aoi_lf = pl.LazyFrame({'stop_id': self.stop_ids_in_aoi}).with_columns(pl.lit(True).alias("isin_aoi"))
        lf = lf.join(stop_ids_in_aoi_lf, on='stop_id', how="left")
        lf = lf.with_columns(pl.col("isin_aoi").fill_null(False))

        return lf

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

    def __service_intensity_in_date_range(self,gtfs_lf,route_types,date_df):
        gtfs_lf = gtfs_lf.filter(pl.col("isin_aoi"))
        if route_types is not None:
            gtfs_lf = self._filter_by_route_type(gtfs_lf, route_types=route_types)

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

        return total_by_date
    

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

    def _filter_by_date_range(
        self,
        data: pl.LazyFrame,
        start_date: datetime | date | None = None,
        end_date: datetime | date | None = None,
        date_type: str | list[str] | None = None,
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

    def _filter_by_date(
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

    def _filter_by_time_range(
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

    def _filter_by_route_type(
        self, data: pl.LazyFrame, route_types: list | int | str | None
    ) -> pl.LazyFrame:
        if (route_types == 'all') or (route_types is None):
            return data 
        
        if isinstance(route_types, list):
            if 'all' in route_types:
                return data 
            
            route_types = [gtfs_checker.normalize_route_type(i) for i in route_types]
        else:
            route_types = [gtfs_checker.normalize_route_type(route_types)]

        route_types_df = pl.DataFrame({"route_type": route_types})
        data = data.join(route_types_df.lazy(), on="route_type", how="semi")
        return data
    
    def _filter(
            self,
            data: pl.LazyFrame,
            start_date: datetime | date | None = None,
            end_date: datetime | date | None = None,
            date: datetime | date | None = None,
            date_type: str | list[str] | None = None,
            start_time: datetime | time = datetime.min,
            end_time: datetime | time = datetime.max,
            route_types: list | int | str | None = None,
            frequencies:bool = True,
            in_aoi:bool = False, 
            delete_last_stop:bool = False
        ):
        if in_aoi:
            data = data.filter(pl.col("isin_aoi"))

        if delete_last_stop:
            data = data.filter(pl.col("stop_sequence") != pl.col("stop_sequence").max().over("trip_id"))

        if route_types is not None:
            data = self._filter_by_route_type(data, route_types)

        if date is not None:
            data = self._filter_by_date(data, date)
        elif (start_date is not None) | (end_date is not None):
            data = self._filter_by_date_range(data, start_date,end_date,date_type)

        if (start_time is not None) | (end_time is not None):
            data = self._filter_by_time_range(data, start_time, end_time)

        if not frequencies:
            data = self._frequencies_to_stop_times(data)
            data = self._filter_by_time_range(data, start_time, end_time)

        data = data.collect().lazy()

        return data 
    
    def filter_by_date_range(
        self,
        start_date: datetime | date | None = None,
        end_date: datetime | date | None = None,
        date_type: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a date range.

        It uses the `Calendar` object to find all `service_id`s active within
        the specified date range and then semi-joins the input data with these
        service IDs.

        Args:
            start_date (datetime): The start of the date range (inclusive).
            end_date (datetime): The end of the date range (inclusive).

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """
        return self._filter_by_date_range(self.lf,start_date,end_date,date_type)

    def filter_by_date(
        self,
        date: datetime | date,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a date.

        It uses the `Calendar` object to find all `service_id`s active within
        the specified date range and then semi-joins the input data with these
        service IDs.

        Args:
            date (datetime): The desired date.

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """

        return self._filter_by_date(self.lf,date)

    def filter_by_time_range(
        self,
        start_time: datetime | time = datetime.min,
        end_time: datetime | time = datetime.max,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame based on a time-of-day range.

        It handles trips defined by `frequencies.txt` differently from those with
        explicit schedules.

        Args:
            start_time (datetime): The start of the time range. Defaults to 00:00:00.
            end_time (datetime): The end of the time range. Defaults to 23:59:59.

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """
        return self._filter_by_time_range(self.lf,start_time,end_time)

    def filter_by_route_type(
        self, route_types: list | int | str
    ) -> pl.LazyFrame:
        return self._filter_by_route_type(self.lf,route_types)

    def filter(
            self,
            start_date: datetime | date | None = None,
            end_date: datetime | date | None = None,
            date: datetime | date | None = None,
            date_type: str | list[str] | None = None,
            start_time: datetime | time = datetime.min,
            end_time: datetime | time = datetime.max,
            route_types: list | int | str | None = None,
            frequencies:bool = True,
            in_aoi:bool = False, 
            delete_last_stop:bool = False
        ):
        lf = self.lf
        return self._filter(
            lf,
            start_date,
            end_date,
            date,
            date_type,
            start_time,
            end_time,
            route_types,
            frequencies,
            in_aoi,
            delete_last_stop
        ) 
    
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

    def get_service_intensity_in_date_range(
        self,
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        date_type: Optional[str | list[str]] = None,
        route_types: Optional[str | int | list[str] | list[int]] = None,
        by_feed:bool=False
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

        if by_feed:
            ids = self.lf.select("file_id").unique().collect().to_series().to_list()
            total_by_date = []
            for id in ids:
                gtfs_lf = self.lf.filter(pl.col("file_id") == id)
                result = self.__service_intensity_in_date_range(gtfs_lf,route_types,date_df)
                result = result.with_columns(
                    pl.lit(id).alias("file_id")
                )
                total_by_date.append(result)

            total_by_date = pl.concat(total_by_date)
        else:
            gtfs_lf = self.lf
            total_by_date = self.__service_intensity_in_date_range(gtfs_lf,route_types,date_df)


        total_by_date = self.calendar.add_holidays_and_weekends(
            total_by_date, lon=self.stops.mean_lon, lat=self.stops.mean_lat
        )

        return total_by_date.sort("date")

    def _get_mean_interval_at_stops(
        self,
        lf: pl.LazyFrame,
        date: datetime | date | None,
        start_time: datetime | time = time.min,
        end_time: datetime | time = time.max,
        route_types: list[int] | int | str | None = None,
        by: str = "route_id",
        at: str = "parent_station",
        how: str = "all",
        n_divisions: int = 1,
        mix_directions:bool = False,
        frequencies:bool=False,
        in_aoi:bool=True,
        delete_last_stop:bool = True
    ) -> pl.LazyFrame:

        # --------------------
        # Base GTFS filtering
        # --------------------
        gtfs_lf = self._filter(
            lf,
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            in_aoi=in_aoi,
            frequencies=frequencies,
            delete_last_stop=delete_last_stop
        )
        gtfs_lf = gtfs_lf.filter(pl.col(at).is_not_null())

        start_sec = utils.time_to_seconds(start_time)
        end_sec = utils.time_to_seconds(end_time)


        # =====================================================================
        # CASE 1: Group by explicit route/direction (NOT shape-direction method)
        # =====================================================================
        if by != "shape_direction":

            # Collect columns required for grouping
            groupby = list(np.unique([at, by, "direction_id"]))

            # Schema validation
            missing = [col for col in groupby if col not in gtfs_lf.collect_schema().names()]
            if missing:
                raise Exception(f"Missing required columns {missing} in GTFS schema.")

            # ---- Compute per-route headway ----
            gtfs_lf = (
                gtfs_lf.sort("stop_sequence")
                .group_by(groupby)
                .agg(
                    pl.col("route_id").unique().alias("route_ids"),
                    pl.col("departure_time").sort().alias("departure_times"),
                    utils.mean_angle("shape_direction").alias("shape_direction"),
                    utils.mean_angle("shape_direction_backwards").alias("shape_direction_backwards"),
                    (
                        (pl.col("departure_time").min() - start_sec)
                        + (end_sec - pl.col("departure_time").max())
                    ).alias("initial_interval"),      
                )
            )

            gtfs_lf = (
                gtfs_lf.with_columns(
                    (
                        (
                            (pl.col("departure_times").list.diff(null_behavior="drop"))
                            .list.eval(pl.element().pow(2))
                            .list.sum()
                            + pl.col("initial_interval") ** 2
                        )
                        / (end_sec - start_sec)
                    ).alias("mean_interval")
                )
                .drop("initial_interval")
                .with_columns(
                    pl.col("direction_id").cast(int).alias("direction_id")
                )
                .collect()
                .lazy()
            )

            # ----------------------------------------------------------
            # HOW-aggregation for route-based method (grouped at bottom)
            # ----------------------------------------------------------
            if how == "best":
                gtfs_lf = gtfs_lf.group_by(at).agg(
                    [
                        pl.col("route_ids").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("route_ids"),
                        pl.col("shape_direction").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_direction"),
                        pl.col("direction_id").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().cast(int).alias("direction_id"),
                        pl.col("mean_interval").min().alias("mean_interval"),
                    ]
                )
            elif how == "add":
                if mix_directions == False:
                    # Pick best headway within each directional group
                    gtfs_lf = gtfs_lf.group_by(at,"direction_id").agg(
                        [
                            pl.col("route_ids").flatten().unique().alias("route_ids"),
                            (1 / (1 / pl.col("mean_interval")).sum()).alias("mean_interval"),
                            pl.col("shape_direction").flatten().unique().alias("shape_directions"),
                        ]
                    )
                    gtfs_lf = gtfs_lf.group_by(at).agg(
                        [
                            pl.col("route_ids").sort_by(
                                "mean_interval",  
                                nulls_last=True,
                                maintain_order=True
                            ).first().alias("route_ids"),
                            pl.col("shape_directions").sort_by(
                                "mean_interval",  
                                nulls_last=True,
                                maintain_order=True
                            ).first().alias("shape_directions"),
                            pl.col("direction_id").sort_by(
                                "mean_interval",  
                                nulls_last=True,
                                maintain_order=True
                            ).first().alias("direction_id"),
                            pl.col("mean_interval").min().alias("mean_interval"),
                        ]
                    )
                else:
                    gtfs_lf = gtfs_lf.group_by(at).agg(
                        [
                            pl.col("route_ids").flatten().unique().alias("route_ids"),
                            (1 / (1 / pl.col("mean_interval")).sum()).alias("mean_interval"),
                            pl.col("shape_direction").flatten().unique().alias("shape_directions"),
                            pl.col("direction_id")
                                .flatten()
                                .unique()
                                .drop_nulls()
                                .cast(int)
                                .alias("direction_ids"),
                        ]
                    )


            else:  # how == "all"
                gtfs_lf = gtfs_lf.drop("departure_times").with_columns(
                    pl.col("direction_id").cast(int).alias("direction_id")
                )

            gtfs_lf = gtfs_lf.with_columns(
                (pl.col("mean_interval") / 60).alias("mean_interval")
            )

            if by == "route_id":
                if how != "mean": 
                    gtfs_lf = gtfs_lf.rename({"route_ids":"route_id"})
                    gtfs_lf = gtfs_lf.with_columns(
                        pl.col("route_id").list.first().alias("route_id")
                    )
            else: 
                if how != "mean": 
                    gtfs_lf = gtfs_lf.rename({"route_ids":"route_id"})

            return gtfs_lf.collect()

        # =====================================================================
        # CASE 2: Shape-direction method (directional clustering)
        # =====================================================================
        # Number of angular bins (forward + backward)
        n_sectors = n_divisions * 2

        # Adjust shape directions to ensure proper forward/backward separation
        gtfs_lf = (
            gtfs_lf.with_columns(
                (
                    pl.when(
                        (
                            (
                                pl.col("shape_direction") + 360 - pl.col("shape_direction_backwards")
                            ) % 360
                        )
                        > (
                            (
                                pl.col("shape_direction_backwards") + 360 - pl.col("shape_direction")
                            ) % 360
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
                pl.when(pl.col("shape_diff").is_null() | pl.col("shape_diff").is_nan())
                .then(pl.lit(0))
                .otherwise(pl.col("shape_diff"))
                .alias("shape_diff")
            )
            .with_columns(
                ((pl.col("shape_direction") + 360 + pl.col("shape_diff")) % 360)
                .alias("shape_direction")
            )
            .drop("shape_diff")
        )

        # Compute direction split per stop
        gtfs_lf = gtfs_lf.group_by(at).agg(pl.all()).collect()
        gtfs_lf = gtfs_lf.with_columns(
            utils.max_separation_angle(gtfs_lf, "shape_direction").alias("shape_split_direction")
        )
        gtfs_lf = gtfs_lf.explode(pl.exclude([at, "shape_split_direction"]))
        gtfs_lf = gtfs_lf.lazy()

        # Offset for even number of divisions
        if n_divisions % 2 == 0:
            gtfs_lf = gtfs_lf.with_columns(pl.col("shape_split_direction") + 90 / n_divisions)

        gtfs_lf = gtfs_lf.with_columns(pl.col("shape_split_direction") % 360)

        # Assign angular bins
        gtfs_lf = (
            gtfs_lf.with_columns(
                (
                    (pl.col("shape_direction") - pl.col("shape_split_direction") + 360) % 360
                ).alias("angle")
            )
            .with_columns(
                ((pl.col("angle") * n_sectors / 360).floor().cast(pl.Int32))
                .alias("shape_direction_id")
            )
            .drop("angle")
        )

        # Compute headway per angular bin
        gtfs_lf = (
            gtfs_lf.sort("stop_sequence")
            .group_by([at, "shape_direction_id"])
            .agg(
                [
                    pl.col("departure_time").sort().alias("departure_times"),
                    utils.mean_angle("shape_direction").alias("shape_direction"),
                    pl.col("route_id").unique().alias("route_ids"),
                    pl.col("shape_id").unique().alias("shape_ids"),
                    (
                        (pl.col("departure_time").min() - start_sec)
                        + (end_sec - pl.col("departure_time").max())
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
                    / (end_sec - start_sec)
                ).alias("mean_interval"),
                (pl.col("shape_direction_id") % n_divisions).alias("shape_direction_group_id"),
            ]
        )

        # --------------------------------------------------------------
        # HOW-aggregation for shape-direction method (grouped at bottom)
        # --------------------------------------------------------------
        if how == "best":
            gtfs_lf = gtfs_lf.group_by(at).agg(
                [
                    pl.col("shape_direction").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("shape_direction"),
                    pl.col("shape_ids").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("shape_ids"),
                    pl.col("route_ids").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("route_ids"),
                    pl.col("mean_interval").min().alias("mean_interval"),
                ]
            )

        elif how == "add":
            if mix_directions == False:
                # Pick best headway within each directional group
                gtfs_lf = gtfs_lf.group_by([at, "shape_direction_group_id"]).agg(
                    [
                        pl.col("shape_direction").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_direction"),
                        pl.col("shape_ids").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_ids"),
                        pl.col("route_ids").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("route_ids"),
                        pl.col("mean_interval").min().alias("mean_interval"),
                    ]
                )

            gtfs_lf = gtfs_lf.group_by(at).agg(
                [
                    (1 / (1 / pl.col("mean_interval")).sum()).alias("mean_interval"),
                    pl.col("shape_direction").alias("shape_directions"),
                    pl.col("shape_ids").flatten().unique().alias("shape_ids"),
                    pl.col("route_ids").flatten().unique().alias("route_ids"),
                ]
            )
        else:
            gtfs_lf = gtfs_lf.with_columns(
                (pl.col("shape_direction_id") % 2).alias("shape_direction_id")
            )

        gtfs_lf = gtfs_lf.with_columns(
            (pl.col("mean_interval") / 60).alias("mean_interval")
        )
        return gtfs_lf.collect()

    def get_mean_interval_at_stops(
        self,
        date: datetime | date | None,
        start_time: datetime | time = time.min,
        end_time: datetime | time = time.max,
        route_types: list[int] | int | str | None = None,
        by: str = "route_id",
        at: str = "parent_station",
        how: str = "all",
        n_divisions: int = 1,
        mix_directions:bool = False,
    ) -> pl.LazyFrame:
        """
        Compute the mean headway (service interval) within a time window.

        Headway is computed using the harmonic mean of inter-departure intervals.
        Data may be aggregated by route or by directional clusters.

        Parameters
        ----------
        date : datetime | date
            Date for filtering service.
        start_time : datetime | time, default time.min
            Start of the analysis window.
        end_time : datetime | time, default time.max
            End of the analysis window.
        route_types : list[int] | int | str | None
            Filter by GTFS route types.
        by : str, {"route_id", "shape_direction"}
            Determines how services are grouped before headway computation.
        at : str, {"parent_station", "stop_id"}
            Spatial unit for the interval calculation.
        how : {"all", "best", "mean"}
            Post-aggregation method:
            - "all": return all route/direction combinations  
            - "best": pick the service with smallest interval  
            - "add": harmonic mean of all service intervals together (route intervals are added together)
        n_divisions : int, default 1
            Number of directional bins when using `shape_direction`.
        mix_directions : bool, default False 
            For how 'mean' mix outbound and inbound directions of same route as different routes

        Returns
        -------
        pl.LazyFrame
            A lazy frame containing headway metrics.

        Raises
        ------
        ValueError
            If unsupported combination of parameters is passed.
        """

        # --------------------
        # Base GTFS filtering
        # --------------------
        gtfs_lf = self.filter(
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            in_aoi=True,
            frequencies=False,
            delete_last_stop=True
        )
        gtfs_lf = gtfs_lf.filter(pl.col(at).is_not_null())

        return self._get_mean_interval_at_stops(
            gtfs_lf,
            date = date,
            start_time = start_time,
            end_time=end_time,
            route_types=route_types,
            by=by,
            at=at,
            how=how,
            n_divisions=n_divisions,
            mix_directions=mix_directions,
            frequencies=True,
            in_aoi=False,
            delete_last_stop = False,
        )
    
    def get_mean_speed_at_stops(
            self,
            date: datetime | date,
            start_time: datetime | time = time.min,
            end_time: datetime | time = time.max,
            route_types: list | int | str | None = None,
            by="route_id",
            at="parent_station",
            how="mean",
            direction="both",
            n_stops=1
        ):
        gtfs_lf = self.filter(
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            in_aoi=False,
            frequencies=True,
            delete_last_stop=False
        )
        gtfs_lf = gtfs_lf.filter(pl.col(at).is_not_null())

        if direction == "both":
            backwards_stops = n_stops 
            forward_stops = n_stops
        elif direction == "forward":
            backwards_stops = 0 
            forward_stops = n_stops
        elif direction == "backward":
            backwards_stops = n_stops 
            forward_stops = 0
        else:
            raise Exception(f"Direction {direction} not  implemented. Only 'both', 'forward' and 'backward' are valid.")

        gtfs_lf = (
            gtfs_lf
            .sort(["trip_id", "stop_sequence"])
            .with_columns(
                (
                    pl.col("shape_dist_traveled")
                        .shift(forward_stops)
                        .over("trip_id") -
                    pl.col("shape_dist_traveled")
                        .shift(-backwards_stops)
                        .over("trip_id")
                    
                ).alias("distance_weight"),
                (
                    pl.col("departure_time")
                        .shift(forward_stops)
                        .over("trip_id") - 
                    pl.col("departure_time")
                        .shift(-backwards_stops)
                        .over("trip_id")
                ).alias("time_weight")
            )
            .with_columns(
                pl.when(
                    (
                        pl.col("stop_sequence").max().over("trip_id") - 
                        pl.col("stop_sequence").min().over("trip_id") 
                    ) < forward_stops 
                ).then(
                    pl.col("shape_total_distance")
                ).otherwise(
                    pl.col("distance_weight")
                ).alias("distance_weight"),
                pl.when(
                    (
                        pl.col("stop_sequence").max().over("trip_id") - 
                        pl.col("stop_sequence").min().over("trip_id") 
                    ) < forward_stops 
                ).then(
                    pl.col("shape_total_travel_time")
                ).otherwise(
                    pl.col("time_weight")
                ).alias("time_weight")
            )
            .with_columns(
                pl.when(
                    (
                        pl.col("stop_sequence").max().over("trip_id") - 
                        pl.col("stop_sequence").min().over("trip_id") 
                    ) < backwards_stops 
                ).then(
                    pl.col("shape_total_distance")
                ).otherwise(
                    pl.col("distance_weight")
                ).alias("distance_weight"),
                pl.when(
                    (
                        pl.col("stop_sequence").max().over("trip_id") - 
                        pl.col("stop_sequence").min().over("trip_id") 
                    ) < backwards_stops
                ).then(
                    pl.col("shape_total_travel_time")
                ).otherwise(
                    pl.col("time_weight")
                ).alias("time_weight"),
            )
            .with_columns(
                pl.col("distance_weight").forward_fill().over("trip_id").alias("distance_weight"),
                pl.col("time_weight").forward_fill().over("trip_id").alias("time_weight"),
            )
            .with_columns(
                pl.col("distance_weight").backward_fill().over("trip_id").alias("distance_weight"),
                pl.col("time_weight").backward_fill().over("trip_id").alias("time_weight"),
            )
            .with_columns(
                (
                    (pl.col("distance_weight") / 1000) / (pl.col("time_weight") / 3600)
                ).alias("speed")
            )
        )

        if how == "max":
            gtfs_lf = gtfs_lf.with_columns(
                pl.col("speed").fill_null(float('-inf'))
            )
            gtfs_lf = gtfs_lf.group_by(list(np.unique([by,at]))).agg(
                pl.col("route_id").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last().alias("route_ids"),
                pl.col("speed").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col("distance_weight").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col("time_weight").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col("n_trips").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col("isin_aoi").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last().alias("isin_aoi")
            )
            gtfs_lf = gtfs_lf.with_columns(
                pl.when(pl.col("speed") == pl.lit(float('-inf')))
                .then(pl.lit(None))
                .otherwise(pl.col("speed"))
                .alias("speed")
            )
        elif how == "min":
            gtfs_lf = gtfs_lf.with_columns(
                pl.col("speed").fill_null(float('inf'))
            )
            gtfs_lf = gtfs_lf.group_by(list(np.unique([by,at]))).agg(
                pl.col("route_id").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first().alias("route_ids"),
                pl.col("speed").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col("distance_weight").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col("time_weight").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col("n_trips").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col("isin_aoi").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first().alias("isin_aoi")
            ) 
            gtfs_lf = gtfs_lf.with_columns(
                pl.when(pl.col("speed") == pl.lit(float('inf')))
                .then(pl.lit(None))
                .otherwise(pl.col("speed"))
                .alias("speed")
            )
        elif how == "mean":
            gtfs_lf = gtfs_lf.with_columns(
                pl.when(pl.col("speed").is_null())
                .then(pl.lit(0))
                .otherwise(pl.col("n_trips"))
                .alias("n_trips")
            )
            gtfs_lf = gtfs_lf.group_by(list(np.unique([by, at]))).agg(
                pl.col("route_id").unique().alias("route_ids"),
                (
                    (pl.col("distance_weight").abs() * pl.col("n_trips")).sum()
                    / pl.col("n_trips").sum()
                ).alias("distance_weight"),
                (
                    (pl.col("time_weight").abs() * pl.col("n_trips")).sum()
                    / pl.col("n_trips").sum()
                ).alias("time_weight"),
                pl.col("n_trips").sum().alias("n_trips"),
                pl.col("isin_aoi").any().alias("isin_aoi")
            ).with_columns(
                (
                    (pl.col("distance_weight") / 1000) / (pl.col("time_weight") / 3600)
                ).alias("speed")
            )
    
        if by == "route_id":
            if "route_ids" in gtfs_lf.collect_schema().names():
                gtfs_lf = gtfs_lf.drop("route_ids")
        else: 
            if how != "mean": 
                if "route_id" not in gtfs_lf.collect_schema().names():
                    gtfs_lf = gtfs_lf.rename({"route_ids":"route_id"})

        return gtfs_lf.filter(pl.col("isin_aoi") == True).drop("isin_aoi").collect()

    def get_mean_interval_at_edges(            
            self,
            date: datetime | date,
            start_time: datetime | time = time.min,
            end_time: datetime | time = time.max,
            route_types: list | int | str | None = None,
            by="edge_id",
            at="parent_station",
            how="mean",
            min_trips:int=2,
            mix_directions:bool=False,
        ):

        gtfs_lf = self.filter(
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            in_aoi=False,
            frequencies=False,
            delete_last_stop=False
        )
        gtfs_lf = gtfs_lf.filter(pl.col(at).is_not_null())

        start_sec = utils.time_to_seconds(start_time)
        end_sec = utils.time_to_seconds(end_time)

        gtfs_lf = (
            gtfs_lf
            .sort(["trip_id","stop_sequence"])
            .with_columns(
                pl.col("stop_id").alias("stop_id_A"),
                pl.col("stop_id").shift(1).over("trip_id").alias("stop_id_B"),
                pl.col("parent_station").alias("parent_station_A"),
                pl.col("parent_station").shift(1).over("trip_id").alias("parent_station_B"),
                
            )
            .filter(pl.col("stop_id_B").is_not_null())
            .select([
                "stop_id_A",
                "stop_id_B",
                "parent_station_A",
                "parent_station_B",
                "trip_id",
                "route_id",
                "route_type",
                "direction_id",
                "shape_id",
                "shape_direction",
                "shape_direction_backwards",
                "departure_time",
                "arrival_time",
                "stop_sequence",
                "n_trips",
            ])
            .with_columns(
                pl.when(pl.col(f"{at}_A") > pl.col(f"{at}_B")).then(
                    pl.concat_str([
                        pl.col(f"{at}_A").cast(pl.Utf8),
                        pl.lit("_stop_A_-_"),
                        pl.col(f"{at}_B").cast(pl.Utf8),
                        pl.lit("_stop_B")
                    ])
                ).otherwise(
                    pl.concat_str([
                        pl.col(f"{at}_B").cast(pl.Utf8),
                        pl.lit("_stop_A_-_"),
                        pl.col(f"{at}_A").cast(pl.Utf8),
                        pl.lit("_stop_B")
                    ])
                ).alias("edge_id"),
                pl.when(pl.col(f"{at}_A") > pl.col(f"{at}_B")).then(
                    pl.lit(0)
                ).otherwise(
                    pl.lit(1)
                ).alias("direction_id"),
            )
        )

        # Collect columns required for grouping
        groupby = list(np.unique(["edge_id", by, "direction_id"]))

        # Schema validation
        missing = [col for col in groupby if col not in gtfs_lf.collect_schema().names()]
        if missing:
            raise Exception(f"Missing required columns {missing} in GTFS schema.")

        # ---- Compute per-route headway ----
        gtfs_lf = (
            gtfs_lf.sort(["trip_id","stop_sequence"])
            .group_by(groupby)
            .agg(
                pl.col("route_id").unique().alias("route_ids"),
                pl.col("departure_time").sort().alias("departure_times"),
                utils.mean_angle("shape_direction").alias("shape_direction"),
                utils.mean_angle("shape_direction_backwards").alias("shape_direction_backwards"),
                (
                    (pl.col("departure_time").min() - start_sec)
                    + (end_sec - pl.col("departure_time").max())
                ).alias("initial_interval"),    
                pl.col(at+"_A").first(), 
                pl.col(at+"_B").first(),  
                pl.col("n_trips").sum().alias("n_trips"),
            )
        )

        gtfs_lf = (
            gtfs_lf.with_columns(
                (
                    (
                        (pl.col("departure_times").list.diff(null_behavior="drop"))
                        .list.eval(pl.element().pow(2))
                        .list.sum()
                        + pl.col("initial_interval") ** 2
                    )
                    / (end_sec - start_sec)
                ).alias("mean_interval")
            )
            .drop("initial_interval")
            .collect()
            .lazy()
        )

        if by == "edge_id":
            by = "route_id"
        # ----------------------------------------------------------
        # HOW-aggregation for route-based method (grouped at bottom)
        # ----------------------------------------------------------
        if how == "best":
            gtfs_lf = gtfs_lf.group_by("edge_id").agg(
                [
                    pl.col("route_ids").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("route_ids"),
                    pl.col("shape_direction").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("shape_direction"),
                    pl.col("mean_interval").min().alias("mean_interval"),
                    pl.col(at+"_A").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias(at+"_A"),
                    pl.col(at+"_B").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias(at+"_B"),
                    pl.col("direction_id").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("direction_id"),
                    pl.col("n_trips").sort_by(
                        "mean_interval",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("n_trips"),
                ]
            )

        elif how == "add":
            if mix_directions == False:
                # Pick best headway within each directional group
                gtfs_lf = gtfs_lf.group_by("edge_id","direction_id").agg(
                    [
                        pl.col("route_ids").flatten().unique().alias("route_ids"),
                        (1 / (1 / pl.col("mean_interval")).sum()).alias("mean_interval"),
                        pl.col("shape_direction").flatten().unique().alias("shape_directions"),
                        pl.col(at+"_A").first().alias(at+"_A"),
                        pl.col(at+"_B").first().alias(at+"_B"),
                        pl.col("n_trips").sum().alias("n_trips"),
                    ]
                )
                gtfs_lf = gtfs_lf.group_by("edge_id").agg(
                    [
                        pl.col("route_ids").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("route_ids"),
                        pl.col("shape_directions").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_directions"),
                        pl.col("mean_interval").min().alias("mean_interval"),
                        pl.col(at+"_A").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias(at+"_A"),
                        pl.col(at+"_B").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias(at+"_B"),
                        pl.col("direction_id").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("direction_id"),
                        pl.col("n_trips").sort_by(
                            "mean_interval",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("n_trips"),
                    ]
                )
            else:
                gtfs_lf = gtfs_lf.group_by("edge_id").agg(
                    [
                        pl.col("route_ids").flatten().unique().alias("route_ids"),
                        (1 / (1 / pl.col("mean_interval")).sum()).alias("mean_interval"),
                        pl.col("shape_direction").flatten().unique().alias("shape_directions"),
                        pl.col(at+"_A").first().alias(at+"_A"),
                        pl.col(at+"_B").first().alias(at+"_B"),
                        pl.col("direction_id").unique().alias("direction_id"),
                        pl.col("n_trips").sum().alias("n_trips"),
                    ]
                )


        else:  # how == "all"
            gtfs_lf = gtfs_lf.drop("departure_times")
            if by == "route_id":
                gtfs_lf = gtfs_lf.drop("route_ids")

        gtfs_lf = gtfs_lf.with_columns(
            (pl.col("mean_interval") / 60).alias("mean_interval")
        ).filter(pl.col("n_trips") > min_trips)

        return gtfs_lf.collect()


    def get_mean_speed_at_edges(            
            self,
            date: datetime | date,
            start_time: datetime | time = time.min,
            end_time: datetime | time = time.max,
            route_types: list | int | str | None = None,
            by="edge_id",
            at="parent_station",
            how="mean",
            min_trips:int=2,
        ):

        gtfs_lf = self.filter(
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            in_aoi=False,
            frequencies=True,
            delete_last_stop=False
        )
        gtfs_lf = gtfs_lf.filter(pl.col(at).is_not_null())
        gtfs_lf = (
            gtfs_lf
            .sort(["trip_id","stop_sequence"])
            .with_columns(
                pl.col("stop_id").alias("stop_id_A"),
                pl.col("stop_id").shift(1).over("trip_id").alias("stop_id_B"),
                pl.col("parent_station").alias("parent_station_A"),
                pl.col("parent_station").shift(1).over("trip_id").alias("parent_station_B"),
                (
                    pl.col("shape_dist_traveled").shift(1).over("trip_id")-pl.col("shape_dist_traveled")
                ).alias("distance_weight"),
                (
                    pl.col("departure_time").shift(1).over("trip_id")-pl.col("departure_time")
                ).alias("time_weight"),
                
            )
            .filter(pl.col("stop_id_B").is_not_null())
            .select([
                "stop_id_A",
                "stop_id_B",
                "parent_station_A",
                "parent_station_B",
                "trip_id",
                "route_id",
                "route_type",
                "direction_id",
                "shape_id",
                "shape_direction",
                "shape_direction_backwards",
                "departure_time",
                "arrival_time",
                "stop_sequence",
                "distance_weight",
                "time_weight",
                "n_trips",
            ])
            .with_columns(
                (
                    (pl.col("distance_weight") / 1000) / (pl.col("time_weight") / 3600)
                ).alias("speed"),
                pl.when(pl.col(f"{at}_A") > pl.col(f"{at}_B")).then(
                    pl.concat_str([
                        pl.col(f"{at}_A").cast(pl.Utf8),
                        pl.lit("_stop_A_-_"),
                        pl.col(f"{at}_B").cast(pl.Utf8),
                        pl.lit("_stop_B")
                    ])
                ).otherwise(
                    pl.concat_str([
                        pl.col(f"{at}_B").cast(pl.Utf8),
                        pl.lit("_stop_A_-_"),
                        pl.col(f"{at}_A").cast(pl.Utf8),
                        pl.lit("_stop_B")
                    ])
                ).alias("edge_id"),
                pl.when(pl.col(f"{at}_A") > pl.col(f"{at}_B")).then(
                    pl.lit(0)
                ).otherwise(
                    pl.lit(1)
                ).alias("direction_id"),
            )
        )

        if how == "max":
            gtfs_lf = gtfs_lf.with_columns(
                pl.col("speed").fill_null(float('-inf'))
            )
            gtfs_lf = gtfs_lf.group_by(list(np.unique([by,"edge_id"]))).agg(
                pl.col("route_id").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last().alias("route_ids"),
                pl.col("speed").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col("distance_weight").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col("time_weight").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),
                pl.col(at+"_A").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(), 
                pl.col(at+"_B").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last(),  
                pl.col("n_trips").sort_by(
                    "speed",  
                    nulls_last=False,
                    maintain_order=True
                ).last().alias("n_trips"),
            )
            gtfs_lf = gtfs_lf.with_columns(
                pl.when(pl.col("speed") == pl.lit(float('-inf')))
                .then(pl.lit(None))
                .otherwise(pl.col("speed"))
                .alias("speed")
            )
        elif how == "min":
            gtfs_lf = gtfs_lf.with_columns(
                pl.col("speed").fill_null(float('inf'))
            )
            gtfs_lf = gtfs_lf.group_by(list(np.unique([by,"edge_id"]))).agg(
                pl.col("route_id").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first().alias("route_ids"),
                pl.col("speed").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col("distance_weight").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col("time_weight").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(),
                pl.col(at+"_A").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(), 
                pl.col(at+"_B").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first(), 
                pl.col("n_trips").sort_by(
                    "speed",  
                    nulls_last=True,
                    maintain_order=True
                ).first().alias("n_trips"), 
            ) 
            gtfs_lf = gtfs_lf.with_columns(
                pl.when(pl.col("speed") == pl.lit(float('inf')))
                .then(pl.lit(None))
                .otherwise(pl.col("speed"))
                .alias("speed")
            )
        elif how == "mean":
            gtfs_lf = gtfs_lf.with_columns(
                pl.when(pl.col("speed").is_null())
                .then(pl.lit(0))
                .otherwise(pl.col("n_trips"))
                .alias("n_trips")
            )
            gtfs_lf = gtfs_lf.group_by(list(np.unique([by,"edge_id"]))).agg(
                pl.col("route_id").unique().alias("route_ids"),
                (
                    (pl.col("distance_weight").abs() * pl.col("n_trips")).sum()
                    / pl.col("n_trips").sum()
                ).alias("distance_weight"),
                (
                    (pl.col("time_weight").abs() * pl.col("n_trips")).sum()
                    / pl.col("n_trips").sum()
                ).alias("time_weight"),
                pl.col(at+"_A").first(), 
                pl.col(at+"_B").first(),  
                pl.col("n_trips").sum().alias("n_trips"),
            ).with_columns(
                (
                    (pl.col("distance_weight") / 1000) / (pl.col("time_weight") / 3600)
                ).alias("speed"),
            )

        if by == "route_id":
            if "route_ids" in gtfs_lf.collect_schema().names():
                gtfs_lf = gtfs_lf.drop("route_ids")
        else: 
            if how != "mean": 
                if "route_id" not in gtfs_lf.collect_schema().names():
                    gtfs_lf = gtfs_lf.rename({"route_ids":"route_id"})

        return gtfs_lf.collect()#.filter(pl.col("n_trips") > min_trips).collect()


    def add_stop_coords(self,df:pd.DataFrame|pl.DataFrame|pl.LazyFrame):
        if isinstance(df,pd.DataFrame):
            lf = pl.from_pandas(df).lazy()
        elif isinstance(df,pl.DataFrame):
            lf = df.lazy()
        else:
            lf = df

        column_priority = ['stop_id','parent_station','stop_id_A','parent_station_A']
        stop_column = None 
        edges = False
        for c in column_priority:
            if c in lf.collect_schema().names():
                stop_column = c 
                if stop_column.endswith("_A"):
                    edges = True
                    stop_column = stop_column.removesuffix("_A")
                break 
        
        if stop_column is None:
            warnings.warn(f"The provided dataframe should have one of the following columns {column_priority}.")
            return df
        
        if 'stop_name' in self.stops.lf.collect_schema().names():
            stops_lf = (
                self.stops.lf.select(["stop_id", "parent_station", "stop_lat", "stop_lon", "stop_name"])
            ) 
            if 'parent_station' == stop_column:
                    stops_lf = (
                        stops_lf
                        .with_columns(
                            pl.when(pl.col("parent_station") == pl.col('stop_id'))
                            .then(pl.col('stop_name'))
                            .otherwise(pl.lit(None))
                            .alias('_stop_name')
                        )
                        .with_columns(
                            pl.when(pl.col("_stop_name").is_null().over("parent_station").all())
                            .then(pl.col("stop_name"))
                            .otherwise(pl.col("_stop_name"))
                            .alias("stop_name")
                        ).drop("_stop_name")
                        .with_columns([
                            pl.col("stop_lat").mean().over("parent_station").alias("stop_lat"),
                            pl.col("stop_lon").mean().over("parent_station").alias("stop_lon"),
                            pl.col("stop_name").min().over("parent_station").alias("stop_name"),
                        ])
                        .drop("stop_id")
                    )
            else:
                stops_lf = stops_lf.drop("parent_station")

            stops_lf = stops_lf.unique(stop_column)
        else:
            stops_lf = (
                self.stops.lf.select([stop_column, "stop_lat", "stop_lon"])
            )
            if 'parent_station' == stop_column:
                stops_lf = (
                    stops_lf
                    .with_columns([
                        pl.col("stop_lat").mean().over(stop_column).alias("stop_lat"),
                        pl.col("stop_lon").mean().over(stop_column).alias("stop_lon"),
                        pl.lit(None).alias("stop_name")
                    ])
                )
            stops_lf = stops_lf.unique(stop_column)

        stops_lf = stops_lf.with_columns(
            pl.when(pl.col("stop_name").is_null())
            .then(pl.col(stop_column).str.replace(r"_file_\d+$", ""))
            .otherwise(
                pl.col("stop_name")
            )
            .alias("stop_name")
        )
        
        if edges:
            if 'edge_linestring' in lf.collect_schema().names():
                lf = lf.drop('edge_linestring')
            if 'stop_name_A' in lf.collect_schema().names():
                lf = lf.drop('stop_name_A')
            if 'stop_name_B' in lf.collect_schema().names():
                lf = lf.drop('stop_name_B')

            lf = lf.join(stops_lf.rename({stop_column:stop_column+"_A"}),on=stop_column+"_A",how='left')
            lf = lf.rename({"stop_lat":"stop_lat_A","stop_lon":"stop_lon_A","stop_name":"stop_name_A"})
            lf = lf.join(stops_lf.rename({stop_column:stop_column+"_B"}),on=stop_column+"_B",how='left')
            lf = lf.rename({"stop_lat":"stop_lat_B","stop_lon":"stop_lon_B","stop_name":"stop_name_B"})
            lf = lf.with_columns(
                pl.concat_str([
                    pl.lit("LINESTRING("),
                    pl.col("stop_lon_A").cast(str), pl.lit(" "),
                    pl.col("stop_lat_A").cast(str), pl.lit(", "),
                    pl.col("stop_lon_B").cast(str), pl.lit(" "),
                    pl.col("stop_lat_B").cast(str),
                    pl.lit(")")
                ]).alias("edge_linestring")
            ).drop(["stop_lon_A","stop_lat_A","stop_lon_B","stop_lat_B"])
        else:
            if 'stop_lat' in lf.collect_schema().names():
                lf = lf.drop('stop_lat')
            if 'stop_lon' in lf.collect_schema().names():
                lf = lf.drop('stop_lon')
            if 'stop_name' in lf.collect_schema().names():
                lf = lf.drop('stop_name')

            lf = lf.join(stops_lf,on=stop_column,how='left')

        if isinstance(df,pd.DataFrame):
            return lf.collect().to_pandas()
        elif isinstance(df,pl.DataFrame):
            return lf.collect()
        else:
            return lf.collect().lazy()


    def add_route_names(self,df:pd.DataFrame|pl.DataFrame|pl.LazyFrame):
        if isinstance(df,pd.DataFrame):
            lf = pl.from_pandas(df).lazy()
        elif isinstance(df,pl.DataFrame):
            lf = df.lazy()
        else:
            lf = df

        if ('route_id' not in lf.collect_schema().names()) and ('route_ids' not in lf.collect_schema().names()):
            warnings.warn(f"The provided dataframe should have the column 'route_id' or 'route_ids'")
            return df 
          
        if 'route_ids' in lf.collect_schema().names():
            lf = lf.with_row_index("_row_number")
            lf = lf.explode('route_ids')

        routes_lf = self.routes.lf.with_columns(
            pl.when(pl.col("route_short_name").is_not_null())
            .then(pl.col("route_short_name"))
            .when(pl.col("route_long_name").is_not_null())
            .then(pl.col("route_long_name"))
            .otherwise(
                # remove '_file_<digits>' from route_id
                pl.col("route_id").str.replace(r"_file_\d+$", "")
            )
            .alias("route_name")
        )

        routes_lf = routes_lf.select(['route_id','route_short_name','route_long_name','route_name','route_type','route_type_text'])
        if 'route_id' in lf.collect_schema().names():
            if 'route_short_name' in lf.collect_schema().names():
                lf = lf.drop('route_short_name')
            if 'route_long_name' in lf.collect_schema().names():
                lf = lf.drop('route_long_name')
            if 'route_name' in lf.collect_schema().names():
                lf = lf.drop('route_name')
            if 'route_type' in lf.collect_schema().names():
                lf = lf.drop('route_type')
            if 'route_type_text' in lf.collect_schema().names():
                lf = lf.drop('route_type_text')

            lf = lf.join(routes_lf,on='route_id',how='left')
        
        if 'route_ids' in lf.collect_schema().names():
            if 'route_short_names' in lf.collect_schema().names():
                lf = lf.drop('route_short_names')
            if 'route_long_names' in lf.collect_schema().names():
                lf = lf.drop('route_long_names')
            if 'route_names' in lf.collect_schema().names():
                lf = lf.drop('route_names')
            if 'route_types' in lf.collect_schema().names():
                lf = lf.drop('route_types')
            if 'route_type_texts' in lf.collect_schema().names():
                lf = lf.drop('route_type_texts')

            routes_lf = routes_lf.rename({
                'route_id':'route_ids',
                'route_short_name':'route_short_names',
                'route_long_name':'route_long_names',
                'route_name':'route_names',
                'route_type':'route_types',
                'route_type_text':'route_type_texts'
            })
            lf = lf.join(routes_lf,on='route_ids',how='left')
            lf = (
                lf.group_by("_row_number")
                .agg(
                    pl.exclude(["route_short_names", "route_long_names", "route_names", "route_ids", "route_types", "route_type_texts"]).first(),
                    pl.col("route_short_names").unique(),
                    pl.col("route_long_names").unique(),
                    pl.col("route_names").unique(),
                    pl.col("route_ids").unique(),
                    pl.col("route_types").unique(),
                    pl.col("route_type_texts").unique(),
                )
                .drop("_row_number")
            )

        if isinstance(df,pd.DataFrame):
            return lf.collect().to_pandas()
        elif isinstance(df,pl.DataFrame):
            return lf.collect()
        else:
            return lf.collect().lazy()