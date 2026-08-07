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
    -   Calculating the mean headway (headway) between services at stops.
    -   Filtering the integrated data by specific date or time ranges.

The final result is a powerful `Feed` object that holds a clean, integrated, and
analysis-ready representation of the entire GTFS schedule.


Columns that might have duplicates between gtfs files: stop_id, route_id
Columns that autoresolve duplicates between gtfs files: trip_id, service_id, shape_id
TODOs:
- TODO when grouping by parent_station shape_direction should be the direction with the most remaining stops and the oposite for
the direction with less remaining stops
- TODO avoid performing non necesary checks at the begining. Do first all the filters and then the checks.
- TODO check that all trips have a route and if not generate it.
- TODO check that trips from stop_times have a trip in trips and same for stops
- TODO add direction_id to routes based on shape direction per shape_id
- TODO finish dealing with shape_id and shapes.txt
- TODO If in a trip_id a stop_id is repeated divide it into 2 trip ids check if this is really needed
- TODO Check service intensity results. Sometimes (Valdemoro, Madrid) there are strange peaks in the plot. 
- TODO Computing speed if service passes to next day the first stops of the next day are needed. Otherwise there are trips with one stop as the rest happens the next day.
- TODO What about services with different timezones?
"""

from __future__ import annotations

"""TODO: revise filter_by_time_range with frequencies
TODO: revise headway func to work with all possible by, at and hows especially 'shape_direction' 'all' instead of 'max'
"""

from .models import StopTimes, Stops, Trips, Calendar, Routes, Shapes
from .utils import gtfs_checker
from .utils import io

from pathlib import Path
from datetime import datetime, time, date, timedelta
from typing import Optional, Union, List
import geopandas as gpd
import polars as pl
import pandas as pd 
import warnings 
import numpy as np 

SECS_PER_DAY: int = 86400


def concat_feeds(
    feeds: "Feed | list[Feed]",
    stop_group_distance: float = 0
) -> "Feed":
    """Concatenate several already-loaded `Feed` objects into one.

    Combines the `calendar`, `routes`, `stop_times`, `stops` and `trips`
    components of every feed in `feeds` by concatenating their underlying
    Polars/GeoPandas frames, then rebuilds shapes (`load_shapes`) and the
    integrated `lf` (`build_lf`) on top of the combined data. Useful for
    merging multiple regional GTFS feeds (e.g. one `Feed` per operator)
    into a single queryable object.

    Args:
        feeds: A single `Feed`, or a list of `Feed` objects to merge. If a
            single `Feed` is passed it is returned unchanged.
        stop_group_distance: If greater than 0, stops within this distance
            (in the stops' CRS units) are clustered together after
            concatenation via `Stops.group_stops`.

    Returns:
        Feed: The first feed in `feeds`, mutated in place so its
        `calendar`, `routes`, `stop_times`, `stops`, `trips`, `shapes`,
        `trip_shape_ids_lf` and `lf` attributes reflect the combined data
        of every feed passed in.
    """
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

    result.shapes, result.trip_shape_ids_lf = result.load_shapes(result.stops,result.stop_times,result.trips,result.gtfs_dir)
 
    result.lf = result.build_lf(result.calendar, result.routes, result.shapes, result.stop_times, result.stops, result.trips, result.trip_shape_ids_lf)

    return result

from .analysis.filtering import FeedFilteringMixin
from .analysis.stops import FeedAnalysisMixin
from .analysis.edges import FeedEdgeAnalysisMixin


class Feed(FeedFilteringMixin, FeedAnalysisMixin, FeedEdgeAnalysisMixin):
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
        min_file_id: int = 0,
        load_shapes: bool = True,
    ):
        """Loads, filters and integrates one or more GTFS directories into a `Feed`.

        This is the main entry point of the library: pass one or more
        uncompressed GTFS directories (or zip files) and get back a `Feed`
        whose `self.lf` is a single denormalized LazyFrame joining stops,
        stop_times, trips, routes, calendar and shapes. Internally this
        just calls `self.load(...)` (component loading/filtering), then
        `self.load_shapes(...)` and `self.build_lf(...)` -- see those
        methods' docstrings for the detailed pipeline.

        Args:
            gtfs_dirs: One or more paths to GTFS directories or `.zip` files.
            aoi: Optional Area of Interest (GeoDataFrame/GeoSeries) used to
                geospatially filter stops.
            stop_group_distance: If greater than 0, cluster stops within
                this distance of each other under a shared `parent_station`.
            start_date: Inclusive lower bound on service dates to load.
            end_date: Inclusive upper bound on service dates to load.
            date_type: Optional weekday/weekend/holiday classification
                filter (see `Calendar.filter_by_date_type`).
            start_time: Lower bound of the daily time window to load.
            end_time: Upper bound of the daily time window to load.
            route_types: Optional route type filter (names, codes, or a
                mix); `None`/`"all"` keeps every route type.
            service_ids: Optional explicit `service_id` allow-list.
            trip_ids: Optional explicit `trip_id` allow-list.
            stop_ids: Optional explicit `stop_id` allow-list.
            route_ids: Optional explicit `route_id` allow-list.
            check_files: If `True` (default), validate each GTFS file's
                schema/mandatory columns via `utils.gtfs_checker` while loading.
            min_file_id: Starting index used to tag rows with which source
                `gtfs_dirs` entry they came from (`file_id` column); relevant
                mainly when merging feeds with `concat_feeds`.
            load_shapes: If `True` (default), read real `shapes.txt`
                geometry where available; if `False`, always use
                straight-line stop-to-stop shapes (faster, see
                `load_shapes`'s docstring).

        Raises:
            ValueError: If any of `gtfs_dirs` is not a valid directory/zip.
            Exception: If the combination of filters leaves no stops,
                routes, services, trips, or stop_times.
        """
        self.calendar, self.routes, self.gtfs_dir, self.stop_times, self.stops, self.trips = self.load(
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

        self.shapes, self.trip_shape_ids_lf = self.load_shapes(
            self.stops, self.stop_times, self.trips, self.gtfs_dir, use_real_shapes=load_shapes
        )

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
                gtfs_dir[i] = Path(io.unzip(gtfs_dir[i],delete=False))
                warnings.warn(f"Extracting {orig_file} to {gtfs_dir[i]}")

            if not gtfs_dir[i].is_dir():
                raise ValueError(f"{gtfs_dir[i]} is not a valid directory.")

        # Loading the exact same resolved directory twice (e.g. the caller
        # passed a duplicate path, or two different-looking paths that
        # resolve to the same place via symlinks/relative segments) must be
        # idempotent, not duplicate every stop/trip/route -- each source
        # directory should only ever be read once.
        seen_resolved_paths: set = set()
        deduplicated_gtfs_dir: List[Path] = []
        for directory in gtfs_dir:
            resolved = directory.resolve()
            if resolved in seen_resolved_paths:
                warnings.warn(f"{directory} was passed more than once; loading it only once.")
                continue
            seen_resolved_paths.add(resolved)
            deduplicated_gtfs_dir.append(directory)
        gtfs_dir = deduplicated_gtfs_dir

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

        return calendar, routes, gtfs_dir, stop_times, stops, trips

    def load_shapes(self, stops, stop_times, trips, gtfs_dir, use_real_shapes: bool = True):
        """Builds `self.shapes`/`self.trip_shape_ids_lf`.

        Args:
            use_real_shapes: When `True` (default), real polyline geometry
                is read from `shapes.txt` (if present) and each synthetic
                shape's stops are inserted into it at their nearest-segment
                position (see `Shapes._insert_stops_into_real_shapes`).
                When `False`, `shapes.txt` is never read and every shape is
                just a straight line stop-to-stop -- skips the real-geometry
                matching entirely, which is the most expensive part of
                loading a feed with a large `shapes.txt` (e.g. detailed
                rail/metro polylines), for callers who don't need real
                geometry and want faster loads.
        """
        # --- 4. Load Shapes and Perform Advanced Time Interpolation ---
        trip_shape_ids_lf: pl.LazyFrame = (
            stop_times.generate_shape_ids().collect().lazy()
        )

        # The synthetic `shape_id` generated above groups trips purely by
        # identical stop sequence + travel time -- it has no relationship to
        # the real `shape_id` a feed's `trips.txt` may reference (which is
        # what `shapes.txt`'s actual polyline points are keyed by). Look up,
        # for each synthetic group, which real `shape_id` its member trips
        # actually use (if any and if consistent), so `Shapes` can fetch the
        # real geometry for that group instead of always falling back to a
        # straight line between stops.
        trip_to_real_shape_id = trips.lf.select("trip_id", "shape_id").rename(
            {"shape_id": "real_shape_id"}
        )
        real_shape_id_per_group = (
            trip_shape_ids_lf.select(["shape_id", "trip_ids"])
            .explode("trip_ids")
            .rename({"trip_ids": "trip_id"})
            .join(trip_to_real_shape_id, on="trip_id", how="left")
            .filter(pl.col("real_shape_id").is_not_null() & (pl.col("real_shape_id") != ""))
            .group_by("shape_id")
            # `.mode()` can return more than one value when two or more
            # `real_shape_id`s are tied for most-frequent within a
            # synthetic group, in whatever order polars' internal
            # (randomized-per-process) hashing happens to produce them --
            # `.first()` alone would then pick a different winner on every
            # run, attaching a different real geometry (and so a different
            # `direction_id`/`direction_conflict` outcome) to the same
            # synthetic shape from run to run on identical input. Sorting
            # first makes the tie-break deterministic (alphabetically
            # smallest `real_shape_id` wins ties).
            .agg(pl.col("real_shape_id").mode().sort().first().alias("real_shape_id"))
        )
        trip_shape_ids_lf = trip_shape_ids_lf.join(real_shape_id_per_group, on="shape_id", how="left")

        shapes = Shapes()
        shapes_path = gtfs_dir if use_real_shapes else None
        shapes.load(shapes_path, trip_shape_ids_lf, stops.lf, check_files=False, min_file_id=0)
        # `stops`/`stop_times`/`trips` here are already the post-filter (date,
        # time window, AOI, route_types, service/trip/stop/route id) versions
        # produced by `self.load(...)` above, so this direction_id assignment
        # -- and the warning it may emit -- only ever reflects the feed's
        # actually-in-scope shapes/trips, not the full unfiltered source data.
        shapes.assign_direction_ids(trip_shape_ids_lf, trips.lf, stops.lf)
        return shapes, trip_shape_ids_lf
    
    def build_lf(
        self,
        calendar: Calendar,
        routes: Optional[Routes],
        shapes: Shapes,
        stop_times: StopTimes,
        stops: Stops,
        trips: Trips,
        trip_shape_ids_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Joins the loaded components into the single integrated `self.lf`.

        Starts from `stop_times.lf`, left-joins frequencies (if present),
        trips (`service_id`, `route_id`, `direction_id`), shape ids,
        `parent_station` (deduplicating consecutive stops at the same
        parent station within a trip), route metadata, and per-stop
        shape geometry fields (`shape_dist_traveled`, `shape_direction`,
        ...). If `stop_times.fixed_times` is set, also runs the
        shape-distance-based interpolation (`_fix_null_times`) for any
        still-missing arrival/departure times. Finally flags each row with
        `isin_aoi` based on the AOI-filtered stop set computed in `load`.

        Args:
            calendar: Loaded `Calendar` instance.
            routes: Loaded `Routes` instance (or `None`).
            shapes: Loaded `Shapes` instance.
            stop_times: Loaded `StopTimes` instance.
            stops: Loaded `Stops` instance.
            trips: Loaded `Trips` instance.
            trip_shape_ids_lf: Mapping from synthetic `shape_id` to member
                `trip_id`s, as produced by `load_shapes`.

        Returns:
            pl.LazyFrame: The denormalized schedule, one row per
            (trip, stop) pair, assigned to `self.lf`.
        """
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
                "day_offset",
                "time_of_day",
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
            trips.lf.select(
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

        if routes.lf is not None:
            lf = lf.join(
                routes.lf.select(["route_id", "route_type"]),
                on=["route_id"],
                how="left",
            )
            lf = lf.with_columns(
                pl.when(
                    pl.col("route_id").is_null()
                ).then(
                    pl.col("trip_id")
                ).otherwise(
                    pl.col("route_id")
                ).alias("route_id")
            )
        else:
            lf = lf.with_columns(
                pl.col("route_id").alias("trip_id"),
                pl.lit(-1).alias("route_type")
            )

        lf = lf.with_columns(
            pl.col("route_type").fill_null(-1)
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
            lf = self._fix_null_times(lf)

        lf = lf.drop("fixed_time")

        # `service_id` is kept exactly as authored in the GTFS feed. Which
        # real calendar date a stop_time belongs to is `service_date +
        # day_offset` (resolved at query time by `filter_by_date`/
        # `filter_by_date_range`/`get_service_intensity_in_date_range`), not
        # baked into the id itself the way the old `"_night"` suffix scheme
        # did (which only supported a single day of offset and duplicated
        # every service/trip regardless of whether it ever ran past midnight).
        lf = lf.drop("next_day")

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

    def _fix_null_times(self, stop_times: pl.LazyFrame) -> pl.LazyFrame:
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

        # The midnight-crossing branch above (`dep_time_bwd < dep_time_fwd`)
        # computes the interpolated value in an absolute-seconds space (by
        # adding a full day to `dep_time_bwd`), which can land at or past
        # `SECS_PER_DAY` -- fold it back into the 0-24h `time_of_day`
        # convention that `departure_time`/`arrival_time` use everywhere
        # else. `day_offset` itself needs no change: it was already resolved
        # correctly by `StopTimes.__normalize_times`'s coarse first pass, and
        # this refinement only moves the time-of-day within the same day.
        stop_times = stop_times.with_columns(
            (pl.col("departure_time") % SECS_PER_DAY).alias("departure_time"),
            (pl.col("arrival_time") % SECS_PER_DAY).alias("arrival_time"),
        ).with_columns(
            pl.when(pl.col("fixed_time"))
            .then(pl.col("departure_time"))
            .otherwise(pl.col("time_of_day"))
            .alias("time_of_day")
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
                            pl.col("end_time") + pl.col("shape_time_traveled"),
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

