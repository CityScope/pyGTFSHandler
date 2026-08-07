# -*- coding: utf-8 -*-
"""Per-stop service-intensity/headway/speed analysis methods.

Split out of feed.py (which had grown past ~2700 lines) to keep individual
files within a readable size, per the refactor plan's file-size discipline.
These methods are mixed into `Feed` via multiple inheritance in `feed.py`
(`class Feed(FeedFilteringMixin, FeedAnalysisMixin, FeedEdgeAnalysisMixin)`),
so `self` here is always a fully-constructed `Feed` instance with `self.lf`,
`self.calendar`, `self.stops`, `self.stop_times`, etc. already populated --
these are not usable as a standalone class. See `analysis/edges.py` for the
edge-level (stop-to-stop segment) counterparts of these methods.
"""

from datetime import datetime, time, date, timedelta
from typing import Optional, Union, List
import geopandas as gpd
import polars as pl
import pandas as pd
import warnings
import numpy as np

from ..utils import time_parsing
from ..utils import geo_polars
from ..utils import gtfs_checker
from ..utils import io

SECS_PER_DAY: int = 86400


class FeedAnalysisMixin:
    """Mixin providing Feed.get_service_intensity_in_date_range/get_headway_at_stops/get_speed_at_stops/get_headway_at_edges/get_speed_at_edges/add_stop_coords/add_route_names."""

    def _service_intensity_in_date_range(self,gtfs_lf,route_types,date_df):
        gtfs_lf = gtfs_lf.filter(pl.col("isin_aoi"))
        if route_types is not None:
            gtfs_lf = self._filter_by_route_type(gtfs_lf, route_types=route_types)

        if self.stop_times.frequencies is None:
            gtfs_lf = gtfs_lf.unique(
                ["service_id", "stop_id", "departure_time"]
            ).select("trip_id", "service_id", "n_trips")
        else:
            gtfs_lf = gtfs_lf.unique(
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
            ids = (
                self.lf
                .select(["file_id", "gtfs_name"])
                .unique("file_id")
                .collect()
            )
            total_by_date = []
            for id, name in ids.iter_rows():
                gtfs_lf = self.lf.filter(pl.col("file_id") == id)
                result = self._service_intensity_in_date_range(gtfs_lf,route_types,date_df)
                result = result.with_columns(
                    pl.lit(id).alias("file_id"),
                    pl.lit(name).alias("gtfs_name"),
                )
                total_by_date.append(result)

            total_by_date = pl.concat(total_by_date)
        else:
            gtfs_lf = self.lf
            total_by_date = self._service_intensity_in_date_range(gtfs_lf,route_types,date_df)


        total_by_date = self.calendar.add_holidays_and_weekends(
            total_by_date, lon=self.stops.mean_lon, lat=self.stops.mean_lat
        )

        return total_by_date.sort("date")

    def _get_headway_at_stops(
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
        direction_method: str = "both",
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
        gtfs_lf = gtfs_lf.with_columns(
            pl.col("departure_time").cast(pl.Float64,strict=False)
        ).filter(
            pl.col("departure_time").is_not_null() & 
            pl.col("departure_time").is_not_nan() & 
            pl.col("departure_time").is_finite()
        )
            
        start_sec = time_parsing.time_to_seconds(start_time)
        end_sec = time_parsing.time_to_seconds(end_time)


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
                    geo_polars.mean_angle("shape_direction").alias("shape_direction"),
                    geo_polars.mean_angle("shape_direction_backwards").alias("shape_direction_backwards"),
                    (
                        (pl.col("departure_time").min() - start_sec)
                        + (end_sec - pl.col("departure_time").max())
                    ).alias("initial_headway"),      
                )
            )

            gtfs_lf = (
                gtfs_lf
                .with_columns(
                    (
                        (
                            (pl.col("departure_times").list.diff(null_behavior="drop"))
                            .list.eval(pl.element().pow(2))
                            .list.sum()
                            + pl.col("initial_headway") ** 2
                        )
                        / (end_sec - start_sec)
                    ).alias("headway")
                )
                .drop("initial_headway")
                .with_columns(
                    pl.col("direction_id").cast(int).alias("direction_id") # Cast fails if none !!!!!!!!!!!!!!!!!!!
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
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("route_ids"),
                        pl.col("shape_direction").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_direction"),
                        pl.col("direction_id").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().cast(int).alias("direction_id"),
                        pl.col("headway").min().alias("headway"),
                    ]
                )
            elif how == "add":
                if mix_directions == False:
                    # Pick best headway within each directional group
                    gtfs_lf = gtfs_lf.group_by(at,"direction_id").agg(
                        [
                            pl.col("route_ids").flatten().unique().alias("route_ids"),
                            (1 / (1 / pl.col("headway")).sum()).alias("headway"),
                            pl.col("shape_direction").flatten().unique().alias("shape_directions"),
                        ]
                    )
                    gtfs_lf = gtfs_lf.group_by(at).agg(
                        [
                            pl.col("route_ids").sort_by(
                                "headway",  
                                nulls_last=True,
                                maintain_order=True
                            ).first().alias("route_ids"),
                            pl.col("shape_directions").sort_by(
                                "headway",  
                                nulls_last=True,
                                maintain_order=True
                            ).first().alias("shape_directions"),
                            pl.col("direction_id").sort_by(
                                "headway",  
                                nulls_last=True,
                                maintain_order=True
                            ).first().alias("direction_id"),
                            pl.col("headway").min().alias("headway"),
                        ]
                    )
                else:
                    gtfs_lf = gtfs_lf.group_by(at).agg(
                        [
                            pl.col("route_ids").flatten().unique().alias("route_ids"),
                            (1 / (1 / pl.col("headway")).sum()).alias("headway"),
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
                if by == "route_id":
                    # "route_id" (the group-by key) and "route_ids" (the
                    # aggregated column) would otherwise collide once
                    # "route_ids" is renamed to "route_id" below.
                    gtfs_lf = gtfs_lf.drop("route_id")

            gtfs_lf = gtfs_lf.with_columns(
                (pl.col("headway") / 60).alias("headway")
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

        # Force forward/backward exactly 180 degrees apart before clustering
        # -- only `shape_direction` (forward) is ever used downstream here
        # (the widest-gap split and every sector assignment work off it
        # alone), so `direction_method` only needs to decide what
        # `shape_direction` itself becomes; there's no separate output
        # column for backward to keep in sync, unlike
        # `models.shapes._reconcile_fwd_bwd` (which this mirrors
        # conceptually, for `direction_method="both"`).
        if direction_method == "forward":
            # Forward trusted as-is; whatever correction would be needed
            # is implicitly left for backward to absorb (moot here, since
            # backward isn't used downstream) -- so no adjustment at all.
            pass
        elif direction_method == "backward":
            # Backward trusted as-is; forward is fully replaced by its
            # antipode.
            gtfs_lf = gtfs_lf.with_columns(
                ((pl.col("shape_direction_backwards") + 180) % 360).alias("shape_direction")
            )
        else:
            if direction_method != "both":
                raise ValueError(f"direction_method must be 'both', 'forward', or 'backward', got {direction_method!r}")
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
            geo_polars.max_separation_angle(gtfs_lf, "shape_direction").alias("shape_split_direction")
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
                ((pl.col("angle") * n_sectors / 360).floor().cast(pl.Int32, strict=False))
                .alias("shape_direction_id")
            )
            .drop("angle")
            .collect()
        )

        null_count = gtfs_lf.select(pl.col("shape_direction_id").null_count()).item()

        if null_count > 0:
            warnings.warn(
                f"{null_count} rows have null shape_direction_id and will be dropped",
                RuntimeWarning,
            )

        gtfs_lf = (
            gtfs_lf
            .drop_nulls("shape_direction_id")
            .lazy()
        )

        # Compute headway per angular bin
        gtfs_lf = (
            gtfs_lf.sort("stop_sequence")
            .group_by([at, "shape_direction_id"])
            .agg(
                [
                    pl.col("departure_time").sort().alias("departure_times"),
                    geo_polars.mean_angle("shape_direction").alias("shape_direction"),
                    pl.col("route_id").unique().alias("route_ids"),
                    pl.col("shape_id").unique().alias("shape_ids"),
                    (
                        (pl.col("departure_time").min() - start_sec)
                        + (end_sec - pl.col("departure_time").max())
                    ).alias("initial_headway"),
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
                        + pl.col("initial_headway") ** 2
                    )
                    / (end_sec - start_sec)
                ).alias("headway"),
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
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("shape_direction"),
                    pl.col("shape_ids").sort_by(
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("shape_ids"),
                    pl.col("route_ids").sort_by(
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("route_ids"),
                    pl.col("headway").min().alias("headway"),
                ]
            )

        elif how == "add":
            if mix_directions == False:
                # Pick best headway within each directional group
                gtfs_lf = gtfs_lf.group_by([at, "shape_direction_group_id"]).agg(
                    [
                        pl.col("shape_direction").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_direction"),
                        pl.col("shape_ids").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("shape_ids"),
                        pl.col("route_ids").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("route_ids"),
                        pl.col("headway").min().alias("headway"),
                    ]
                )

            gtfs_lf = gtfs_lf.group_by(at).agg(
                [
                    (1 / (1 / pl.col("headway")).sum()).alias("headway"),
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
            (pl.col("headway") / 60).alias("headway")
        )
        return gtfs_lf.collect()

    def get_headway_at_stops(
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
        direction_method: str = "both",
    ) -> pl.LazyFrame:
        """
        Compute the mean headway (service headway) within a time window.

        Headway is computed using the harmonic mean of inter-departure headways.
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
            Spatial unit for the headway calculation.
        how : {"all", "best", "mean"}
            Post-aggregation method:
            - "all": return all route/direction combinations  
            - "best": pick the service with smallest headway  
            - "add": harmonic mean of all service headways together (route headways are added together)
        n_divisions : int, default 1
            Number of directional bins when using `shape_direction`.
        mix_directions : bool, default False
            For how 'mean' mix outbound and inbound directions of same route as different routes
        direction_method : {"both", "forward", "backward"}, default "both"
            Only used when `by="shape_direction"`: how the forward/backward
            bearing pair at each stop is reconciled before clustering.
            "both" splits the correction evenly between forward and
            backward (the default); "forward" trusts the forward bearing
            as-is; "backward" trusts the backward bearing as-is and
            replaces forward with its antipode. See
            `models.shapes._reconcile_fwd_bwd` for the equivalent used by
            `direction_id` assignment.

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

        return self._get_headway_at_stops(
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
            direction_method=direction_method,
            frequencies=True,
            in_aoi=False,
            delete_last_stop = False,
        )

    def _frequencies_to_departures_lean(self, gtfs_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Lightweight equivalent of `Feed._frequencies_to_stop_times` that only
        expands `frequencies.txt` rows into individual departure instants,
        without reconstructing the full stop_times column set (trip_id
        renumbering, n_trips, fixed_time bookkeeping, etc.).

        This is used by `get_headway_at_stops_no_expand` to avoid the cost of
        materializing a full per-departure stop_times row for every frequency
        instant just to compute headways, which only need `departure_time`
        (and `arrival_time`, kept in sync for downstream time-range
        re-filtering) per group. The expansion formula (aligned_start /
        int_ranges) is copied verbatim from `_frequencies_to_stop_times` so
        the resulting departure instants are numerically identical.
        """
        gtfs_lf = gtfs_lf.collect()
        frequencies_exist = (self.stop_times.frequencies is not None) and (
            gtfs_lf.select(
                (
                    (~pl.col("start_time").is_null()) & (~pl.col("start_time").is_nan())
                ).any()
            ).item()
        )
        gtfs_lf = gtfs_lf.lazy()

        if not frequencies_exist:
            return gtfs_lf

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
        )

        return gtfs_lf

    def get_headway_at_stops_no_expand(
        self,
        date: datetime | date | None,
        start_time: datetime | time = time.min,
        end_time: datetime | time = time.max,
        route_types: list[int] | int | str | None = None,
        by: str = "route_id",
        at: str = "parent_station",
        how: str = "all",
        n_divisions: int = 1,
        mix_directions: bool = False,
        direction_method: str = "both",
    ) -> pl.LazyFrame:
        """Equivalent to `get_headway_at_stops`, but avoids materializing a
        full per-departure stop_times row (via `Feed._frequencies_to_stop_times`)
        for every scheduled frequency-based departure.

        Instead of expanding `frequencies.txt` rows into complete stop_times
        rows carrying every GTFS column, this projects down to only the
        columns needed for the headway computation *before* expanding, then
        performs the same closed-form (`aligned_start` + `int_ranges`)
        per-row departure-instant generation used internally by the
        expansion-based path. The two paths are expected to produce
        numerically identical headway results.

        See `get_headway_at_stops` for parameter documentation.
        """
        gtfs_lf = self.filter(
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            in_aoi=True,
            frequencies=True,
            delete_last_stop=True,
        )
        gtfs_lf = gtfs_lf.filter(pl.col(at).is_not_null())

        needed_cols = list(
            np.unique(
                [
                    at,
                    by,
                    "direction_id",
                    "route_id",
                    "shape_id",
                    "departure_time",
                    "arrival_time",
                    "start_time",
                    "end_time",
                    "headway_secs",
                    "shape_time_traveled",
                    "shape_direction",
                    "shape_direction_backwards",
                    "stop_sequence",
                    "service_id",
                    "day_offset",
                    "route_type",
                ]
            )
        )
        schema_names = gtfs_lf.collect_schema().names()
        needed_cols = [c for c in needed_cols if c in schema_names]
        gtfs_lf = gtfs_lf.select(needed_cols)

        gtfs_lf = self._frequencies_to_departures_lean(gtfs_lf)

        return self._get_headway_at_stops(
            gtfs_lf,
            date=date,
            start_time=start_time,
            end_time=end_time,
            route_types=route_types,
            by=by,
            at=at,
            how=how,
            n_divisions=n_divisions,
            mix_directions=mix_directions,
            direction_method=direction_method,
            frequencies=True,
            in_aoi=False,
            delete_last_stop=False,
        )

    def get_speed_at_stops(
            self,
            date: datetime | date,
            start_time: datetime | time = time.min,
            end_time: datetime | time = time.max,
            route_types: list | int | str | None = None,
            by: str = "route_id",
            at: str = "parent_station",
            how: str = "mean",
            direction: str = "both",
            time_step: int = 15
        ) -> pl.DataFrame:
        """Computes average travel speed of trips passing through each stop.

        Filters the feed to `date`/`start_time`/`end_time`/`route_types`,
        derives per-trip speed from consecutive stop distances/times, and
        aggregates it per stop.

        Args:
            date: Service date to evaluate.
            start_time: Start of the time window (default midnight).
            end_time: End of the time window (default end of day).
            route_types: Optional route type filter.
            by: Grouping key for aggregating trips into a "line" (e.g.
                `"route_id"` or `"shape_direction"`).
            at: Which stop identity column to report (e.g.
                `"parent_station"` vs raw `"stop_id"`).
            how: Aggregation strategy across trips at a stop (`"mean"`,
                `"max"`, ...).
            direction: Which travel direction(s) to include (`"both"`,
                `"forward"`, `"backward"`).
            time_step: Bucket size in minutes used for intermediate
                time-binning of speed samples.

        Returns:
            pl.DataFrame: One row per stop (grouped as requested), with a
            `speed` column.
        """
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
            forward = time_parsing.time_displacement(gtfs_lf,secs_disp=time_step*60)
            forward = forward.rename({'time_weight':'time_weight_f','distance_weight':'distance_weight_f'})
            forward = forward.with_columns(
                pl.col("time_weight_f").fill_null(0),
                pl.col("distance_weight_f").fill_null(0),
            )
            backward = time_parsing.time_displacement(gtfs_lf,secs_disp=-time_step*60)
            backward = backward.rename({'time_weight':'time_weight_b','distance_weight':'distance_weight_b'})
            backward = backward.with_columns(
                pl.col("time_weight_b").fill_null(0),
                pl.col("distance_weight_b").fill_null(0),
            )
            gtfs_lf = gtfs_lf.join(forward.select(['trip_id','stop_id','time_weight_f','distance_weight_f']),on=['trip_id','stop_id'],how='left')
            gtfs_lf = gtfs_lf.join(backward.select(['trip_id','stop_id','time_weight_b','distance_weight_b']),on=['trip_id','stop_id'],how='left')
            gtfs_lf = gtfs_lf.with_columns(
                (pl.col("time_weight_f") + pl.col("time_weight_b")).alias("time_weight"),
                (pl.col("distance_weight_f") + pl.col("distance_weight_b")).alias("distance_weight"),
            )
        elif direction == "forward":
            forward = time_parsing.time_displacement(gtfs_lf,secs_disp=time_step*60)
            forward = forward.with_columns(
                pl.col("time_weight").fill_null(0),
                pl.col("distance_weight").fill_null(0),
            )
            gtfs_lf = gtfs_lf.join(forward.select(['trip_id','stop_id','time_weight','distance_weight']),on=['trip_id','stop_id'],how='left')
        elif direction == "backward":
            backward = time_parsing.time_displacement(gtfs_lf,secs_disp=-time_step*60)
            backward = backward.with_columns(
                pl.col("time_weight").fill_null(0),
                pl.col("distance_weight").fill_null(0),
            )
            gtfs_lf = gtfs_lf.join(backward.select(['trip_id','stop_id','time_weight','distance_weight']),on=['trip_id','stop_id'],how='left')
        else:
            raise Exception(f"Direction {direction} not  implemented. Only 'both', 'forward' and 'backward' are valid.")

        gtfs_lf = gtfs_lf.with_columns(
                (
                    (pl.col("distance_weight") / 1000) / (pl.col("time_weight") / 3600)
                ).alias("speed")
            ).with_columns(
                pl.when(
                    pl.col("speed").is_infinite()
                ).then(
                    None
                ).otherwise(
                    pl.col("speed")
                ).alias("speed"),  
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

        gtfs_lf = gtfs_lf.with_columns(
            pl.when(pl.col("speed").is_infinite() | pl.col("speed").is_nan())
            .then(pl.lit(None))
            .otherwise(pl.col("speed"))
            .alias("speed")
        )
        
        gtfs_lf = gtfs_lf.with_columns(
            pl.when(
                pl.col("speed").is_null()
            ).then(pl.lit(None)).otherwise(
                pl.col("time_weight")).alias("time_weight"),
            pl.when(
                pl.col("speed").is_null()
            ).then(pl.lit(None)).otherwise(
                pl.col("distance_weight")).alias("distance_weight")
        )
        
        return gtfs_lf.filter(pl.col("isin_aoi") == True).drop("isin_aoi").collect()

    def add_stop_coords(self,df:pd.DataFrame|pl.DataFrame|pl.LazyFrame) -> pd.DataFrame|pl.DataFrame|pl.LazyFrame:
        """Joins `stop_lat`/`stop_lon` (and `stop_name`) onto a result DataFrame.

        Looks for a stop-identity column in `df` (checked in priority order:
        `stop_id`, `parent_station`, `stop_id_A`, `parent_station_A`, the
        last two being edge-result columns) and left-joins the matching
        coordinates from `self.stops`.

        Args:
            df: A pandas/polars DataFrame or polars LazyFrame produced by
                one of this class's `get_*_at_stops`/`get_*_at_edges`
                methods (or any frame containing one of the recognized
                stop-id columns).

        Returns:
            Same type as `df`, with stop coordinate columns added. Returned
            unchanged (with a warning) if no recognized stop-id column is
            found.
        """
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


    def add_route_names(self,df:pd.DataFrame|pl.DataFrame|pl.LazyFrame) -> pd.DataFrame|pl.DataFrame|pl.LazyFrame:
        """Joins route names (e.g. `route_short_name`) onto a result DataFrame.

        Looks for `route_id` or `route_ids` (list column, from grouped
        results) in `df` and left-joins the matching name(s) from
        `self.routes`.

        Args:
            df: A pandas/polars DataFrame or polars LazyFrame containing a
                `route_id` or `route_ids` column.

        Returns:
            Same type as `df`, with route name column(s) added. Returned
            unchanged (with a warning) if neither `route_id` nor
            `route_ids` is present.
        """
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

        routes_lf = self.routes.lf
        if routes_lf is None: 
            lf = lf.with_columns(
                pl.lit(None).alias("route_short_name"),
                pl.lit(None).alias("route_long_name"),
                pl.lit(None).alias("route_name"),
                pl.lit(None).alias("route_type"),
                pl.lit(None).alias("route_type_text")
            )
        else:
            if "route_short_name" not in routes_lf.collect_schema().names():
                routes_lf = routes_lf.with_columns(
                    pl.lit(None).alias("route_short_name")
                )

            if "route_long_name" not in routes_lf.collect_schema().names():
                routes_lf = routes_lf.with_columns(
                    pl.lit(None).alias("route_long_name")
                )

            if "route_name" not in routes_lf.collect_schema().names():
                routes_lf = routes_lf.with_columns(
                    pl.lit(None).alias("route_name")
                )

            routes_lf = routes_lf.with_columns(
                pl.when(pl.col("route_short_name").is_not_null())
                .then(pl.col("route_short_name"))
                .when(pl.col("route_long_name").is_not_null())
                .then(pl.col("route_long_name"))
                .when(pl.col("route_name").is_not_null())
                .then(pl.col("route_name"))
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

            if routes_lf is not None:
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

