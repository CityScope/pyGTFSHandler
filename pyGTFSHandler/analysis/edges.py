# -*- coding: utf-8 -*-
"""Edge-level (stop-to-stop segment) headway/speed analysis methods.

Split out of `analysis/stops.py` to keep files within a readable size. Mixed
into `Feed` alongside `FeedAnalysisMixin` (`class Feed(FeedFilteringMixin,
FeedAnalysisMixin, FeedEdgeAnalysisMixin)`), so `self` is always a fully
constructed `Feed` instance. See `analysis/stops.py` for the per-stop
counterparts of these methods.
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


class FeedEdgeAnalysisMixin:
    """Mixin providing Feed.get_headway_at_edges/get_speed_at_edges."""

    def get_headway_at_edges(            
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

        start_sec = time_parsing.time_to_seconds(start_time)
        end_sec = time_parsing.time_to_seconds(end_time)

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
                geo_polars.mean_angle("shape_direction").alias("shape_direction"),
                geo_polars.mean_angle("shape_direction_backwards").alias("shape_direction_backwards"),
                (
                    (pl.col("departure_time").min() - start_sec)
                    + (end_sec - pl.col("departure_time").max())
                ).alias("initial_headway"),    
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
                        + pl.col("initial_headway") ** 2
                    )
                    / (end_sec - start_sec)
                ).alias("headway")
            )
            .drop("initial_headway")
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
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("route_ids"),
                    pl.col("shape_direction").sort_by(
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("shape_direction"),
                    pl.col("headway").min().alias("headway"),
                    pl.col(at+"_A").sort_by(
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias(at+"_A"),
                    pl.col(at+"_B").sort_by(
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias(at+"_B"),
                    pl.col("direction_id").sort_by(
                        "headway",  
                        nulls_last=True,
                        maintain_order=True
                    ).first().alias("direction_id"),
                    pl.col("n_trips").sort_by(
                        "headway",  
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
                        (1 / (1 / pl.col("headway")).sum()).alias("headway"),
                        pl.col("shape_direction").flatten().unique().alias("shape_directions"),
                        pl.col(at+"_A").first().alias(at+"_A"),
                        pl.col(at+"_B").first().alias(at+"_B"),
                        pl.col("n_trips").sum().alias("n_trips"),
                    ]
                )
                gtfs_lf = gtfs_lf.group_by("edge_id").agg(
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
                        pl.col("headway").min().alias("headway"),
                        pl.col(at+"_A").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias(at+"_A"),
                        pl.col(at+"_B").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias(at+"_B"),
                        pl.col("direction_id").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("direction_id"),
                        pl.col("n_trips").sort_by(
                            "headway",  
                            nulls_last=True,
                            maintain_order=True
                        ).first().alias("n_trips"),
                    ]
                )
            else:
                gtfs_lf = gtfs_lf.group_by("edge_id").agg(
                    [
                        pl.col("route_ids").flatten().unique().alias("route_ids"),
                        (1 / (1 / pl.col("headway")).sum()).alias("headway"),
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
            (pl.col("headway") / 60).alias("headway")
        ).filter(pl.col("n_trips") > min_trips)

        return gtfs_lf.collect()


    def get_speed_at_edges(            
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


