import polars as pl
import os
from pyGTFSHandler.utils import read_csv_lazy


class StopTimes:
    def __init__(
        self, feed, trip_ids: None | list[str] = None, stop_ids: None | list[str] = None
    ):
        self.feed = feed
        self.lf = self._read_stop_times(trip_ids, stop_ids)
        # self.lf: pl.LazyFrame = load_lazyframe_from_file(dir, "stop_times.txt")

    def _read_stop_times(
        self, trip_ids: None | list[str] = None, stop_ids: None | list[str] = None
    ):
        # TODO: Add the rest of the columns we could encounter in stop_times.txt
        stop_times_list = [
            read_csv_lazy(
                os.path.join(self.feed.gtfs_dir[j], "stop_times.txt"),
                schema_overrides={
                    "trip_id": str,
                    "arrival_time": str,
                    "departure_time": str,
                    "stop_id": str,
                    "shape_dist_traveled": str,
                    "stop_sequence": int,
                },
            ).with_columns(file_number=j)
            for j in range(len(self.feed.gtfs_dir))
        ]
        stop_times: pl.LazyFrame = pl.concat(stop_times_list, how="diagonal_relaxed")

        stop_times = stop_times.filter(
            ~pl.col("stop_id").is_null() & ~pl.col("trip_id").is_null()
        )

        # TODO: Review this, should I check first if it is repeated?
        # TODO: Check if trip_ids are reapetead, if they are, do this and raise a Warning
        stop_times = stop_times.with_columns(
            orig_trip_id=pl.col("trip_id"),
            trip_id=pl.col("trip_id").cast(str) + "_" + pl.col("file_number").cast(str),
            orig_stop_id=pl.col("stop_id"),
            stop_id=pl.col("stop_id").cast(str) + "_" + pl.col("file_number").cast(str),
        )

        # TODO: If trips_ids is a list of str, filter those. Same with stop_ids
        # if trips is not None:
        #     stop_times = stop_times.filter(pl.col('trip_id').is_in(trips['trip_id']))

        stop_times = stop_times.with_columns(
            arrival_time=("0" + pl.col("arrival_time").cast(str)).str.slice(-8, 8),
            departure_time=("0" + pl.col("departure_time").cast(str)).str.slice(-8, 8),
        )

        stop_times = stop_times.with_columns(
            departure_time=pl.when(pl.col("departure_time").is_null())
            .then(pl.col("arrival_time"))
            .otherwise(pl.col("departure_time")),
            arrival_time=pl.when(pl.col("arrival_time").is_null())
            .then(pl.col("departure_time"))
            .otherwise(pl.col("arrival_time")),
        )

        stop_times = stop_times.with_columns(
            departure_time_secs=pl.col("departure_time").str.slice(0, 2).cast(int)
            * 3600
            + pl.col("departure_time").str.slice(3, 2).cast(int) * 60
            + pl.col("departure_time").str.slice(6, 2).cast(int),
            arrival_time_secs=pl.col("arrival_time").str.slice(0, 2).cast(int) * 3600
            + pl.col("arrival_time").str.slice(3, 2).cast(int) * 60
            + pl.col("arrival_time").str.slice(6, 2).cast(int),
        ).sort("trip_id", "stop_sequence")

        stop_times = stop_times.with_columns(
            pl.col("departure_time_secs").interpolate(),
            pl.col("arrival_time_secs").interpolate(),
        )

        # TODO: Check we dont interpolete between different trip ids
        stop_times = stop_times.with_columns(
            departure_time=self.to_hhmmss(stop_times, "departure_time_secs"),
            arrival_time=self.to_hhmmss(stop_times, "arrival_time_secs"),
        )

        # TODO: Maybe we shouldn't use departue_time_secs_24
        stop_times = stop_times.with_columns(
            departure_time_secs_24=pl.when(pl.col("departure_time_secs") >= 24 * 3600)
            .then(pl.col("departure_time_secs") - (24 * 3600))
            .otherwise(pl.col("departure_time_secs"))
        )

        # TODO: Maybe we don't need it
        return stop_times.unique(["trip_id", "stop_id", "departure_time_secs"]).sort(
            "trip_id", "departure_time", "stop_sequence"
        )

    def to_hhmmss(self, df, field):
        """From seconds to hh:mm::ss"""
        hours = ("0" + (df[field] // 3600).cast(int).cast(str)).str.slice(-2, 2)
        minutes = ("0" + ((df[field] % 3600) // 60).cast(int).cast(str)).str.slice(
            -2, 2
        )
        seconds = ("0" + (df[field] % 60).cast(int).cast(str)).str.slice(-2, 2)
        return hours + ":" + minutes + ":" + seconds
