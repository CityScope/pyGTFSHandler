# -*- coding: utf-8 -*-
"""Feed date/time/route-type filtering methods.

Split out of feed.py (which had grown past ~2700 lines) to keep individual
files within a readable size, per the refactor plan's file-size discipline.
These methods are mixed into `Feed` via multiple inheritance in `feed.py`
(`class Feed(FeedFilteringMixin, FeedAnalysisMixin)`), so `self` here is
always a fully-constructed `Feed` instance with `self.lf`, `self.calendar`,
`self.stops`, `self.stop_times`, etc. already populated -- these are not
usable as a standalone class.
"""

from datetime import datetime, time, date, timedelta
from typing import Optional, Union, List
import geopandas as gpd
import polars as pl
import pandas as pd
import warnings
import numpy as np

from ..utils import time_parsing
from ..utils import date_parsing
from ..utils import gtfs_checker
from ..utils import io

SECS_PER_DAY: int = 86400


class FeedFilteringMixin:
    """Mixin providing Feed.filter_by_date_range/filter_by_date/filter_by_time_range/filter_by_route_type/filter and the calendar_new_start_date/calendar_new_end_date helpers."""

    def _max_day_offset(self, data: pl.LazyFrame) -> int:
        """Returns the largest `day_offset` present in `data` (0 if absent/empty).

        Used by `_filter_by_date`/`_filter_by_date_range` to know how many
        "nominal date = real date - day_offset" candidates to check: a stop
        with `day_offset=2` on real date D was scheduled under a service
        whose nominal `service_date` is D-2 days, so date filtering must look
        that far back, not just one day as the old `"_night"` scheme assumed.
        """
        value = data.select(pl.col("day_offset").max()).collect().item()
        return int(value) if value is not None else 0

    def _filter_by_date_range(
        self,
        data: pl.LazyFrame,
        start_date: datetime | date | None = None,
        end_date: datetime | date | None = None,
        date_type: str | list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame to stop_times whose *real* date (`service_date +
        day_offset`) falls within `[start_date, end_date]`.

        For each possible `day_offset` value present in `data`, this looks up
        which services are active on the correspondingly shifted nominal date
        range (`start_date - day_offset`, `end_date - day_offset`), then
        reclassifies weekday/weekend/holiday `date_type` filters against the
        *real* date (`nominal_date + day_offset`) rather than the nominal
        date -- so a Saturday-run trip whose `25:xx` stop actually happens on
        Sunday is correctly treated as a Sunday stop for `date_type="weekend"`
        purposes, even though its `service_id` is only active on Saturdays.

        Args:
            data (pl.LazyFrame): The LazyFrame to be filtered. Must contain
                `service_id` and `day_offset` columns.
            start_date (datetime | date | None): Start of the real-date range
                (inclusive). Defaults to the feed's earliest known date.
            end_date (datetime | date | None): End of the real-date range
                (inclusive). Defaults to the feed's latest known date.
            date_type (str | list[str] | None): Weekday/weekend/holiday
                filter(s), evaluated against the real date.

        Returns:
            pl.LazyFrame: The filtered LazyFrame.
        """
        if start_date is None:
            start_date = self.calendar.min_date
        if end_date is None:
            end_date = self.calendar.max_date
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        max_offset = self._max_day_offset(data)

        matched_service_id_frames = []
        for offset in range(0, max_offset + 1):
            nominal_start = start_date - timedelta(days=offset)
            nominal_end = end_date - timedelta(days=offset)

            date_df = self.calendar.get_services_in_date_range(
                nominal_start, nominal_end, date_type=None,
                lon=self.stops.mean_lon, lat=self.stops.mean_lat,
            )
            if date_df.height == 0:
                continue

            if date_type is not None:
                # Reclassify against the *real* date (nominal + offset), not
                # the nominal service_date used to look up active services.
                date_df = date_df.with_columns(
                    (pl.col("date") + pl.duration(days=offset)).alias("date")
                ).with_columns(
                    pl.col("date").dt.to_string("%A").str.to_lowercase().alias("weekday")
                )
                date_df = self.calendar.filter_by_date_type(
                    date_df, date_type, self.stops.mean_lon, self.stops.mean_lat
                )
                if date_df.height == 0:
                    continue

            offset_service_ids = (
                date_df.select("service_ids").explode("service_ids")
                .rename({"service_ids": "service_id"})
                .unique()
            )
            matched_service_id_frames.append(
                data.filter(pl.col("day_offset") == offset).join(
                    offset_service_ids.lazy(), on="service_id", how="semi"
                )
            )

        if not matched_service_id_frames:
            return data.clear()

        return pl.concat(matched_service_id_frames)

    def _filter_by_date(
        self,
        data: pl.LazyFrame,
        date: datetime | date,
    ) -> pl.LazyFrame:
        """
        Filters a LazyFrame to stop_times whose *real* date (`service_date +
        day_offset`) equals `date`.

        A stop with `day_offset=k` is included if its service is active on
        `date - k days` (its nominal `service_date`), checked independently
        for every `day_offset` value present in `data` -- this is what
        correctly surfaces, e.g., a `25:10:00` stop as part of "services on
        the next calendar day", which the old single-bit `next_day`/`"_night"`
        scheme could not do beyond one day of offset.

        Args:
            data (pl.LazyFrame): The LazyFrame to be filtered. Must contain
                `service_id` and `day_offset` columns.
            date (datetime | date): The desired real date.

        Returns:
            pl.LazyFrame: The filtered LazyFrame.

        Raises:
            Exception: If no services are active on `date` for any offset.
        """
        if isinstance(date, datetime):
            date = date.date()

        max_offset = self._max_day_offset(data)

        matched_frames = []
        for offset in range(0, max_offset + 1):
            nominal_date = date - timedelta(days=offset)
            service_ids = self.calendar.get_services_in_date(nominal_date)
            if not service_ids:
                continue
            service_ids_df = pl.LazyFrame({"service_id": service_ids})
            matched_frames.append(
                data.filter(pl.col("day_offset") == offset).join(
                    service_ids_df, on="service_id", how="semi"
                )
            )

        if not matched_frames:
            raise Exception(f"No services in date {date}")

        return pl.concat(matched_frames)

    def _filter_by_time_range(
        self,
        data: pl.LazyFrame,
        start_time: datetime | time = time(hour=0),
        end_time: datetime | time = time(hour=23, minute=59, second=59),
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
        start_time: int = time_parsing.time_to_seconds(start_time)
        end_time: int = time_parsing.time_to_seconds(end_time)
        if end_time < start_time:
            raise Exception(f"start_time {start_time} should happen before end_time {end_time}")
        
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
            start_time: datetime | time = time(hour=0),
            end_time: datetime | time = time(hour=23, minute=59, second=59),
            route_types: list | int | str | None = None,
            frequencies:bool = True,
            in_aoi:bool = False, 
            delete_last_stop:bool = False
        ):
        if delete_last_stop:
            data = data.filter(pl.col("stop_sequence") != pl.col("stop_sequence").max().over("trip_id"))

        if in_aoi:
            data = data.filter(pl.col("isin_aoi"))

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
        start_time: datetime | time = time(hour=0),
        end_time: datetime | time = time(hour=23, minute=59, second=59),
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
            start_time: datetime | time = time(hour=0),
            end_time: datetime | time = time(hour=23, minute=59, second=59),
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
        end_date = int(date_parsing.datetime_to_days_since_epoch(new_end_date))
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
        start_date = int(date_parsing.datetime_to_days_since_epoch(new_start_date))
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

