# -*- coding: utf-8 -*-
"""GTFS calendar.txt/calendar_dates.txt handling: which services are active
on a given date or date range.

Why this module exists and how it's organized:
-----------------------------------------------
`calendar.txt` (weekday-pattern + date-range rows) and `calendar_dates.txt`
(single-date add/remove exceptions) are both parsed and kept *exactly as the
GTFS feed authored them* -- no row duplication, no date shifting. This is a
deliberate change from an earlier version of this module, which used to
duplicate every row as a `"_night"`-suffixed variant shifted +1 day, to
approximate multi-day/overnight trips. That approach only ever supported a
single day of offset and doubled every service regardless of whether it
ever ran past midnight.

The correct handling of overnight/multi-day trips instead lives in
`models/stop_times.py`'s `day_offset` column: a stop_time's *real* calendar
date is `service_date + day_offset`, resolved at query time (see
`Feed._filter_by_date`/`Feed._filter_by_date_range` in `feed_filtering.py`)
by checking, for each `day_offset` value present in the data, whether the
service is active on `queried_date - day_offset`. This module itself only
needs to answer "which services are active on date X" -- it has no
awareness of `day_offset` at all, and `filter_by_date_type` is written so a
caller (`Feed`) can request weekday/weekend/holiday classification against
a date *other than* the row's own nominal date, which is exactly what's
needed to classify an offset-shifted stop_time's real date correctly.
"""

from datetime import datetime, timedelta, date
from typing import Tuple

import polars as pl
from ..utils import geo_polars
from ..utils import date_parsing
from ..utils import gtfs_checker
from ..utils import io
from ..utils import geocoding

from typing import Union, Optional, List
from pathlib import Path
import warnings

class Calendar:
    def __init__(self,lf=None,exceptions_lf=None,min_date=None,max_date=None,service_ids=None) -> None:
        self.lf = lf 
        self.exceptions_lf = exceptions_lf 
        if (lf is not None) or (exceptions_lf is not None):
            if min_date is None:
                min_date, max_date = self._get_min_max_dates(
                    lf, exceptions_lf
                )

        self.min_date = min_date 
        self.max_date = max_date 
        self.service_ids = service_ids

    def load(
        self,
        path: Union[str, Path, List[str], List[Path]],
        start_date: datetime | date | None = None,
        end_date: datetime | date | None = None,
        date_type: list[str] | str | None = None,
        lon: float | None = None,
        lat: float | None = None,
        service_ids: Optional[List[str]] | None = None,
        check_files:bool=False,
        min_file_id=0
    ):
        """
        A class to manage GTFS calendar data, allowing filtering of active services
        by date using calendar.txt and calendar_dates.txt files.


        Args:
            path (str | Path | list[str | Path]): File or directory path(s) containing GTFS calendar files.
            service_ids (list[str], optional): List of service IDs to filter on.


        Attributes:
        path (Path): Directory path containing GTFS calendar files.
        lf (pl.LazyFrame | None): LazyFrame for calendar.txt data.
        exceptions_lf (pl.LazyFrame | None): LazyFrame for calendar_dates.txt data.
        """
        # Normalize to list of Path
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        self.lf = self._read_calendar(paths, service_ids, check_files=check_files, min_file_id=min_file_id)
        self.exceptions_lf = self._read_calendar_dates(paths, service_ids, check_files=check_files, min_file_id=min_file_id)
        if (self.lf is None) and (self.exceptions_lf is None):
            raise Exception(f"No calendar.txt or calendar_dates.txt files found in paths {paths}")
        
        self.min_date, self.max_date = self._get_min_max_dates(
            self.lf, self.exceptions_lf
        )
        if start_date is not None and start_date > self.min_date:
            self.min_date = start_date
        if end_date is not None and end_date < self.max_date:
            self.max_date = end_date

        if isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, datetime):
            end_date = end_date.date()

        if start_date and end_date:
            if start_date == end_date:
                self.service_ids = list(
                    set(self.get_services_in_date(start_date))
                    | set(self.get_services_in_date(start_date - timedelta(days=1)))
                )
                service_ids_df = pl.DataFrame({"service_id": self.service_ids})
            elif start_date > end_date:
                raise Exception("Start date happens after end date")
            else:
                service_ids_df = (
                    self.get_services_in_date_range(
                        start_date - timedelta(days=1),
                        end_date,
                        date_type=date_type,
                        lon=lon,
                        lat=lat,
                    )
                    .select("service_ids")
                    .rename({"service_ids": "service_id"})
                    .explode("service_id")
                    .unique()
                )
                self.service_ids = service_ids_df["service_id"].to_list()

            if (len(self.service_ids) > 0) and (self.service_ids[0] is None):
                self.service_ids = []

            if self.lf is not None:
                self.lf = self.lf.join(
                    service_ids_df.lazy(), on="service_id", how="semi"
                )

            if self.exceptions_lf is not None:
                self.exceptions_lf = self.exceptions_lf.join(
                    service_ids_df.lazy(), on="service_id", how="semi"
                )

        else:
            self.service_ids = service_ids

    def _read_calendar(
        self, paths, service_ids: Optional[List[str]], check_files=False, min_file_id=0
    ) -> Optional[pl.LazyFrame]:
        """
        Reads the calendar.txt files from all paths using io.read_csv_list.

        Args:
            service_ids (Optional[List[str]]): List of service IDs to filter.

        Returns:
            Optional[pl.LazyFrame]: Filtered calendar data or None if no files found.
        """
        calendar_paths: List[Path] = []
        file = "calendar.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                calendar_paths.append(None)
            else:
                calendar_paths.append(new_p)


        schema_dict, _ = gtfs_checker.get_df_schema_dict("calendar.txt")  # assume same schema
        calendar = io.read_csv_list(calendar_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id)
        if (calendar is None) or (calendar.select(pl.len()).collect().item() == 0):
            return None 

        calendar = geo_polars.filter_by_id_column(calendar, "service_id", service_ids)

        # Convert start_date and end_date (YYYYMMDD int) to days since year 1-01-01
        # Safely parse dates to integer days since 1970-01-01
        calendar = calendar.with_columns([
            (
                pl.col("start_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)  # invalid → null
                .dt.epoch(time_unit="d")
            ).alias("start_date"),
            (
                pl.col("end_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .dt.epoch(time_unit="d")
            ).alias("end_date"),
        ])

        # Lazily count rows with nulls (invalid dates)
        null_count_expr = (
            (pl.col("start_date").is_null() | pl.col("end_date").is_null())
            .sum()
            .alias("num_invalid_rows")
        )

        null_count_df = calendar.select(null_count_expr).collect()
        num_invalid_rows = null_count_df.item()  # scalar

        # Drop invalid rows
        calendar = calendar.filter(
            pl.col("start_date").is_not_null() & pl.col("end_date").is_not_null()
        )

        # Warn if any rows were removed
        if num_invalid_rows > 0:
            warnings.warn(f"{num_invalid_rows} rows dropped due to invalid start/end dates.", UserWarning)


        return calendar

    def _read_calendar_dates(
        self, paths, service_ids: Optional[List[str]], check_files=False, min_file_id=0
    ) -> Optional[pl.LazyFrame]:
        """
        Reads the calendar_dates.txt files from all paths using io.read_csv_list.

        Args:
            service_ids (Optional[List[str]]): List of service IDs to filter.

        Returns:
            Optional[pl.LazyFrame]: Filtered calendar_dates data or None if no files found.
        """
        calendar_dates_paths: List[Path] = []
        file = "calendar_dates.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                calendar_dates_paths.append(None)
            else:
                calendar_dates_paths.append(new_p)

        schema_dict, _ = gtfs_checker.get_df_schema_dict("calendar_dates.txt")
        calendar_dates = io.read_csv_list(
            calendar_dates_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id
        )
        if (calendar_dates is None) or (calendar_dates.select(pl.len()).collect().item() == 0):
            return None 
        
        calendar_dates = geo_polars.filter_by_id_column(
            calendar_dates, "service_id", service_ids
        )

        # Convert start_date and end_date (YYYYMMDD int) to days since year 1-01-01
        # Safely parse `date` and convert `exception_type`
        calendar_dates = calendar_dates.with_columns([
            # Parse date safely
            (
                pl.col("date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)  # invalid → null
                .dt.epoch(time_unit="d")
                .alias("date")
            ),
            # Convert exception_type to 1/2 safely
            (
                pl.col("exception_type")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .replace({"added": "1", "removed": "2"})
                .cast(pl.Int32, strict=False)  # invalid → null
                .alias("exception_type")
            ),
        ])

        # Lazily count rows with nulls (invalid date or exception_type)
        null_count_expr = (
            (pl.col("date").is_null() | pl.col("exception_type").is_null())
            .sum()
            .alias("num_invalid_rows")
        )

        null_count_df = calendar_dates.select(null_count_expr).collect()
        num_invalid_rows = null_count_df.item()  # scalar

        # Drop invalid rows
        calendar_dates = calendar_dates.filter(
            pl.col("date").is_not_null() & pl.col("exception_type").is_not_null()
        )

        # Warn if any rows were removed
        if num_invalid_rows > 0:
            warnings.warn(f"{num_invalid_rows} rows dropped due to invalid date or exception_type values.", UserWarning)


        return calendar_dates

    def _get_min_max_dates(
        self, lf: pl.LazyFrame, exceptions_lf: pl.LazyFrame
    ) -> Tuple[date, date]:
        """
        Determines the overall date range of the GTFS feed.

        This internal helper method inspects the calendar and calendar_dates
        dataframes to find the earliest start date and the latest end date
        across all services defined in the feed.

        Args:
            lf: A LazyFrame representing the GTFS `calendar.txt` file.
                It must contain 'start_date' and 'end_date' columns.
            exceptions_lf: A LazyFrame representing the GTFS `calendar_dates.txt`
                           file. It must contain a 'date' column.

        Returns:
            A tuple containing two `datetime.date` objects: the absolute minimum
            and maximum service dates found in the feed.

        Raises:
            ValueError: If neither `lf` nor `exceptions_lf` contains any
                        date information from which to infer a range.
        """
        # A list to collect all start/end date values (as days since epoch).
        date_bounds_as_days: list[int] = []

        # Extract min/max dates from the calendar data (calendar.txt) if available.
        if lf is not None:
            # Collect is necessary to compute min/max on the date columns.
            # This is efficient as calendar.txt is typically very small.
            cal_dates = lf.select(["start_date", "end_date"]).collect()
            if not cal_dates.is_empty():
                date_bounds_as_days.append(cal_dates["start_date"].min())
                date_bounds_as_days.append(cal_dates["end_date"].max())

        # Extract min/max dates from the calendar exceptions (calendar_dates.txt).
        if exceptions_lf is not None:
            # Collect is used here as well for the small exceptions file.
            exception_dates = exceptions_lf.select(["date"]).collect()
            if not exception_dates.is_empty():
                date_bounds_as_days.append(exception_dates["date"].min())
                date_bounds_as_days.append(exception_dates["date"].max())

        # If no dates were found in either file, the feed is invalid.
        if not date_bounds_as_days:
            raise ValueError(
                "Cannot determine date range. No data in 'calendar.txt' or "
                "'calendar_dates.txt'."
            )

        # Find the overall minimum and maximum from the collected date boundaries.
        min_day_offset = min(date_bounds_as_days)
        max_day_offset = max(date_bounds_as_days)

        # Convert the integer day offsets back into standard datetime.date objects.
        min_date = date_parsing.EPOCH + timedelta(days=min_day_offset)
        max_date = date_parsing.EPOCH + timedelta(days=max_day_offset)

        return min_date, max_date

    def get_services_in_date(self, date: datetime | date) -> List[str]:
        """
        Returns a list of service_ids active on a given date.

        Combines data from calendar.txt and calendar_dates.txt to include
        exceptions.

        Args:
            date (datetim | date): Date to check for active services.

        Returns:
            list[str]: Sorted list of active service IDs on the given date.
        """
        date_int = date_parsing.datetime_to_days_since_epoch(date)
        weekday = date.strftime("%A").lower()  # e.g., 'monday'

        # Filter calendar.txt for services active on this weekday and date
        calendar_filtered = None
        if self.lf is not None:
            calendar_filtered = self.lf.filter(
                (pl.col(weekday) == 1)
                & (pl.col("start_date") <= date_int)
                & (pl.col("end_date") >= date_int)
            ).select("service_id")

        # Filter calendar_dates.txt for exceptions on this date
        remove_services = None
        add_services = None
        if self.exceptions_lf is not None:
            remove_services = self.exceptions_lf.filter(
                (pl.col("date") == date_int)
                & (pl.col("exception_type") == 2)  # Removed service
            ).select("service_id")

            add_services = self.exceptions_lf.filter(
                (pl.col("date") == date_int)
                & (pl.col("exception_type") == 1)  # Added service
            ).select("service_id")

        # Collect data frames as needed
        lazyframes = [
            df
            for df in [calendar_filtered, remove_services, add_services]
            if df is not None
        ]
        collected = pl.collect_all(lazyframes)

        calendar_df = (
            collected[0]
            if calendar_filtered is not None
            else pl.DataFrame({"service_id": []})
        )
        idx = 1 if calendar_filtered is not None else 0
        remove_df = (
            collected[idx]
            if remove_services is not None
            else pl.DataFrame({"service_id": []})
        )
        add_df = (
            collected[idx + 1]
            if add_services is not None
            else pl.DataFrame({"service_id": []})
        )

        # Compute final active service IDs
        active_services = (
            set(calendar_df["service_id"])
            .union(add_df["service_id"])
            .difference(remove_df["service_id"])
        )

        return list(active_services)

    def get_services_in_date_range(
        self,
        start_date: Optional[datetime | date] = None,
        end_date: Optional[datetime | date] = None,
        date_type: Optional[str | list[str]] = None,
        lon: float = None,
        lat: float = None,
    ) -> pl.DataFrame:
        """
        Returns a Polars DataFrame with active service IDs for each date in the range.

        The output contains columns:
            - 'date' (YYYY-MM-DD string)
            - 'weekday' (lowercase weekday name)
            - 'service_ids' (sorted list of active service IDs on that date)

        If start_date or end_date is not provided, they are inferred from available data.

        Args:
            start_date (datetime|date, optional): Start date of range.
            end_date (datetime|date, optional): End date of range.

        Returns:
            pl.DataFrame: DataFrame with active services per date.
        """

        # Use provided or inferred dates
        start_date = start_date or (self.min_date)
        end_date = end_date or (self.max_date)

        if isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, datetime):
            end_date = end_date.date()

        if end_date < start_date:
            raise Exception("end_date should be after start_date")

        # Generate list of dates in range with weekday info
        date_list = [
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        ]
        date_info = [
            {"date": d.isoformat(), "weekday": d.strftime("%A").lower(), "date_int": int(date_parsing.datetime_to_days_since_epoch(d))}
            for d in date_list
        ]

        # Map weekday to service IDs based on calendar.txt
        weekday_service_map = {
            wd: set()
            for wd in [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
        }

        if self.lf is not None:
            start_int = date_parsing.datetime_to_days_since_epoch(start_date)
            end_int = date_parsing.datetime_to_days_since_epoch(end_date)

            calendar_df = (
                self.lf.select(
                    [
                        "service_id",
                        "start_date",
                        "end_date",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ]
                )
                .filter(
                    (pl.col("start_date") <= end_int)
                    & (pl.col("end_date") >= start_int)
                )
                .collect()
            )

            # Initialize date to services mapping based on weekday
            date_service_map = {
                entry["date"]: {
                    "weekday": entry["weekday"],
                    "services": set(
                        calendar_df.filter(
                            (pl.col("start_date") <= entry["date_int"])
                            & (pl.col("end_date") >= entry["date_int"]) 
                            & (pl.col(entry["weekday"]) == 1)
                        )["service_id"].to_list()
                    ),
                }
                for entry in date_info
            }
        else:
            # Initialize date to services mapping based on weekday
            date_service_map = {
                entry["date"]: {
                    "weekday": entry["weekday"],
                    "services": set(),
                }
                for entry in date_info
            }

        # Apply exceptions from calendar_dates.txt
        if self.exceptions_lf is not None:
            start_int = date_parsing.datetime_to_days_since_epoch(start_date)
            end_int = date_parsing.datetime_to_days_since_epoch(end_date)

            calendar_dates_df = (
                self.exceptions_lf.select(["date", "service_id", "exception_type"])
                .filter((pl.col("date") >= start_int) & (pl.col("date") <= end_int))
                .collect()
            )

            for row in calendar_dates_df.iter_rows(named=True):
                date_str = (date_parsing.EPOCH + timedelta(days=row["date"])).isoformat()
                service_id = row["service_id"]
                exception = row["exception_type"]

                if date_str not in date_service_map:
                    date_service_map[date_str] = {"weekday": None, "services": set()}

                if exception == 1:
                    date_service_map[date_str]["services"].add(service_id)
                elif exception == 2:
                    date_service_map[date_str]["services"].discard(service_id)

        # Prepare final DataFrame output
        result = pl.DataFrame(
            [
                {
                    "date": date,
                    "weekday": data["weekday"],
                    "service_ids": sorted(data["services"]),
                }
                for date, data in sorted(date_service_map.items())
            ] 
        ).with_columns(
            pl.col("date")
            .str.strptime(pl.Datetime, "%Y-%m-%d")
            .cast(pl.Date)
            .alias("date")
        )

        if date_type is not None:
            result = self.filter_by_date_type(result, date_type, lon, lat)

        return result

    VALID_DATE_TYPES = {
        "workday",
        "weekday",
        "businessday",
        "holiday",
        "non_workday",
        "non_businessday",
        "non_weekday",
        "weekend",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }

    def filter_by_date_type(
        self,
        result: pl.DataFrame,
        date_type: str | list[str],
        lon: float | None,
        lat: float | None,
    ) -> pl.DataFrame:
        """Filters a per-date DataFrame down to rows matching `date_type`.

        Extracted out of `get_services_in_date_range` so `Feed` can reuse the
        exact same weekday/weekend/holiday classification logic against a
        *different* date column than the one the row was originally computed
        for. This matters for multi-day trips: a stop_time's real calendar
        date is `service_date + day_offset`, which can fall on a different
        weekday (or even cross into a holiday) than its nominal service_date.
        `Feed._filter_by_date_range` calls this with `result["date"]`/
        `result["weekday"]` already replaced by the *offset-adjusted* real
        date before requesting a "weekend"/"holiday"/etc. classification, so
        the classification is always evaluated against the day a stop_time
        actually, physically happens on.

        Args:
            result: DataFrame with at least `date` (pl.Date) and `weekday`
                (lowercase weekday name) columns, one row per calendar date.
            date_type: One or more of `VALID_DATE_TYPES`.
            lon: Longitude used to resolve the country/subdivision for
                holiday lookups (only needed if `date_type` requests
                holiday-aware filtering).
            lat: Latitude, see `lon`.

        Returns:
            pl.DataFrame: `result` filtered to rows matching every requested
            `date_type` (AND semantics across multiple values).
        """
        if isinstance(date_type, str):
            date_type = [date_type]

        date_type = [dt.lower() for dt in date_type]
        invalid = [dt for dt in date_type if dt not in self.VALID_DATE_TYPES]
        if invalid:
            raise Exception(f"Date type(s) not implemented: {invalid}")

        needs_holiday_lookup = (
            ("holiday" in date_type) or
            ("workday" in date_type) or
            ("businessday" in date_type) or
            ("non_workday" in date_type) or
            ("non_businessday" in date_type)
        )
        needs_weekend = (
            needs_holiday_lookup
            or ("weekend" in date_type)
            or ("non_weekday" in date_type)
        )
        if needs_weekend:
            result = self.add_holidays_and_weekends(
                result, lon, lat, needs_holiday=needs_holiday_lookup
            )

        # Apply filters one by one (AND logic)
        if ("workday" in date_type) or ("businessday" in date_type):
            result = result.filter(
                (~ pl.col("holiday"))
                & (~ pl.col("weekend"))
            )

        if "weekday" in date_type:
            result = result.filter(
                (~ pl.col("weekend"))
            )

        if ("non_workday" in date_type) or ("non_businessday" in date_type):
            result = result.filter(
                (pl.col("holiday")) | (pl.col("weekend"))
            )

        if "holiday" in date_type:
            result = result.filter(pl.col("holiday"))

        if ("weekend" in date_type) or ("non_weekday" in date_type):
            result = result.filter(pl.col("weekend"))

        for day in [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]:
            if day in date_type:
                result = result.filter(pl.col("weekday") == day)

        if "holiday" in date_type:
            result = result.drop("holiday", "weekend")

        return result

    def add_holidays_and_weekends(self, data, lon, lat, needs_holiday: bool = True):
        """Adds `weekend` (always, purely from `weekday`, no network) and,
        when `needs_holiday` is True, `holiday` (requires an external
        country/subdivision lookup + holiday-calendar fetch via `utils`).

        Split out so date_type filters that only need `weekend` (e.g. a
        plain `"weekend"` request) never trigger the network-dependent
        holiday lookup.
        """
        # If LazyFrame, collect to DataFrame
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Weekend: check if weekday is Saturday or Sunday (no network needed).
        data = data.with_columns(
            pl.col("weekday").is_in(["saturday", "sunday"]).alias("weekend")
        )

        if not needs_holiday:
            return data

        # Determine country and subdivision (your utils function)
        country_code, subdivision_code = geocoding.get_country_region(lat, lon)

        # Extract unique years as list of ints
        years = data.select(pl.col("date").dt.year()).unique().to_series().to_list()

        # Collect holidays for each year
        holidays_df = [
            date_parsing.get_holidays(year, country_code, subdivision_code) for year in years
        ]
        holidays_df = pl.concat(holidays_df)

        # Convert holidays_df 'date' column (assumed days since epoch) to datetime (Polars Date)
        holidays_df = holidays_df.with_columns(
            (
                pl.lit(date_parsing.EPOCH).cast(pl.Date) + pl.duration(days=pl.col("date"))
            ).alias("date")
        )

        # Holiday: if 'date' is in holidays_df
        data = data.with_columns(
            pl.col("date").is_in(holidays_df.get_column("date")).alias("holiday")
        )

        return data
