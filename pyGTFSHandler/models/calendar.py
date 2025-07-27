from datetime import datetime, timedelta, date
from typing import Tuple

import polars as pl
from utils import (
    get_df_schema_dict,
    read_csv_list,
    datetime_to_days_since_epoch,
    EPOCH,
)

from typing import Union, Optional, List
from pathlib import Path


class Calendar:
    def __init__(
        self,
        path: Union[str, Path, List[str], List[Path]],
        start_date: datetime = None,
        end_date: datetime = None,
        service_ids: Optional[List[str]] = None,
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
            self.paths = [Path(path)]
        else:
            self.paths = [Path(p) for p in path]

        self.lf = self.__read_calendar(service_ids)
        self.exceptions_lf = self.__read_calendar_dates(service_ids)
        self.min_date, self.max_date = self.__get_min_max_dates(
            self.lf, self.exceptions_lf
        )
        if start_date and end_date:
            if start_date.date == end_date.date:
                self.service_ids = self.get_services_in_date(start_date)
            elif start_date.date > end_date.date:
                raise Exception("Start date happens after end date")
            else:
                service_df = self.get_services_in_date_range(start_date, end_date)
                self.service_ids = (
                    service_df.explode("service_ids").agg("service_ids").to_list()
                )

            service_ids_df = pl.DataFrame({"service_id": self.service_ids})
            if self.lf:
                self.lf = self.lf.join(
                    service_ids_df.lazy(), on="service_id", how="inner"
                )

            if self.exceptions_lf:
                self.exceptions_lf = self.exceptions_lf.join(
                    service_ids_df.lazy(), on="service_id", how="inner"
                )

        else:
            self.service_ids = service_ids

    def __read_calendar(
        self, service_ids: Optional[List[str]]
    ) -> Optional[pl.LazyFrame]:
        """
        Reads the calendar.txt files from all paths using utils.read_csv_list.

        Args:
            service_ids (Optional[List[str]]): List of service IDs to filter.

        Returns:
            Optional[pl.LazyFrame]: Filtered calendar data or None if no files found.
        """
        calendar_paths = [
            p / "calendar.txt" for p in self.paths if (p / "calendar.txt").exists()
        ]
        if not calendar_paths:
            return None

        schema_dict = get_df_schema_dict(calendar_paths[0])  # assume same schema
        calendar = read_csv_list(calendar_paths, schema_overrides=schema_dict)

        if service_ids:
            service_ids_df = pl.DataFrame({"service_id": service_ids})
            calendar = calendar.join(
                service_ids_df.lazy(), on="service_id", how="inner"
            )

        # Convert start_date and end_date (YYYYMMDD int) to days since year 1-01-01
        calendar = calendar.with_columns(
            [
                pl.col("start_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .dt.epoch(time_unit="d")  # days since 1970-01-01 (int)
                .alias("start_date"),
                pl.col("end_date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .dt.epoch(time_unit="d")
                .alias("end_date"),
            ]
        )

        night_services = calendar.with_columns(
            [
                (pl.col("service_id") + "_night").alias("service_id"),
                (pl.col("start_date") + 1).alias("start_date"),
                (pl.col("end_date") + 1).alias("end_date"),
                # Shift weekdays forward by one day
                pl.col("sunday").alias("monday"),
                pl.col("monday").alias("tuesday"),
                pl.col("tuesday").alias("wednesday"),
                pl.col("wednesday").alias("thursday"),
                pl.col("thursday").alias("friday"),
                pl.col("friday").alias("saturday"),
                pl.col("saturday").alias("sunday"),
            ]
        )

        calendar = pl.concat([calendar, night_services])

        return calendar

    def __read_calendar_dates(
        self, service_ids: Optional[List[str]]
    ) -> Optional[pl.LazyFrame]:
        """
        Reads the calendar_dates.txt files from all paths using utils.read_csv_list.

        Args:
            service_ids (Optional[List[str]]): List of service IDs to filter.

        Returns:
            Optional[pl.LazyFrame]: Filtered calendar_dates data or None if no files found.
        """
        calendar_dates_paths = [
            p / "calendar_dates.txt"
            for p in self.paths
            if (p / "calendar_dates.txt").exists()
        ]
        if not calendar_dates_paths:
            return None

        schema_dict = get_df_schema_dict(calendar_dates_paths[0])
        calendar_dates = read_csv_list(
            calendar_dates_paths, schema_overrides=schema_dict
        )

        if service_ids:
            service_ids_df = pl.DataFrame({"service_id": service_ids})
            calendar_dates = calendar_dates.join(
                service_ids_df.lazy(), on="service_id", how="inner"
            )

        # Convert start_date and end_date (YYYYMMDD int) to days since year 1-01-01
        calendar_dates = calendar_dates.with_columns(
            [
                pl.col("date")
                .cast(pl.Utf8)
                .str.strptime(pl.Date, "%Y%m%d")
                .dt.epoch(time_unit="d")  # days since 1970-01-01 (int)
                .alias("date"),
            ]
        )

        # Create night services by duplicating and shifting dates +1 day
        night_services = calendar_dates.with_columns(
            [
                (pl.col("service_id") + "_night").alias("service_id"),
                (pl.col("date") + 1).alias("date"),  # shift date one day forward
            ]
        )

        calendar_dates = pl.concat([calendar_dates, night_services])

        return calendar_dates

    def __get_min_max_dates(
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
        min_date = self.EPOCH + timedelta(days=min_day_offset)
        max_date = self.EPOCH + timedelta(days=max_day_offset)

        return min_date, max_date

    def get_services_in_date(self, date: datetime) -> List[str]:
        """
        Returns a list of service_ids active on a given date.

        Combines data from calendar.txt and calendar_dates.txt to include
        exceptions.

        Args:
            date (datetime): Date to check for active services.

        Returns:
            list[str]: Sorted list of active service IDs on the given date.
        """
        date_int = datetime_to_days_since_epoch(date)
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
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Returns a Polars DataFrame with active service IDs for each date in the range.

        The output contains columns:
            - 'date' (YYYY-MM-DD string)
            - 'weekday' (lowercase weekday name)
            - 'service_ids' (sorted list of active service IDs on that date)

        If start_date or end_date is not provided, they are inferred from available data.

        Args:
            start_date (datetime, optional): Start date of range.
            end_date (datetime, optional): End date of range.

        Returns:
            pl.DataFrame: DataFrame with active services per date.
        """

        # Use provided or inferred dates
        start_date = start_date or (self.min_date)
        end_date = end_date or (self.max_date)

        # Generate list of dates in range with weekday info
        date_list = [
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        ]
        date_info = [
            {"date": d.isoformat(), "weekday": d.strftime("%A").lower()}
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
            start_int = datetime_to_days_since_epoch(start_date)
            end_int = datetime_to_days_since_epoch(end_date)

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

            for wd in weekday_service_map:
                weekday_service_map[wd] = set(
                    calendar_df.filter(pl.col(wd) == 1)["service_id"].to_list()
                )

        # Initialize date to services mapping based on weekday
        date_service_map = {
            entry["date"]: {
                "weekday": entry["weekday"],
                "services": set(weekday_service_map[entry["weekday"]]),
            }
            for entry in date_info
        }

        # Apply exceptions from calendar_dates.txt
        if self.exceptions_lf is not None:
            start_int = datetime_to_days_since_epoch(start_date)
            end_int = datetime_to_days_since_epoch(end_date)

            calendar_dates_df = (
                self.exceptions_lf.select(["date", "service_id", "exception_type"])
                .filter((pl.col("date") >= start_int) & (pl.col("date") <= end_int))
                .collect()
            )

            for row in calendar_dates_df.iter_rows(named=True):
                date_str = (EPOCH + timedelta(days=row["date"])).isoformat()
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
        )

        return result
