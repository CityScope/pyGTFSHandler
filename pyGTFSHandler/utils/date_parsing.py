# -*- coding: utf-8 -*-
"""Date-related helpers: the epoch-day convention used throughout this
codebase for representing GTFS dates as plain integers, and public-holiday
lookups.

Why this module exists and how it's organized:
-----------------------------------------------
- **`EPOCH`/`datetime_to_days_since_epoch`**: GTFS dates (`calendar.txt`
  `start_date`/`end_date`, `calendar_dates.txt` `date`) are stored throughout
  `models/calendar.py` as plain integer days since `1970-01-01`, not as
  `pl.Date`/`datetime.date` objects -- this keeps date-range comparisons
  (`start_date <= d <= end_date`) simple integer comparisons instead of
  requiring `pl.Date` arithmetic everywhere. `datetime_to_days_since_epoch`
  is how a caller-supplied `datetime.date`/`datetime.datetime` gets converted
  into that same integer space.
- **`get_holidays`**: the one network call related to dates in this package
  (the `date.nager.at` public-holidays API), used by `models/calendar.py`
  for `date_type="holiday"` and related filters. Region resolution
  (`get_country_region`) lives in `geocoding.py`, not here, since it's a
  location lookup rather than a date one.
"""

import warnings
from datetime import date, datetime
from typing import Optional, Union

import polars as pl
import requests

# Constants
EPOCH = date(1970, 1, 1)


def datetime_to_days_since_epoch(dt: Union[datetime, date]) -> int:
    """Convert datetime/date to number of days since 1970-01-01."""
    if isinstance(dt, datetime):
        dt = dt.date()
    return (dt - EPOCH).days


def get_holidays(year: int, country_code: str, subdivision_code: Optional[str] = None) -> pl.DataFrame:
    """Fetch public holidays for a country and optional subdivision."""
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"

    empty_df = pl.DataFrame(schema={
        "date": pl.Int32,
        "localName": pl.Utf8,
        "name": pl.Utf8,
        "countryCode": pl.Utf8,
        "fixed": pl.Boolean,
        "global": pl.Boolean,
        "counties": pl.List(pl.Utf8),
        "launchYear": pl.Int32,
        "types": pl.List(pl.Utf8),
    })

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            warnings.warn(f"Holiday API request failed: {resp.status_code}")
            return empty_df
        holidays = resp.json()
        if not holidays:
            return empty_df
    except Exception as e:
        warnings.warn(f"Holiday API request error: {e}")
        return empty_df

    if subdivision_code:
        holidays = [h for h in holidays if not h.get("counties") or subdivision_code in h.get("counties", [])]

    df = pl.DataFrame(holidays)
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .dt.epoch(time_unit="d")
        )
    return df
