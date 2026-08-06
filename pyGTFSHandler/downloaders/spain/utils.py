# -*- coding: utf-8 -*-
"""Helpers specific to the Spanish NAP downloader (`downloaders.spain.nap`).

These functions are not reused by any other downloader (unlike
`downloaders.utils`, which holds source-agnostic helpers), which is why
they live under `downloaders/spain/` instead: they deal with NAP-specific
concerns -- parsing the flexible date formats accepted by NAP's
`find_files`, and stitching together consecutive historic NAP GTFS
publications (`calendar.txt`/`calendar_dates.txt` trimming) ahead of
`pyGTFSHandler.utils.stack_gtfs.historic_stack`.
"""

import csv
import os
import warnings
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple, Union

import pandas as pd

from ...utils.gtfs_checker import normalize_string


def input_date(
    start_date: Optional[Union[str, date, datetime]],
    end_date: Optional[Union[str, date, datetime]],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Normalize NAP's flexible start/end date inputs into `datetime`s.

    Accepts `None`, the literal string `"today"`, `date`/`datetime`
    objects, or strings in `%d%m%Y` or `%d-%m-%Y` format. If only one of
    `start_date`/`end_date` is given, the other is filled in with the same
    value.

    Args:
        start_date: Start date, in any of the accepted forms, or `None`.
        end_date: End date, in any of the accepted forms, or `None`.

    Returns:
        A tuple `(start_date, end_date)` of `datetime` objects, or
        `(None, None)` if both inputs were `None`.
    """
    if end_date is None:
        end_date = start_date
    elif start_date is None:
        start_date = end_date

    if start_date is None and end_date is None:
        return None, None

    if start_date == "today":
        start_date = datetime.now().strftime("%d-%m-%Y")
    if end_date == "today":
        end_date = datetime.now().strftime("%d-%m-%Y")

    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())

    if isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, "%d%m%Y")
        except ValueError:
            start_date = datetime.strptime(start_date, "%d-%m-%Y")

    if isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, "%d%m%Y")
        except ValueError:
            end_date = datetime.strptime(end_date, "%d-%m-%Y")

    return start_date, end_date


def _sniff_separator(path: str) -> str:
    """Detect the CSV delimiter used by a GTFS text file.

    Args:
        path: Path to the CSV/GTFS text file.

    Returns:
        The detected delimiter character.
    """
    with open(path, "r") as f:
        sample = f.read(1024)
        dialect = csv.Sniffer().sniff(sample)
    return dialect.delimiter


def resolve_publication_start_date(
    calendar_path: Optional[str],
    calendar_dates_path: Optional[str],
    candidate_date: datetime,
) -> datetime:
    """Determine a historic publication's true service start date.

    NAP's `/Fichero/historico` only exposes a publication/upload date
    (`fecha`); that is not necessarily when the feed's own service
    actually starts (publications are sometimes uploaded ahead of or
    behind their own `calendar.txt`/`calendar_dates.txt` service period).
    `_trim_calendars` needs a start date that is actually inside the
    feed's service period, since it's used as the lower bound when
    trimming the previous publication's `end_date` in `process_calendar`/
    `process_calendar_dates`.

    If `candidate_date` (the publication date) falls inside the feed's own
    combined `calendar.txt`/`calendar_dates.txt` service date range, it is
    used as-is. Otherwise, it isn't a meaningful anchor for this feed, so
    the feed's own earliest service date is used instead.

    Args:
        calendar_path: Path to the feed's `calendar.txt`, or `None`/a
            nonexistent path if the feed has none.
        calendar_dates_path: Path to the feed's `calendar_dates.txt`, or
            `None`/a nonexistent path if the feed has none.
        candidate_date: The publication date to validate (NAP's `fecha`).

    Returns:
        `candidate_date` if it falls inside the feed's service date
        range, otherwise the feed's earliest service date. Falls back to
        `candidate_date` unchanged if no service dates could be read from
        either file.
    """
    min_date, max_date = None, None

    if calendar_path and os.path.isfile(calendar_path):
        sep = _sniff_separator(calendar_path)
        df = pd.read_csv(calendar_path, dtype=str, sep=sep)
        df = df.rename(columns={c: normalize_string(c) for c in df.columns})
        if len(df) and "start_date" in df.columns and "end_date" in df.columns:
            starts = pd.to_datetime(df["start_date"], format="%Y%m%d", errors="coerce")
            ends = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
            if starts.notna().any():
                min_date = starts.min() if min_date is None else min(min_date, starts.min())
            if ends.notna().any():
                max_date = ends.max() if max_date is None else max(max_date, ends.max())

    if calendar_dates_path and os.path.isfile(calendar_dates_path):
        sep = _sniff_separator(calendar_dates_path)
        df = pd.read_csv(calendar_dates_path, dtype=str, sep=sep)
        df = df.rename(columns={c: normalize_string(c) for c in df.columns})
        if len(df) and "date" in df.columns:
            dates = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            if dates.notna().any():
                min_date = dates.min() if min_date is None else min(min_date, dates.min())
                max_date = dates.max() if max_date is None else max(max_date, dates.max())

    if min_date is None or max_date is None:
        return candidate_date

    if min_date <= candidate_date <= max_date:
        return candidate_date

    return min_date.to_pydatetime()


def process_calendar(
    path: str,
    file_date: datetime,
    possible_dates: List[datetime],
    day_separation: int,
    calendar_path: str,
) -> Tuple[Optional[int], datetime]:
    """Trim a historic `calendar.txt` so it stops before the next publication.

    Used while stitching together successive NAP GTFS publications: caps
    `end_date` at whichever comes first among `file_date + day_separation`
    days, the file's own minimum `end_date`, or the next candidate
    publication date in `possible_dates`.

    Args:
        path: Path to the source `calendar.txt`.
        file_date: Publication date of this GTFS file.
        possible_dates: Candidate dates of subsequent publications, in
            chronological order.
        day_separation: Minimum number of days a publication is assumed to
            stay valid for, if service periods don't force it shorter.
        calendar_path: Path to write the trimmed `calendar.txt` to.

    Returns:
        A tuple `(next_index, min_end_date)`: `next_index` is the index
        into `possible_dates` of the publication that should follow this
        one (or `None` if none qualifies), and `min_end_date` is the
        computed cutoff date.
    """
    sep = _sniff_separator(path)
    df = pd.read_csv(path, dtype=str, sep=sep)
    df = df.rename(columns={c: normalize_string(c) for c in df.columns})
    df["start_date"] = df["start_date"].astype(int)
    df["end_date"] = df["end_date"].astype(int)
    date_int = int(file_date.strftime("%Y%m%d"))
    df["start_date"] = df["start_date"].apply(lambda x: min(x, date_int))

    if len(possible_dates) == 0:
        return None, file_date

    min_end_date = pd.to_datetime(df["end_date"].astype(str), format="%Y%m%d").min()
    min_end_date = min(file_date + timedelta(days=day_separation), min_end_date)
    min_end_date = min_end_date + timedelta(days=1)

    next_index = None
    for i in range(len(possible_dates)):
        if possible_dates[i] <= min_end_date:
            next_index = i
        else:
            break

    if next_index is not None:
        end_date = possible_dates[next_index]
        end_date_int = int((end_date - timedelta(days=1)).strftime("%Y%m%d"))
        if end_date > file_date:
            df["end_date"] = df["end_date"].apply(lambda x: min(x, end_date_int))
        else:
            df["end_date"] = df["end_date"].apply(lambda x: min(x, date_int))
        df.to_csv(calendar_path, index=False, sep=sep)
        return next_index, min_end_date

    df = df[df["start_date"] <= df["end_date"]]
    df.to_csv(calendar_path, index=False, sep=sep)
    return None, min_end_date


def process_calendar_dates(
    path: str,
    file_date: datetime,
    possible_dates: List[datetime],
    day_separation: int,
    calendar_dates_path: str,
    next_index: Optional[int] = None,
    has_calendar: bool = True,
) -> Tuple[Optional[int], Optional[datetime]]:
    """Trim a historic `calendar_dates.txt` to match the stitched period.

    Companion to `process_calendar`, used for GTFS publications that rely
    on (or supplement with) `calendar_dates.txt` exceptions rather than a
    plain weekly `calendar.txt` pattern.

    Args:
        path: Path to the source `calendar_dates.txt`.
        file_date: Publication date of this GTFS file.
        possible_dates: Candidate dates of subsequent publications, in
            chronological order.
        day_separation: Minimum number of days a publication is assumed to
            stay valid for, if service exceptions don't force it shorter.
        calendar_dates_path: Path to write the trimmed
            `calendar_dates.txt` to.
        next_index: Index into `possible_dates` already selected by
            `process_calendar`, if this file also has a `calendar.txt`.
        has_calendar: Whether this publication also has a `calendar.txt`
            (only affects a diagnostic warning).

    Returns:
        A tuple `(next_index, cutoff_date)`, with the same meaning as
        `process_calendar`'s return value.
    """
    sep = _sniff_separator(path)
    df = pd.read_csv(path, dtype=str, sep=sep)
    df = df.rename(columns={c: normalize_string(c) for c in df.columns})
    df["date"] = df["date"].astype(int)
    date_int = int(file_date.strftime("%Y%m%d"))
    df = df[df["date"] >= date_int]

    if next_index is not None:
        end_date = possible_dates[next_index]
        end_date_int = int((end_date - timedelta(days=1)).strftime("%Y%m%d"))
        df = df[df["date"] <= end_date_int]
        df.to_csv(calendar_dates_path, index=False, sep=sep)
        return next_index, end_date

    df["exception_type"] = df["exception_type"].astype(int)
    dates = pd.to_datetime(
        df.loc[df["exception_type"] == 1, "date"].astype(str), format="%Y%m%d"
    )
    if len(dates) == 0:
        if not has_calendar:
            warnings.warn(
                f"File {path.replace('calendar_dates.txt', '')} has no calendar.txt, "
                "and calendar_dates.txt has no exception_type 1 rows, so it has no "
                "service dates."
            )
        if next_index is not None and next_index < (len(possible_dates) - 1):
            return next_index + 1, None
        return None, None

    counts = dates.value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index
    min_end_date = candidates.max()
    min_end_date = min(file_date + timedelta(days=day_separation), min_end_date)
    min_end_date = min_end_date + timedelta(days=1)

    new_index = None
    for i in range(len(possible_dates)):
        if possible_dates[i] <= min_end_date:
            new_index = i
        else:
            break

    if new_index is not None:
        end_date = possible_dates[new_index]
        end_date_int = int((end_date - timedelta(days=1)).strftime("%Y%m%d"))
        if end_date > file_date:
            df = df[df["date"] <= end_date_int]
        else:
            df = df[df["date"] <= date_int]
        df.to_csv(calendar_dates_path, index=False, sep=sep)
        return new_index, min_end_date

    df.to_csv(calendar_dates_path, index=False, sep=sep)
    return None, min_end_date
