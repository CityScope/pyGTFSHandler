# -*- coding: utf-8 -*-
"""Shared helpers for downloading+stitching historic GTFS publications.

Several sources expose a per-feed list of past published versions. Unlike
Spain's NAP -- which only exposes a raw upload timestamp that has to be
validated against each version's own calendar (see
`downloaders.spain.nap._trim_calendars` and
`downloaders.spain.utils.resolve_publication_start_date`) -- Mobility
Database and Transitland both already tag each version with the real
service date range they computed from that version's own
`calendar.txt`/`calendar_dates.txt` (`service_date_range_start/end` and
`earliest_calendar_date`/`latest_calendar_date`, respectively). So
`download_and_stitch_versions` here is a simpler variant of NAP's
`_download_and_stitch_history`: it doesn't need to guess a version's start
date, only trim each version's calendar so it stops before the next
version starts, then stitch everything with
`pyGTFSHandler.utils.stack_gtfs.historic_stack`.
"""

import logging
import os
import shutil
import zipfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...utils import hashing
from ..spain.utils import process_calendar, process_calendar_dates

logger = logging.getLogger(__name__)


def select_versions_covering_range(
    entries: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    get_start: Callable[[Dict[str, Any]], Optional[datetime]],
) -> List[Dict[str, Any]]:
    """Keep entries starting inside `[start_date, end_date]`, plus one before.

    Mirrors the history-selection logic in
    `NAPDownloader.download_historic_stack`: entries whose start date
    falls in the requested range, plus the single latest entry starting
    before `start_date` (so the beginning of the range is still covered
    even if the closest version predates it).

    Args:
        entries: Version entries, in any order. Entries for which
            `get_start` returns `None` are skipped.
        start_date: Start of the requested date range.
        end_date: End of the requested date range.
        get_start: Callable returning an entry's start date.

    Returns:
        Matching entries, sorted chronologically by `get_start`.
    """
    first_entry, first_start = None, None
    selected = []
    for entry in entries:
        entry_start = get_start(entry)
        if entry_start is None:
            continue
        if entry_start < start_date and (first_start is None or first_start < entry_start):
            first_start, first_entry = entry_start, entry
        if start_date <= entry_start <= end_date:
            selected.append(entry)

    if first_entry is not None and first_entry not in selected:
        selected.append(first_entry)

    selected.sort(key=get_start)
    return selected


def download_and_stitch_versions(
    versions: List[Dict[str, Any]],
    main_path: str,
    day_separation: int,
    end_date: datetime,
    overwrite: bool,
    download_fn: Callable[[Dict[str, Any], str], bool],
) -> List[str]:
    """Download each historic version and trim calendars against the next.

    Args:
        versions: Version entries, sorted chronologically (oldest first),
            each a dict with a `"start_date"` key (a `datetime` already
            known to fall inside that version's own service date range).
        main_path: Base output path for this feed's stitched result.
        day_separation: Minimum number of days a version is assumed to
            stay valid for, if service periods don't force it shorter.
        end_date: End of the requested date range.
        overwrite: If True, re-download versions that already exist on
            disk.
        download_fn: Callback `(version, dest_zip_path) -> bool` that
            downloads `version`'s GTFS zip to `dest_zip_path`, returning
            True on success (False/exception skips that version).

    Returns:
        Paths (without extension) to each downloaded/extracted version,
        in the order they should be stitched.
    """
    dates = [v["start_date"] for v in versions]
    path_stack: List[str] = []
    i = 0
    while i < len(versions):
        version = versions[i]
        file_date = dates[i]
        file_path = os.path.normpath(f"{main_path}_start_date_{file_date.strftime('%Y%m%d')}")

        if os.path.isdir(file_path) and not overwrite:
            logger.info(f"File '{file_path}' already exists. Skipping download.")
        else:
            if not download_fn(version, file_path + ".zip"):
                i += 1
                continue

            if path_stack and os.path.isfile(path_stack[-1] + ".zip"):
                if hashing.compare_paths(path_stack[-1] + ".zip", file_path + ".zip"):
                    os.remove(file_path + ".zip")
                    logger.info(f"Version '{file_path}' is identical to the previous one. Skipping.")
                    i += 1
                    continue
                os.remove(path_stack[-1] + ".zip")

            os.makedirs(file_path, exist_ok=True)
            with zipfile.ZipFile(file_path + ".zip", "r") as zip_ref:
                zip_ref.extractall(file_path)

        next_index, min_end_date = _trim_calendars(file_path, file_date, dates, day_separation)

        path_stack.append(file_path)
        if next_index is None or next_index <= i:
            next_index = i + 1
        elif dates[next_index] >= end_date or (min_end_date is not None and min_end_date >= end_date):
            break

        if next_index >= len(versions):
            break

        i = next_index

    return path_stack


def _trim_calendars(
    file_path: str,
    file_date: datetime,
    dates: List[datetime],
    day_separation: int,
) -> Tuple[Optional[int], Optional[datetime]]:
    """Trim `calendar.txt`/`calendar_dates.txt` for one historic version.

    Same trimming logic as `NAPDownloader._trim_calendars`, minus the
    `resolve_publication_start_date` step: `file_date` here already comes
    from the source API's own service-date-range field, so it doesn't
    need to be validated against the version's calendar first.
    """
    calendar_path = os.path.normpath(os.path.join(file_path, "calendar.txt"))
    calendar_dates_path = os.path.normpath(os.path.join(file_path, "calendar_dates.txt"))
    has_calendar = os.path.isfile(calendar_path)
    has_calendar_dates = os.path.isfile(calendar_dates_path)

    next_index, min_end_date = None, None
    if has_calendar:
        next_index, min_end_date = process_calendar(
            calendar_path, file_date, dates, day_separation, calendar_path
        )

    if has_calendar_dates:
        next_index, min_end_date = process_calendar_dates(
            calendar_dates_path,
            file_date,
            dates,
            day_separation,
            calendar_dates_path,
            next_index=next_index,
            has_calendar=has_calendar,
        )

    if not has_calendar and not has_calendar_dates:
        logger.warning(f"Version '{file_path}' has no calendar.txt or calendar_dates.txt")

    return next_index, min_end_date


def zip_stitched_feed(
    folder_path: str, source_name: str, start_date: datetime, end_date: datetime, output_dir: str
) -> str:
    """Zip a stitched GTFS folder as `{source_name}_{start}_{end}.zip`.

    Args:
        folder_path: Path to the stitched GTFS folder (as written by
            `historic_stack`).
        source_name: Identifies the feed/dataset within its source, used
            as the output zip's filename prefix.
        start_date: Start of the covered date range.
        end_date: End of the covered date range.
        output_dir: Directory the zip is written into.

    Returns:
        Path to the written zip file.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_stem = f"{source_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    zip_path = os.path.join(output_dir, zip_stem)
    return shutil.make_archive(zip_path, "zip", folder_path)


def cleanup_version_paths(path_stack: List[str]) -> None:
    """Remove downloaded/extracted intermediate version files and folders."""
    for f in path_stack:
        if os.path.isfile(f + ".zip"):
            os.remove(f + ".zip")
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
