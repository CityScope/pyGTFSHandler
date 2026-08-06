"""Tests for `pyGTFSHandler.downloaders.spain.utils`."""

from datetime import date, datetime

import pandas as pd
import pytest

from pyGTFSHandler.downloaders.spain.utils import (
    input_date,
    process_calendar,
    process_calendar_dates,
    resolve_publication_start_date,
)


def test_input_date_both_none():
    assert input_date(None, None) == (None, None)


def test_input_date_fills_missing_end_with_start():
    start, end = input_date("01-02-2024", None)
    assert start == end == datetime(2024, 2, 1)


def test_input_date_accepts_ddmmyyyy_no_dashes():
    start, end = input_date("01022024", "02022024")
    assert start == datetime(2024, 2, 1)
    assert end == datetime(2024, 2, 2)


def test_input_date_accepts_date_object():
    start, end = input_date(date(2024, 2, 1), date(2024, 2, 2))
    assert start == datetime(2024, 2, 1)
    assert end == datetime(2024, 2, 2)


def test_input_date_today_keyword():
    start, end = input_date("today", "today")
    assert start.date() == datetime.now().date()
    assert end.date() == datetime.now().date()


def test_process_calendar_trims_end_date_to_next_publication(tmp_path):
    src = tmp_path / "calendar.txt"
    src.write_text("service_id,monday,start_date,end_date\nS1,1,20240101,20241231\n")
    out = tmp_path / "calendar_out.txt"

    file_date = datetime(2024, 3, 1)
    possible_dates = [datetime(2024, 3, 10), datetime(2024, 4, 1)]

    next_index, min_end_date = process_calendar(
        str(src), file_date, possible_dates, day_separation=10, calendar_path=str(out)
    )

    assert next_index == 0
    df = pd.read_csv(out)
    assert df.loc[0, "end_date"] == 20240310 - 1  # capped to the day before the next publication


def test_process_calendar_no_possible_dates_returns_none(tmp_path):
    src = tmp_path / "calendar.txt"
    src.write_text("service_id,monday,start_date,end_date\nS1,1,20240101,20241231\n")
    out = tmp_path / "calendar_out.txt"

    next_index, min_end_date = process_calendar(
        str(src), datetime(2024, 3, 1), [], day_separation=5, calendar_path=str(out)
    )
    assert next_index is None


def test_process_calendar_dates_filters_by_exception_type(tmp_path):
    src = tmp_path / "calendar_dates.txt"
    src.write_text(
        "service_id,date,exception_type\n"
        "S1,20240301,1\n"
        "S1,20240302,1\n"
        "S1,20240320,1\n"
    )
    out = tmp_path / "calendar_dates_out.txt"

    # day_separation is large enough that the computed cutoff reaches past
    # the first candidate publication date, so it gets selected as next_index.
    next_index, min_end_date = process_calendar_dates(
        str(src),
        datetime(2024, 3, 1),
        [datetime(2024, 3, 5), datetime(2024, 4, 5)],
        day_separation=10,
        calendar_dates_path=str(out),
    )

    assert next_index == 0
    df = pd.read_csv(out)
    assert df["date"].max() <= 20240305


def test_process_calendar_dates_empty_exceptions_advances_index(tmp_path):
    src = tmp_path / "calendar_dates.txt"
    src.write_text("service_id,date,exception_type\nS1,20240301,2\n")
    out = tmp_path / "calendar_dates_out.txt"

    next_index, min_end_date = process_calendar_dates(
        str(src),
        datetime(2024, 3, 1),
        [datetime(2024, 3, 5), datetime(2024, 4, 5)],
        day_separation=1,
        calendar_dates_path=str(out),
        next_index=None,
        has_calendar=True,
    )

    assert next_index is None
    assert min_end_date is None


def test_resolve_start_date_keeps_candidate_when_inside_calendar_range(tmp_path):
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("service_id,monday,start_date,end_date\nS1,1,20240101,20241231\n")

    candidate = datetime(2024, 3, 1)
    resolved = resolve_publication_start_date(str(calendar), None, candidate)
    assert resolved == candidate


def test_resolve_start_date_falls_back_to_calendar_start_when_candidate_outside_range(tmp_path):
    calendar = tmp_path / "calendar.txt"
    # Service only starts well after the publication ("fecha") date.
    calendar.write_text("service_id,monday,start_date,end_date\nS1,1,20240601,20241231\n")

    candidate = datetime(2024, 1, 1)
    resolved = resolve_publication_start_date(str(calendar), None, candidate)
    assert resolved == datetime(2024, 6, 1)


def test_resolve_start_date_uses_calendar_dates_when_no_calendar(tmp_path):
    calendar_dates = tmp_path / "calendar_dates.txt"
    calendar_dates.write_text(
        "service_id,date,exception_type\nS1,20240310,1\nS1,20240401,1\n"
    )

    # Candidate before the earliest exception date -> falls back to it.
    resolved = resolve_publication_start_date(None, str(calendar_dates), datetime(2024, 1, 1))
    assert resolved == datetime(2024, 3, 10)

    # Candidate inside the exception date range -> kept as-is.
    candidate = datetime(2024, 3, 20)
    resolved = resolve_publication_start_date(None, str(calendar_dates), candidate)
    assert resolved == candidate


def test_resolve_start_date_combines_calendar_and_calendar_dates(tmp_path):
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("service_id,monday,start_date,end_date\nS1,1,20240201,20240601\n")
    calendar_dates = tmp_path / "calendar_dates.txt"
    calendar_dates.write_text("service_id,date,exception_type\nS1,20240701,1\n")

    # Range widened by calendar_dates.txt beyond calendar.txt's end_date.
    candidate = datetime(2024, 6, 15)
    resolved = resolve_publication_start_date(str(calendar), str(calendar_dates), candidate)
    assert resolved == candidate


def test_resolve_start_date_no_files_returns_candidate_unchanged():
    candidate = datetime(2024, 3, 1)
    resolved = resolve_publication_start_date(None, None, candidate)
    assert resolved == candidate
