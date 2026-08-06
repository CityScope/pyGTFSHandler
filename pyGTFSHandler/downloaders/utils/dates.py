# -*- coding: utf-8 -*-
"""Generic start/end date normalization shared by historic-download flows.

Mirrors `downloaders.spain.utils.input_date`'s None-filling behavior (only
one of `start_date`/`end_date` needs to be given; the other defaults to
it), but expects ISO `"YYYY-MM-DD"` strings rather than NAP's `dd-mm-yyyy`,
matching the convention of the APIs this is used for (Mobility Database,
Transitland).
"""

from datetime import date, datetime
from typing import Optional, Tuple, Union

DateLike = Union[str, date, datetime]


def normalize_date_range(
    start_date: Optional[DateLike], end_date: Optional[DateLike]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Normalize a start/end date pair into `datetime`s.

    Args:
        start_date: `None`, the literal string `"today"`, a `date`/
            `datetime`, or an ISO `"YYYY-MM-DD"` string.
        end_date: Same accepted forms as `start_date`.

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

    def _one(d: DateLike) -> datetime:
        if d == "today":
            return datetime.combine(date.today(), datetime.min.time())
        if isinstance(d, datetime):
            return d
        if isinstance(d, date):
            return datetime.combine(d, datetime.min.time())
        if isinstance(d, str):
            return datetime.strptime(d, "%Y-%m-%d")
        raise TypeError(f"Unsupported date value: {d!r}")

    return _one(start_date), _one(end_date)
