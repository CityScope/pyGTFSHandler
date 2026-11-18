# -*- coding: utf-8 -*-
"""Detect GTFS feeds that publish the same service twice.

Open-data portals often list one operator's network under several entries (the operator's own file and a regional authority's copy,
or successive uploads). Loading both counts every trip twice: stops are doubled and headways come out half as long.
`find_duplicate_feeds` compares feeds by their stop locations and route names and reports which ones to drop.
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def _open(src: Path, name: str):
    """Text reader of `name` inside a feed folder or zip (case-insensitive, nested folder allowed); None when absent."""
    src = Path(src)
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file() and p.name.lower() == name:
                return io.TextIOWrapper(open(p, "rb"), encoding="utf-8-sig", errors="replace")
        return None
    with zipfile.ZipFile(src) as z:
        for n in z.namelist():
            if n.lower().split("/")[-1] == name:
                return io.TextIOWrapper(io.BytesIO(z.read(n)), encoding="utf-8-sig", errors="replace")
    return None


def feed_signature(src: Path, coord_decimals: int = 4) -> Tuple[Set[Tuple[float, float]], Set[str]]:
    """``(stop locations rounded to ~10 m, route short names)`` of one feed.

    Args:
        src: Feed folder or zip.
        coord_decimals: Decimals kept on coordinates (4 = about 10 m).
    """
    stops: Set[Tuple[float, float]] = set()
    f = _open(src, "stops.txt")
    if f is not None:
        for r in csv.DictReader(f):
            try:
                stops.add((round(float(r["stop_lat"]), coord_decimals), round(float(r["stop_lon"]), coord_decimals)))
            except (KeyError, TypeError, ValueError):
                continue
    routes: Set[str] = set()
    f = _open(src, "routes.txt")
    if f is not None:
        for r in csv.DictReader(f):
            name = (r.get("route_short_name") or r.get("route_long_name") or r.get("route_id") or "").strip().lower()
            if name:
                routes.add(name)
    return stops, routes


def trips_on_date(src: Path, day) -> int:
    """Trips of a feed that run on `day` (a `datetime.date`): calendar.txt weekday + date range, then calendar_dates.txt exceptions.

    Args:
        src: Feed folder or zip.
        day: The date.

    Returns:
        Number of trips.
    """
    ymd = day.strftime("%Y%m%d"); wd = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][day.weekday()]
    active: Set[str] = set()
    f = _open(src, "calendar.txt")
    if f is not None:
        for r in csv.DictReader(f):
            if (r.get(wd) or "").strip() == "1" and (r.get("start_date") or "").strip() <= ymd <= (r.get("end_date") or "").strip():
                active.add((r.get("service_id") or "").strip())
    f = _open(src, "calendar_dates.txt")
    if f is not None:
        for r in csv.DictReader(f):
            if (r.get("date") or "").strip() == ymd:
                sid, ex = (r.get("service_id") or "").strip(), (r.get("exception_type") or "").strip()
                if ex == "1":
                    active.add(sid)
                elif ex == "2":
                    active.discard(sid)
    if not active:
        return 0
    f = _open(src, "trips.txt")
    return sum(1 for r in csv.DictReader(f) if (r.get("service_id") or "").strip() in active) if f is not None else 0


def find_duplicate_feeds(feeds: Sequence[Path], min_stop_overlap: float = 0.9, min_route_overlap: float = 0.8, service_date=None) -> Dict[Path, Path]:
    """``{duplicate feed: feed to keep}`` among `feeds`.

    Feed B duplicates A when at least `min_stop_overlap` of the smaller stop set lies at the same (~10 m) locations in the larger one
    AND at least `min_route_overlap` of the smaller route-name set occurs in the larger. The feed with more stops is kept (the first
    one on a tie), so an umbrella feed that contains a smaller operator's feed replaces it.

    With `service_date`, the copy with the most trips running that day is kept first (then the one with more stops): two copies of one
    network from different upload dates cover different timetable periods, and keeping the bigger file can keep the copy whose timetable
    has not started yet (King County Metro, 2026-10: the kept copy ran 7 stops' worth of service on the analysis date, the dropped one all of it).

    Args:
        feeds: Feed folders or zips.
        min_stop_overlap: Share of stops that must match.
        min_route_overlap: Share of route names that must match.
        service_date: Optional date: the copy with more trips that day is kept.
    """
    sig = {Path(f): feed_signature(Path(f)) for f in feeds}
    trips = {p: trips_on_date(p, service_date) for p in sig} if service_date is not None else {p: 0 for p in sig}
    order = sorted(sig, key=lambda p: (-trips[p], -len(sig[p][0]), str(p)))
    kept: List[Path] = []
    dup: Dict[Path, Path] = {}
    for f in order:
        s, r = sig[f]
        owner = None
        if s:
            for k in kept:
                ks, kr = sig[k]
                if len(s & ks) / len(s) >= min_stop_overlap and (not r or len(r & kr) / len(r) >= min_route_overlap):
                    owner = k
                    break
        if owner is None:
            kept.append(f)
        else:
            dup[f] = owner
            logger.info(f"feed {f.name} duplicates {owner.name}: dropped")
    return dup


def drop_duplicate_feeds(feeds: Sequence[Path], **kwargs) -> List[Path]:
    """`feeds` without the duplicates found by `find_duplicate_feeds`, original order kept.

    Args:
        feeds: Feed folders or zips.
        **kwargs: Passed to :func:`find_duplicate_feeds`.

    Returns:
        The kept feeds.
    """
    dup = find_duplicate_feeds(feeds, **kwargs)
    return [Path(f) for f in feeds if Path(f) not in dup]


def feed_agencies(src: Path) -> Set[str]:
    """The operators of a feed, one key per `agency.txt` row: the website's host (without "www."), else the folded agency name."""
    import re
    from urllib.parse import urlparse

    f = _open(Path(src), "agency.txt") if Path(src).exists() else None
    if f is None:
        return set()
    out = set()
    with f:
        for row in csv.DictReader(f):
            host = urlparse((row.get("agency_url") or "").strip().lower()).netloc.removeprefix("www.")
            name = re.sub(r"[^a-z0-9]+", " ", re.sub(r"\(.*?\)", " ", (row.get("agency_name") or "").lower())).strip()
            if host or name:
                out.add(host or name)
    return out


def same_operators(feed: Path, others: Sequence[Path]) -> bool:
    """Whether every operator of `feed` already runs in one of `others` (same website, or same name when there is none): another
    export of the same service, even when its stops or route names were changed enough that `find_duplicate_feeds` misses it."""
    mine = feed_agencies(feed)
    if not mine:
        return False
    theirs = set().union(*(feed_agencies(o) for o in others)) if others else set()
    return mine <= theirs
