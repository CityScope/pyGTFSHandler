# -*- coding: utf-8 -*-
"""Remove trips that publish the same service twice.

Two trips are the same service when they call at the same sequence of PARENT STATIONS (consecutive repeats collapsed; a stop without a
parent counts as its own station) on at least one common day, and their times coincide:

- a scheduled trip whose first departure falls inside a frequency window of another trip with that sequence is covered by the
  frequency: the scheduled trip is deleted (even if no timetable row matches exactly);
- two overlapping frequency windows of such trips are joined into ONE window (earliest start, latest end; the headway of the longer
  window) kept on the trip with the smallest id, and the other trip's window is removed;
- two scheduled trips whose first departures differ by at most `nonfreq_tolerance_s` seconds: the later id is deleted.

With `single_day=True` (the frames hold only the trips of one selected service day) the service check is skipped altogether.
Otherwise both trips must have the SAME `service_id` (ignoring the loader's ``_file_<n>`` suffix) and that service must run on a common day
(a different service id is never a duplicate, even with the same stops and times). Day overlap is computed from `calendar` (weekday mask between start and end date) and `calendar_dates` exceptions (added days in,
removed days out); a service that cannot be resolved never counts as overlapping, so nothing is deleted on a guess.
All frames follow the `Feed` conventions: times in seconds, dates as days since 1970-01-01.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

import re

import polars as pl


def _parent_ids(stops: pl.LazyFrame) -> pl.LazyFrame:
    names = stops.collect_schema().names()
    pid = pl.col("stop_id")
    if "parent_station" in names:
        par = pl.col("parent_station").cast(pl.Utf8)
        pid = pl.when(par.is_not_null() & (par != "")).then(par).otherwise(pl.col("stop_id"))
    return stops.select("stop_id", pid.alias("_pid")).unique("stop_id")


def trip_signatures(stop_times: pl.LazyFrame, stops: pl.LazyFrame) -> pl.DataFrame:
    """``trip_id, sig, first_dep``: parent-station sequence (joined with '|') and the departure (s) at the first stop of each trip.

    Args:
        stop_times: Stop times.
        stops: Stops (for parent stations).
    """
    names = stop_times.collect_schema().names()
    dep = pl.col("departure_time")
    if "arrival_time" in names:
        dep = pl.coalesce([dep, pl.col("arrival_time")])
    seq = (
        stop_times.select("trip_id", "stop_id", "stop_sequence", dep.alias("_t"))
        .join(_parent_ids(stops), on="stop_id", how="left")
        .with_columns(pl.coalesce([pl.col("_pid"), pl.col("stop_id")]).alias("_pid"))
        .sort("trip_id", "stop_sequence")
        .with_columns(pl.col("_pid").shift(1).over("trip_id").alias("_prev"))
    )
    first = seq.group_by("trip_id", maintain_order=True).agg(pl.col("_t").first().alias("first_dep"))
    sig = (
        seq.filter(pl.col("_prev").is_null() | (pl.col("_pid") != pl.col("_prev")))
        .group_by("trip_id", maintain_order=True)
        .agg(pl.col("_pid").str.join("|").alias("sig"))
    )
    return sig.join(first, on="trip_id", how="inner").collect()


_FILE_SUFFIX = re.compile(r"_file_\d+$")


def base_service_id(service_id: str) -> str:
    """`service_id` as written in its own feed (the loader appends ``_file_<n>`` when feeds loaded together share an id).

    Args:
        service_id: Service id.

    Returns:
        The id without the suffix.
    """
    return _FILE_SUFFIX.sub("", str(service_id))


class ServiceDays:
    """Days (since 1970-01-01) each ``service_id`` runs, from `calendar` + `calendar_dates` (resolved lazily, cached)."""

    _DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

    def __init__(self, calendar: Optional[pl.LazyFrame], exceptions: Optional[pl.LazyFrame]):
        self._cal: Dict[str, Tuple[Tuple[int, ...], int, int]] = {}
        if calendar is not None:
            cols = [c for c in self._DAYS if c in calendar.collect_schema().names()]
            for r in calendar.select("service_id", *cols, "start_date", "end_date").collect().iter_rows(named=True):
                self._cal[r["service_id"]] = (tuple(int(r.get(c) or 0) for c in self._DAYS), int(r["start_date"]), int(r["end_date"]))
        self._add: Dict[str, set] = {}
        self._rem: Dict[str, set] = {}
        if exceptions is not None:
            for sid, d, t in exceptions.select("service_id", "date", "exception_type").collect().iter_rows():
                (self._add if int(t) == 1 else self._rem).setdefault(sid, set()).add(int(d))
        self._cache: Dict[str, FrozenSet[int]] = {}

    def days(self, service_id: str) -> FrozenSet[int]:
        """Days (since 1970-01-01) ``service_id`` runs.

        Args:
            service_id: Service id.

        Returns:
            Frozen set of day numbers.
        """
        if service_id not in self._cache:
            days = set()
            if service_id in self._cal:
                mask, a, b = self._cal[service_id]
                days = {d for d in range(a, b + 1) if mask[(d + 3) % 7]}   # 1970-01-01 was a Thursday
            days |= self._add.get(service_id, set())
            days -= self._rem.get(service_id, set())
            self._cache[service_id] = frozenset(days)
        return self._cache[service_id]

    def overlap(self, a: str, b: str) -> bool:
        """Whether the two services share at least one day.

        Args:
            a: First service id.
            b: Second service id.

        Returns:
            True if they share a day.
        """
        return a == b or not self.days(a).isdisjoint(self.days(b))

    def same_service(self, a: str, b: str) -> bool:
        """Two trips may only be duplicates when they carry the same `service_id` (the ``_file_<n>`` suffix aside) AND that service
        really runs on a common day in both feeds.

        Args:
            a: First service id.
            b: Second service id.

        Returns:
            True if they may be duplicates.
        """
        return base_service_id(a) == base_service_id(b) and self.overlap(a, b)


def dedupe_trips(
    trips: pl.LazyFrame,
    stop_times: pl.LazyFrame,
    stops: pl.LazyFrame,
    frequencies: Optional[pl.LazyFrame],
    calendar: Optional[pl.LazyFrame] = None,
    exceptions: Optional[pl.LazyFrame] = None,
    nonfreq_tolerance_s: int = 60,
    single_day: bool = False,
    routes: Optional[pl.LazyFrame] = None,
) -> Tuple[pl.LazyFrame, pl.LazyFrame, Optional[pl.LazyFrame], Dict[str, int]]:
    """``(trips, stop_times, frequencies, report)`` without the duplicate trips (see the module docstring).

    Args:
        trips: Trips.
        stop_times: Stop times.
        stops: Stops.
        frequencies: Frequencies, if any.
        calendar: Calendar.
        exceptions: Calendar dates.
        nonfreq_tolerance_s: Departures closer than this are the same trip.
        single_day: Every trip runs on one selected day (service ids are not compared).
        routes: Routes (the mode is part of the key).
    """
    report = {"scheduled_in_frequency": 0, "frequencies_joined": 0, "scheduled_repeated": 0}
    t = trips.select("trip_id", "service_id", *(["route_id"] if routes is not None else [])).unique("trip_id").collect()
    if routes is not None:   # a bus and a train with the same stops and times are not one service: the mode (route_type) is part of the key
        t = t.join(routes.select("route_id", "route_type").unique("route_id").collect(), on="route_id", how="left").drop("route_id")
    sigs = trip_signatures(stop_times, stops).join(t, on="trip_id", how="left")
    if "route_type" in sigs.columns:
        sigs = sigs.with_columns((pl.col("sig") + "#" + pl.col("route_type").cast(pl.Utf8).fill_null("")).alias("sig"))
    info = {r["trip_id"]: r for r in sigs.iter_rows(named=True)}
    svc = ServiceDays(calendar, exceptions)
    if single_day:   # every trip given already runs on the one selected day: service ids and calendars need no comparison
        svc.same_service = lambda a, b: True

    freq = frequencies.collect() if frequencies is not None else None
    drop_trips: set = set()
    entries: List[dict] = []
    if freq is not None and freq.height:
        entries = [dict(r, _alive=True) for r in freq.iter_rows(named=True)]

    # 1. overlapping frequency windows of one parent-station sequence -> one window
    by_sig: Dict[str, List[dict]] = {}
    for e in entries:
        i = info.get(e["trip_id"])
        if i is not None:
            by_sig.setdefault(i["sig"], []).append(e)
    for group in by_sig.values():
        if len(group) < 2:
            continue
        parent = list(range(len(group)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ea, eb = group[a], group[b]
                if ea["start_time"] < eb["end_time"] and eb["start_time"] < ea["end_time"] and svc.same_service(info[ea["trip_id"]]["service_id"], info[eb["trip_id"]]["service_id"]):
                    parent[find(a)] = find(b)
        comps: Dict[int, List[dict]] = {}
        for k, e in enumerate(group):
            comps.setdefault(find(k), []).append(e)
        for members in comps.values():
            if len(members) < 2:
                continue
            keep = min(members, key=lambda e: (e["trip_id"], e["start_time"]))
            longest = max(members, key=lambda e: e["end_time"] - e["start_time"])
            keep["start_time"] = min(e["start_time"] for e in members)
            keep["end_time"] = max(e["end_time"] for e in members)
            keep["headway_secs"] = longest["headway_secs"]
            if "n_trips" in keep:
                keep["n_trips"] = max(1, -(-(keep["end_time"] - keep["start_time"]) // keep["headway_secs"]))
            for e in members:
                if e is not keep:
                    e["_alive"] = False
            report["frequencies_joined"] += len(members) - 1

    # 2. scheduled trips covered by a frequency window of the same sequence
    freq_trips = {e["trip_id"] for e in entries}
    # the loader derives helper trips from a frequency template (`<id>_night`, `<id>_48`, ...): they belong to it, never duplicate it
    origins = {e.get("orig_trip_id") or e["trip_id"] for e in entries} | freq_trips
    helpers = {t for t in info if t.rsplit("_", 1)[0] in origins}
    alive = [e for e in entries if e["_alive"]]
    sched = [r for r in info.values() if r["trip_id"] not in freq_trips and r["trip_id"] not in helpers and r["first_dep"] is not None]
    win_by_sig: Dict[str, List[dict]] = {}
    for e in alive:
        if e["trip_id"] in info:
            win_by_sig.setdefault(info[e["trip_id"]]["sig"], []).append(e)
    for r in sched:
        for e in win_by_sig.get(r["sig"], ()):
            if e["start_time"] <= r["first_dep"] < e["end_time"] and svc.same_service(r["service_id"], info[e["trip_id"]]["service_id"]):
                drop_trips.add(r["trip_id"])
                report["scheduled_in_frequency"] += 1
                break

    # 3. scheduled trips repeated at (almost) the same time
    rest = sorted((r for r in sched if r["trip_id"] not in drop_trips), key=lambda r: (r["sig"], r["first_dep"], r["trip_id"]))
    kept: List[dict] = []
    for r in rest:
        kept = [k for k in kept if k["sig"] == r["sig"] and r["first_dep"] - k["first_dep"] <= nonfreq_tolerance_s]
        if any(svc.same_service(r["service_id"], k["service_id"]) for k in kept):
            drop_trips.add(r["trip_id"])
            report["scheduled_repeated"] += 1
        else:
            kept.append(r)

    # templates whose every window was merged away disappear too
    still = {e["trip_id"] for e in entries if e["_alive"]}
    drop_trips |= {t for t in freq_trips if t not in still}

    new_freq = frequencies
    if freq is not None and freq.height:
        cols = freq.columns
        new_freq = pl.DataFrame([{c: e[c] for c in cols} for e in entries if e["_alive"]], schema=freq.schema).lazy() if still else freq.clear().lazy()
    if not drop_trips:
        return trips, stop_times, new_freq, report
    gone = sorted(drop_trips)
    return (
        trips.filter(~pl.col("trip_id").is_in(gone)),
        stop_times.filter(~pl.col("trip_id").is_in(gone)),
        new_freq,
        report,
    )
