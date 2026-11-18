# -*- coding: utf-8 -*-
"""Turn a fleet (number of buses/trains) into line headways, and headways into `frequencies.txt`.

Use this when a operator publishes how many vehicles it runs but no timetable. The method (full write-up in
`docs/methodology/new_area_gtfs_agent_guide.md, section 8`):

1. vehicles *in service* = fleet x (1 - maintenance share - spare share)         (`vehicles_in_service`)
2. one round trip takes  length x 2 / speed x (1 + layover)                       (`round_trip_minutes`)
3. headway of a line      = round trip / vehicles on that line                    (`headway_from_fleet`)
4. when several lines share one fleet, divide the vehicles among them             (`split_fleet`)
5. when a feed already has generic headways and only the *total* is known, rescale them so the vehicles they imply match
   the in-service fleet                                                            (`rescale_headways`)
6. write the result as GTFS `frequencies.txt` windows and give every line its own phase  (`frequency_rows`, `write_frequencies`, `stagger_feed`)

Pure Python (+ the standard library); nothing here needs pandas or geopandas.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: share of the fleet in the workshop at any time (scheduled + unscheduled maintenance)
MAINTENANCE_SHARE = 0.08
#: share of the fleet kept as reserve for breakdowns and peak substitutions
SPARE_SHARE = 0.09
#: extra time at the terminals, as a share of the running time (recovery, driver breaks)
LAYOVER_SHARE = 0.10
#: headways are never set below / above these (minutes) by `headway_from_fleet` unless told otherwise
MIN_HEADWAY_MIN = 2.0
MAX_HEADWAY_MIN = 60.0


def vehicles_in_service(fleet: float, maintenance: float = MAINTENANCE_SHARE, spare: float = SPARE_SHARE) -> float:
    """Vehicles actually running at the peak: `fleet x (1 - maintenance - spare)` (default 83% of the fleet).

    Args:
        fleet: Vehicles owned.
        maintenance: Share in maintenance.
        spare: Share kept as spares.

    Returns:
        Vehicles running at the peak.
    """
    return fleet * (1.0 - maintenance - spare)


def round_trip_minutes(one_way_km: float, speed_kmh: float, layover: float = LAYOVER_SHARE, one_way_only: bool = False) -> float:
    """Minutes for a vehicle to go there and back (or around a loop, `one_way_only=True`) including the terminal layover.

    `speed_kmh` is the *commercial* speed (stops and traffic included), not the top speed.

    Args:
        one_way_km: Line length one way.
        speed_kmh: Commercial speed.
        layover: Terminal layover as a share of running time.
        one_way_only: A loop line (one leg only).

    Returns:
        Round trip in minutes.
    """
    legs = 1 if one_way_only else 2
    return legs * one_way_km / speed_kmh * 60.0 * (1.0 + layover)


def headway_from_fleet(fleet: float, one_way_km: float, speed_kmh: float, *, layover: float = LAYOVER_SHARE,
                       maintenance: float = MAINTENANCE_SHARE, spare: float = SPARE_SHARE, loop: bool = False,
                       min_headway: Optional[float] = MIN_HEADWAY_MIN, max_headway: Optional[float] = MAX_HEADWAY_MIN) -> float:
    """Headway in minutes of ONE line served by `fleet` vehicles: round-trip minutes / vehicles in service.

    The result is clamped to `[min_headway, max_headway]` (pass None to disable); the clamp is the "documented cautious value"
    of the decision order in the methodology -- say so in the feed's ASSUMED.md when it is hit.

    Args:
        fleet: Vehicles owned.
        one_way_km: Line length one way.
        speed_kmh: Commercial speed.
        layover: Terminal layover share.
        maintenance: Share in maintenance.
        spare: Share kept as spares.
        loop: A loop line.
        min_headway: Lower clamp in minutes (None: no clamp).
        max_headway: Upper clamp in minutes (None: no clamp).

    Returns:
        Headway in minutes.

    Raises:
        ValueError: If no vehicle is left in service.
    """
    n = vehicles_in_service(fleet, maintenance, spare)
    if n <= 0:
        raise ValueError("fleet leaves no vehicle in service")
    h = round_trip_minutes(one_way_km, speed_kmh, layover, one_way_only=loop) / n
    if min_headway is not None:
        h = max(h, min_headway)
    if max_headway is not None:
        h = min(h, max_headway)
    return h


def split_fleet(fleet: float, lines: Mapping[str, Mapping[str, float]], *, method: str = "equal_headway",
                maintenance: float = MAINTENANCE_SHARE, spare: float = SPARE_SHARE) -> Dict[str, Dict[str, float]]:
    """Divide ONE fleet among the lines it serves and return each line's vehicles and headway.

    `lines` = `{line: {"round_trip_min": float, "weight": float (optional, default 1)}}` (use `round_trip_minutes` for the first).

    * `method="equal_headway"` (default; right when an operator runs a branch network as one system): every line gets the same
      headway `h = sum(round trips) / vehicles_in_service`, so a long line gets more vehicles than a short one.
    * `method="weighted"`: vehicles are split in proportion to `weight` (e.g. the share of passengers, or of buses the operator says
      it assigns), then each line's headway is its round trip / its vehicles. Use it when the lines are known to differ.

    Returns `{line: {"vehicles": float, "headway_min": float}}`; `sum(vehicles) == vehicles_in_service(fleet)`.

    Args:
        fleet: Vehicles owned by the operator.
        lines: ``{line: {"round_trip_min": ..., "weight": ...}}``.
        method: ``"equal_headway"`` or ``"weighted"``.
        maintenance: Share in maintenance.
        spare: Share kept as spares.

    Raises:
        ValueError: With no line or no vehicle in service.
    """
    n = vehicles_in_service(fleet, maintenance, spare)
    if not lines or n <= 0:
        raise ValueError("need at least one line and a fleet that leaves vehicles in service")
    rt = {k: float(v["round_trip_min"]) for k, v in lines.items()}
    if method == "equal_headway":
        h = sum(rt.values()) / n
        return {k: {"vehicles": rt[k] / h, "headway_min": h} for k in lines}
    if method == "weighted":
        w = {k: float(v.get("weight", 1.0)) for k, v in lines.items()}
        tot = sum(w.values())
        out = {}
        for k in lines:
            veh = n * w[k] / tot
            out[k] = {"vehicles": veh, "headway_min": rt[k] / veh}
        return out
    raise ValueError("method must be 'equal_headway' or 'weighted'")


def share_between_services(vehicles: float, services: Mapping[str, Tuple[float, float]]) -> Dict[str, float]:
    """Vehicles of one corridor shared by several service patterns (all-stops, express, short-turn...), each running with its own
    round trip (minutes) and a relative frequency factor: `services = {name: (round_trip_min, relative_frequency)}`.
    Returns `{name: headway_min}` such that sum(round_trip / headway) == vehicles and headways are in inverse proportion to the factors.

    Args:
        vehicles: Vehicles of the corridor.
        services: ``{name: (round_trip_min, relative_frequency)}``.
    """
    denom = sum(rt * rel for rt, rel in services.values())
    h0 = denom / vehicles  # headway of a service with relative_frequency == 1
    return {k: h0 / rel for k, (rt, rel) in services.items()}


def rescale_headways(headways_min: Mapping[str, float], round_trips_min: Mapping[str, float], target_vehicles: float, *,
                     k_min: float = 0.5, k_max: float = 4.0, fixed: Iterable[str] = ()) -> Tuple[Dict[str, float], float]:
    """Scale a set of generic headways by one factor `k` so the vehicles they imply equal `target_vehicles`.

    Vehicles implied by a line = round trip / headway. Lines in `fixed` (their headway is a published fact) keep theirs and their
    vehicles are subtracted from the target first. `k` is limited to `[k_min, k_max]` (a fleet figure often includes vehicles
    outside the scored area, so do not let it compress headways by more than half). Returns `(new headways, k)`.

    Args:
        headways_min: ``{line: headway}`` to scale.
        round_trips_min: ``{line: round trip minutes}``.
        target_vehicles: Vehicles the headways must imply.
        k_min: Smallest factor.
        k_max: Largest factor.
        fixed: Lines whose headway is a published fact.
    """
    fixed = set(fixed)
    used = sum(round_trips_min[l] / headways_min[l] for l in fixed if l in headways_min)
    need = max(target_vehicles - used, 0.15 * target_vehicles)
    base = sum(round_trips_min[l] / headways_min[l] for l in headways_min if l not in fixed)
    if base <= 0:
        return dict(headways_min), 1.0
    k = min(max(base / need, k_min), k_max)  # vehicles scale as 1/k
    return {l: (h if l in fixed else h * k) for l, h in headways_min.items()}, k


# ----------------------------------------------------------------------------------------------------------------------
# frequencies.txt
# ----------------------------------------------------------------------------------------------------------------------

def hms(seconds: int) -> str:
    """Seconds as ``HH:MM:SS`` (hours may exceed 24).

    Args:
        seconds: Seconds since midnight.

    Returns:
        The time text.
    """
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def parse_hms(text: str) -> int:
    """``HH:MM:SS`` as seconds since midnight.

    Args:
        text: The time text.

    Returns:
        Seconds.
    """
    h, m, s = (int(v) for v in text.split(":"))
    return h * 3600 + m * 60 + s


def frequency_rows(trip_id: str, windows: Sequence[Tuple[str, str, float]]) -> List[Dict[str, object]]:
    """`windows = [("05:30", "09:00", 12.0), ...]` (start, end, headway in minutes) -> `frequencies.txt` rows for `trip_id`.

    The trip must exist in `trips.txt` with `stop_times.txt` times relative to its own start (first stop 00:00:00 is fine);
    `exact_times=0` means "about every N seconds", which is what a headway estimate is.

    Args:
        trip_id: Template trip of the line.
        windows: ``(start, end, headway minutes)`` per time window.

    Returns:
        Rows for ``frequencies.txt``.
    """
    rows = []
    for start, end, headway_min in windows:
        s = start if start.count(":") == 2 else start + ":00"
        e = end if end.count(":") == 2 else end + ":00"
        rows.append({"trip_id": trip_id, "start_time": s, "end_time": e, "headway_secs": int(round(headway_min * 60)), "exact_times": 0})
    return rows


def write_frequencies(folder, rows: Iterable[Mapping[str, object]]) -> Path:
    """Write (replace) `frequencies.txt` in a feed folder.

    Args:
        folder: Feed folder.
        rows: Rows from :func:`frequency_rows`.

    Returns:
        Path of the written file.
    """
    path = Path(folder) / "frequencies.txt"
    rows = list(rows)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trip_id", "start_time", "end_time", "headway_secs", "exact_times"])
        w.writeheader()
        w.writerows(rows)
    return path


def stagger_feed(folder, skip_types: Sequence[int] = (1,)) -> int:
    """Give every frequency-based trip its own phase so independent lines do not depart on the same seconds.

    Synthetic feeds start many windows on the same minute with round headways; the scoring merges departures at the same second
    into one, so lines sharing a station would be counted once. The start of each window is moved forward by a fixed
    pseudo-random offset smaller than the headway (md5 of the trip id). Route types in `skip_types` (default 1 = metro, whose lines
    are timetabled together) are left alone. Idempotent: the unstaggered file is kept as `frequencies.nostagger.txt` and a
    rebuilt `frequencies.txt` is detected by its modification time. Returns the number of windows moved.

    Args:
        folder: Feed folder with ``frequencies.txt``.
        skip_types: Route types left unchanged.
    """
    d = Path(folder)
    src, bak = d / "frequencies.txt", d / "frequencies.nostagger.txt"
    if not src.is_file():
        return 0
    if not bak.exists() or src.stat().st_mtime > bak.stat().st_mtime + 1:
        shutil.copy2(src, bak)
    rtype: Dict[str, int] = {}
    if (d / "routes.txt").is_file() and (d / "trips.txt").is_file():
        with open(d / "routes.txt", encoding="utf-8-sig", newline="") as fh:
            rt = {r["route_id"]: int(r.get("route_type") or 3) for r in csv.DictReader(fh)}
        with open(d / "trips.txt", encoding="utf-8-sig", newline="") as fh:
            rtype = {r["trip_id"]: rt.get(r["route_id"], 3) for r in csv.DictReader(fh)}
    with open(bak, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys()) if rows else ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"]
    moved = 0
    for r in rows:
        if rtype.get(r["trip_id"], 3) in skip_types:
            continue
        hw, s, e = int(float(r["headway_secs"])), parse_hms(r["start_time"]), parse_hms(r["end_time"])
        off = int(hashlib.md5(r["trip_id"].encode()).hexdigest(), 16) % max(hw, 1)
        if s + off < e - hw // 2:  # keep windows long enough for at least one more departure
            r["start_time"] = hms(s + off)
            moved += 1
    with open(src, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.utime(src, (bak.stat().st_atime, bak.stat().st_mtime))  # same mtime as the backup: the next run does not stagger twice
    return moved
