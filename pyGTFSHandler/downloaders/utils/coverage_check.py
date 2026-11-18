# -*- coding: utf-8 -*-
"""Automatic check for MISSING routes and MISSING / SUSPICIOUS frequencies in a set of GTFS feeds.

The check never edits a feed; it produces a ranked work list for a human or an AI agent (see
`docs/methodology/new_area_gtfs_agent_guide.md, section 7` for what to do with each finding). Four independent checks:

1. `route_service_table`      one row per route: weekday departures, peak / midday headway, span, per direction.
2. `flag_service_table`       rules on that table: no service, only 1-2 trips a day, rail/tram/BRT less frequent than their mode usually is.
3. `missing_osm_routes`       OpenStreetMap route relations (bus, tram, subway, ...) whose `ref` / name appears in no feed ->
                              lines that exist on the ground but not in the data.
   `missing_osm_stations`     OSM rail / tram / metro stations with no feed stop within `radius_m`.
   `osm_interval_conflicts`   OSM relations that carry an `interval` tag much shorter than the feed's headway of the same line.
4. `underserved_clusters`     clusters of populated cells with a low score (any table with lat, lon, population, score): a large cluster
                              with few feed stops is "transit missing"; one with many stops but a low score is "frequency too low".
                              `cluster_context` then reads what OSM has there (bus stops, rail, places) so the agent knows where to look.

Command line (feeds only, no score table needed; omit `--osm` to check only the feeds):

    python -m pyGTFSHandler.downloaders.utils.coverage_check --feeds gtfs/feedA gtfs/feedB --bbox S W N E --osm streets/region.osm.pbf --cache cache/osm/coverage_check --out report.json

OpenStreetMap is read from a DOWNLOADED extract (`.osm.pbf`, the same Geofabrik files the study uses for streets; see `osm_extract.py`) -- never from
Overpass, which is not meant for metro-area queries. The parsed result is cached per file and bbox, so reruns are instant.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .osm_extract import OsmData

#: GTFS route_type -> family used by the rules (extended route types are folded by `_family`)
_RAIL = {"metro": 1, "tram": 0, "rail": 2}
#: peak headway (minutes) above which a mode is *suspicious* in a city (a missing frequency, or a real low-frequency line to confirm)
SUSPICIOUS_PEAK_HEADWAY_MIN = {"metro": 10, "tram": 15, "rail": 30, "bus": 60}


def _family(route_type: int) -> str:
    t = int(route_type)
    if t == 1 or 400 <= t < 500:
        return "metro"
    if t == 0 or 900 <= t < 1000:
        return "tram"
    if t == 2 or 100 <= t < 200:
        return "rail"
    return "bus"


def _sec(t: str) -> Optional[int]:
    try:
        h, m, s = (int(v) for v in str(t).strip().split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def _read(folder: Path, name: str, **kw) -> pd.DataFrame:
    p = folder / name
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig", keep_default_na=False, **kw)


# ----------------------------------------------------------------------------------------------------------------------
# 1. service table
# ----------------------------------------------------------------------------------------------------------------------

def weekday_services(folder: Path, day: Optional[date] = None) -> Tuple[set, date]:
    """Service ids active on a representative weekday (a Wednesday inside the feed's validity, the one closest to `day` or to today).

    Args:
        folder: Unzipped GTFS folder.
        day: Preferred date; ``None`` = today.

    Returns:
        ``(active service ids, the Wednesday used)``.
    """
    cal, cd = _read(folder, "calendar.txt"), _read(folder, "calendar_dates.txt")
    if cal.empty and cd.empty:
        return set(), day or date.today()
    starts = [datetime.strptime(x, "%Y%m%d").date() for x in cal.get("start_date", pd.Series(dtype=str)) if x]
    ends = [datetime.strptime(x, "%Y%m%d").date() for x in cal.get("end_date", pd.Series(dtype=str)) if x]
    if not starts and not cd.empty:
        ds = [datetime.strptime(x, "%Y%m%d").date() for x in cd["date"]]
        starts, ends = [min(ds)], [max(ds)]
    lo, hi = min(starts), max(ends)
    ref = day or date.today()
    ref = min(max(ref, lo), hi)
    ref += timedelta(days=(2 - ref.weekday()) % 7)  # next Wednesday
    if ref > hi:
        ref -= timedelta(days=7)
    ymd = ref.strftime("%Y%m%d")
    active = set()
    if not cal.empty:
        ok = cal[(cal["start_date"] <= ymd) & (cal["end_date"] >= ymd) & (cal["wednesday"] == "1")]
        active = set(ok["service_id"])
    if not cd.empty:
        today = cd[cd["date"] == ymd]
        active |= set(today[today["exception_type"] == "1"]["service_id"])
        active -= set(today[today["exception_type"] == "2"]["service_id"])
    return active, ref


def route_service_table(feed_dirs: Iterable, day: Optional[date] = None) -> pd.DataFrame:
    """One row per (feed, route, direction) for a typical weekday.

    Columns: feed, route_id, short_name, long_name, route_type, family, direction, departures, first, last, peak_headway_min (07-09),
    midday_headway_min (10-15), has_frequencies (the route's trips are described by frequencies.txt), reference_day.
    Departures are those at the first stop of the trip; `frequencies.txt` windows are expanded (exact_times 0 or 1).

    Args:
        feed_dirs: Unzipped GTFS folders.
        day: Preferred date for the representative weekday.

    Returns:
        The table described above.
    """
    rows = []
    for fd in map(Path, feed_dirs):
        routes, trips = _read(fd, "routes.txt"), _read(fd, "trips.txt")
        if routes.empty or trips.empty:
            continue
        active, ref = weekday_services(fd, day)
        if active:
            trips = trips[trips["service_id"].isin(active)]
        st = _read(fd, "stop_times.txt", usecols=["trip_id", "stop_sequence", "departure_time", "arrival_time"])
        if not st.empty:
            st["seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
            st = st.sort_values(["trip_id", "seq"]).groupby("trip_id", as_index=False).first()
            st["dep"] = st["departure_time"].where(st["departure_time"] != "", st["arrival_time"]).map(_sec)
            first = dict(zip(st["trip_id"], st["dep"]))
        else:
            first = {}
        fq = _read(fd, "frequencies.txt")
        fq_by_trip = {}
        for r in fq.itertuples():
            fq_by_trip.setdefault(r.trip_id, []).append((_sec(r.start_time), _sec(r.end_time), int(float(r.headway_secs))))
        if "direction_id" not in trips:
            trips["direction_id"] = "0"
        rinfo = routes.set_index("route_id")
        for (rid, direction), g in trips.groupby(["route_id", trips["direction_id"].replace("", "0")]):
            deps: List[int] = []
            via_freq = False
            for tid in g["trip_id"]:
                if tid in fq_by_trip:
                    via_freq = True
                    for s, e, h in fq_by_trip[tid]:
                        if s is None or e is None or h <= 0:
                            continue
                        deps.extend(range(s, e, h) if e > s else [])
                elif first.get(tid) is not None:
                    deps.append(int(first[tid]))
            deps.sort()
            ri = rinfo.loc[rid] if rid in rinfo.index else None
            rtype = int(ri["route_type"]) if ri is not None and str(ri.get("route_type", "")).isdigit() else 3

            def headway(lo, hi):
                n = sum(1 for d in deps if lo * 3600 <= d < hi * 3600)
                return (hi - lo) * 60.0 / n if n else np.nan

            rows.append(dict(
                feed=fd.name, route_id=rid, short_name=(ri["route_short_name"] if ri is not None else ""),
                long_name=(ri["route_long_name"] if ri is not None else ""), route_type=rtype, family=_family(rtype), direction=direction,
                departures=len(deps), first=(min(deps) / 3600 if deps else np.nan), last=(max(deps) / 3600 if deps else np.nan),
                peak_headway_min=headway(7, 9), midday_headway_min=headway(10, 15), has_frequencies=via_freq, reference_day=ref.isoformat()))
    return pd.DataFrame(rows)


def flag_service_table(table: pd.DataFrame, limits: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Add a `issues` column (list of strings) to the output of `route_service_table`.

    * `no_service`        route defined, zero weekday departures (calendar mismatch, or a route listed without trips)
    * `few_departures`    <= 2 weekday departures in a direction (often a timetable that was cut, or a line only partly entered)
    * `no_peak_service`   service exists but none between 07:00 and 09:00
    * `slow_<family>`     peak headway above the limit for the mode (`SUSPICIOUS_PEAK_HEADWAY_MIN`; metro/tram/rail lines
                          rarely run that rarely in a city, so check the operator's timetable before believing it)

    Args:
        table: Output of :func:`route_service_table`.
        limits: ``{mode family: peak headway limit in minutes}`` overrides.

    Returns:
        The table with an ``issues`` column.
    """
    lim = {**SUSPICIOUS_PEAK_HEADWAY_MIN, **(limits or {})}
    flags = []
    for r in table.itertuples():
        f = []
        if r.departures == 0:
            f.append("no_service")
        elif r.departures <= 2:
            f.append("few_departures")
        elif np.isnan(r.peak_headway_min):
            f.append("no_peak_service")
        elif r.peak_headway_min > lim[r.family]:
            f.append(f"slow_{r.family}")
        flags.append(f)
    out = table.copy()
    out["issues"] = flags
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 3. OpenStreetMap
# ----------------------------------------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zÀ-ɏ一-鿿]+", "", str(s).lower())


def missing_osm_routes(table: pd.DataFrame, osm: pd.DataFrame) -> pd.DataFrame:
    """OSM route relations not found in any feed.

    A relation is *found* when its normalised `ref` equals a feed `route_short_name` (or `route_id`), or, for relations with no ref, when its
    normalised name equals a feed long name. Both directions of a line are usually two relations, so the result is grouped by
    (route, ref or name): one row per line with the number of relations, operator/network and an example OSM id. Sorted by relation count
    (a line with many relations - variants - is usually a real, important one).

    Args:
        table: Output of :func:`route_service_table`.
        osm: OSM route relations (``OsmData.routes``).

    Returns:
        One row per missing line.
    """
    if osm.empty:
        return osm
    have = set(map(_norm, table["short_name"])) | set(map(_norm, table["route_id"]))
    have_long = set(map(_norm, table["long_name"]))
    have.discard("")
    miss = []
    for r in osm.itertuples():
        ref = _norm(r.ref)
        if ref:
            if ref in have:
                continue
        else:
            nm = _norm(r.name)
            if not nm or nm in have_long or nm in have:
                continue
        miss.append(r)
    if not miss:
        return pd.DataFrame(columns=["route", "key", "relations", "operator", "network", "example_osm_id", "name"])
    df = pd.DataFrame(miss)
    df["key"] = df.apply(lambda r: r.ref or r.name, axis=1)
    out = (df.groupby(["route", "key"], as_index=False)
             .agg(relations=("id", "size"), operator=("operator", "first"), network=("network", "first"), example_osm_id=("id", "first"), name=("name", "first")))
    return out.sort_values("relations", ascending=False).reset_index(drop=True)


def _minutes(v: str) -> Optional[float]:
    """OSM `interval` value ('10', '00:10', '00:10:00') -> minutes."""
    v = str(v).strip()
    m = re.fullmatch(r"(\d+):(\d\d)(?::\d\d)?", v)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", v)
    return float(m.group(1)) if m else None


def osm_interval_conflicts(table: pd.DataFrame, osm: pd.DataFrame, factor: float = 2.0) -> pd.DataFrame:
    """Lines whose OSM `interval` tag (minutes between vehicles) is at least `factor` times shorter than the feed's peak headway.

    OSM intervals are volunteer data (often a timetable value at one hour): treat a conflict as a reason to look for the operator's
    timetable, not as the truth.

    Args:
        table: Output of :func:`route_service_table`.
        osm: OSM route relations.
        factor: How many times shorter the OSM interval must be.

    Returns:
        One row per conflicting line.
    """
    if osm.empty or table.empty:
        return pd.DataFrame()
    tt = table.assign(k=table["short_name"].map(_norm)).groupby("k")["peak_headway_min"].min()
    rows = []
    for r in osm.itertuples():
        mi, k = _minutes(r.interval), _norm(r.ref)
        if mi is None or not k or k not in tt.index or np.isnan(tt[k]):
            continue
        if tt[k] >= factor * mi:
            rows.append(dict(ref=r.ref, osm_id=r.id, osm_interval_min=mi, feed_peak_headway_min=round(float(tt[k]), 1), name=r.name))
    return pd.DataFrame(rows)


def feed_stops(feed_dirs: Iterable) -> pd.DataFrame:
    """Coordinates of every stop of the feeds.

    Args:
        feed_dirs: Unzipped GTFS folders.

    Returns:
        DataFrame with ``stop_lat`` and ``stop_lon``.
    """
    parts = []
    for fd in map(Path, feed_dirs):
        s = _read(fd, "stops.txt", usecols=["stop_lat", "stop_lon"])
        if not s.empty:
            parts.append(s.apply(pd.to_numeric, errors="coerce").dropna())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stop_lat", "stop_lon"])


def missing_osm_stations(feed_dirs: Iterable, osm: OsmData, radius_m: float = 300.0) -> pd.DataFrame:
    """OSM railway=station/halt/tram_stop nodes with no feed stop (any mode) within `radius_m`: stations missing from the data.

    Args:
        feed_dirs: Unzipped GTFS folders.
        osm: Parsed OSM extract.
        radius_m: Search radius around each OSM station.

    Returns:
        Stations with no feed stop nearby, farthest first.
    """
    st = feed_stops(feed_dirs)
    la, lo = st.stop_lat.to_numpy(), st.stop_lon.to_numpy()
    rows = []
    for x in osm.stations.itertuples():
        if len(la):
            dy = (la - x.lat) * 111_320.0
            dx = (lo - x.lon) * 111_320.0 * math.cos(math.radians(x.lat))
            d = float(np.sqrt(dx * dx + dy * dy).min())
        else:
            d = float("inf")
        if d > radius_m:
            rows.append(dict(osm_id=x.osm_id, name=x.name, railway=x.railway, station=x.station, network=x.network, lat=x.lat, lon=x.lon,
                             nearest_feed_stop_m=round(d)))
    return pd.DataFrame(rows).sort_values("nearest_feed_stop_m", ascending=False) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------------------------------------------------
# 4. underserved clusters
# ----------------------------------------------------------------------------------------------------------------------

def underserved_clusters(cells: pd.DataFrame, *, score_col: str = "score", lat_col: str = "lat", lon_col: str = "lon", pop_col: str = "population",
                         low_score: float = 25.0, min_cell_pop: float = 150.0, tile_deg: float = 0.05, min_pop: Optional[float] = None,
                         min_share: float = 0.01, top: int = 8, stops: Optional[pd.DataFrame] = None) -> List[dict]:
    """Large connected groups of populated cells with a score below `low_score`.

    `cells` is any table of populated places with a score (the city-science `results/downloads/h3_res7.parquet` once `lat`/`lon` are taken
    from the H3 centres). Cells are binned in `tile_deg` tiles (0.05 deg ~ 5 km), 8-neighbour tiles are joined, and a cluster is kept when
    its population is at least `max(min_pop or 30,000, min_share x total)`. Each cluster reports its bounding box (padded by 0.03 deg), population,
    population-weighted mean score and, when `stops` is given, how many feed stops lie inside: few stops = transit missing, many stops
    with a low score = frequencies too low (research the operator's timetable).

    Args:
        cells: Populated places with a score.
        score_col: Score column.
        lat_col: Latitude column.
        lon_col: Longitude column.
        pop_col: Population column.
        low_score: Cells scoring below this are underserved.
        min_cell_pop: Cells with fewer people are ignored.
        tile_deg: Tile size in degrees.
        min_pop: Minimum cluster population (default 30,000).
        min_share: Minimum cluster population as a share of the total.
        top: Number of clusters returned.
        stops: Optional feed stops, to count those inside each cluster.

    Returns:
        Clusters (bbox, population, mean score, stops inside), largest first.
    """
    d = cells.copy()
    d[score_col] = d[score_col].fillna(0)
    total = float(d[pop_col].sum())
    low = d[(d[score_col] < low_score) & (d[pop_col] > min_cell_pop)].copy()
    if low.empty:
        return []
    low["i"] = (low[lat_col] / tile_deg).round().astype(int)
    low["j"] = (low[lon_col] / tile_deg).round().astype(int)
    tiles = low.groupby(["i", "j"]).agg(pop=(pop_col, "sum")).reset_index()
    cell = {(r.i, r.j): r.pop for r in tiles.itertuples()}
    seen, clusters = set(), []
    for k in cell:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            a = stack.pop()
            comp.append(a)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (a[0] + di, a[1] + dj)
                    if nb in cell and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        p = sum(cell[x] for x in comp)
        if p >= max(min_pop or 30_000, min_share * total):
            clusters.append((p, comp))
    clusters.sort(reverse=True)
    pad, out = 0.03, []
    for p, comp in clusters[:top]:
        la0, la1 = min(x[0] for x in comp) * tile_deg - pad, max(x[0] for x in comp) * tile_deg + pad
        lo0, lo1 = min(x[1] for x in comp) * tile_deg - pad, max(x[1] for x in comp) * tile_deg + pad
        inside = d[(d[lat_col] >= la0) & (d[lat_col] <= la1) & (d[lon_col] >= lo0) & (d[lon_col] <= lo1)]
        rec = dict(population=int(p), mean_score=round(float(np.average(inside[score_col], weights=inside[pop_col])), 1),
                   bbox=[round(la0, 3), round(lo0, 3), round(la1, 3), round(lo1, 3)])
        if stops is not None and len(stops):
            rec["feed_stops"] = int(((stops.stop_lat >= la0) & (stops.stop_lat <= la1) & (stops.stop_lon >= lo0) & (stops.stop_lon <= lo1)).sum())
        out.append(rec)
    return out


def cluster_context(bbox: Sequence[float], osm: OsmData) -> dict:
    """What OSM knows inside a cluster's bbox: bus stops, stations, route relations (count and refs) and named places.
    Many OSM bus stops / route refs and few feed stops = a bus network missing from the feeds.

    Args:
        bbox: ``(south, west, north, east)``.
        osm: Parsed OSM extract.

    Returns:
        Counts and names, as :meth:`OsmData.inside`.
    """
    return osm.inside(bbox)


# ----------------------------------------------------------------------------------------------------------------------
# report + CLI
# ----------------------------------------------------------------------------------------------------------------------

def run_check(feed_dirs: Sequence, bbox: Sequence[float], *, osm_pbf=None, cache: Optional[Path] = None, day: Optional[date] = None) -> dict:
    """Run checks 1-3 and return a JSON-serialisable report (`summary` first, then the work lists).

    `osm_pbf` = a downloaded `.osm.pbf` covering `bbox` (south, west, north, east); without it only the feed checks run.

    Args:
        feed_dirs: Unzipped GTFS folders.
        bbox: ``(south, west, north, east)`` of the area.
        osm_pbf: Optional ``.osm.pbf`` covering the area.
        cache: Folder for the parsed-OSM cache.
        day: Preferred date for the representative weekday.

    Returns:
        The report.
    """
    table = flag_service_table(route_service_table(feed_dirs, day))
    rep: dict = {"feeds": [Path(f).name for f in feed_dirs], "routes": int(len(table))}
    flagged = table[table["issues"].map(bool)] if len(table) else table
    rep["flagged_routes"] = flagged.drop(columns=["reference_day"]).to_dict("records") if len(flagged) else []
    if osm_pbf:
        osm = OsmData(osm_pbf, bbox, cache)
        rep["osm_relations"] = int(len(osm.routes))
        rep["osm_routes_missing_from_feeds"] = missing_osm_routes(table, osm.routes).to_dict("records")
        rep["osm_interval_conflicts"] = osm_interval_conflicts(table, osm.routes).to_dict("records")
        rep["osm_stations_missing_from_feeds"] = missing_osm_stations(feed_dirs, osm).to_dict("records")
    rep["summary"] = {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items() if k not in ("feeds",)}
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``python -m pyGTFSHandler.downloaders.utils.coverage_check --feeds ... --bbox S W N E [--osm file]
    [--out report.json]``.

    Args:
        argv: Arguments; ``None`` = ``sys.argv``.

    Returns:
        Exit code 0.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--feeds", nargs="+", required=True, help="GTFS folders (unzipped)")
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("S", "W", "N", "E"))
    ap.add_argument("--osm", default=None, help="downloaded OSM extract (.osm.pbf, e.g. from Geofabrik) covering the bbox; omit to check only the feeds")
    ap.add_argument("--cache", default=None, help="folder for the parsed-OSM cache")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    a = ap.parse_args(argv)
    rep = run_check(a.feeds, a.bbox, osm_pbf=a.osm, cache=a.cache)
    text = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
