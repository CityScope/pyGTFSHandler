# -*- coding: utf-8 -*-
"""Read public-transport objects from a downloaded OpenStreetMap extract (`.osm.pbf`) -- no Overpass.

Overpass is meant for small queries; a whole metro area (thousands of route relations) belongs on disk. The study already downloads
Geofabrik extracts for the street network (`city_science_network/streets/<region>.osm.pbf`); this module reads the SAME files:

    from pyGTFSHandler.downloaders.utils.osm_extract import OsmData, download_extract
    pbf = download_extract("europe/andorra-latest.osm.pbf", "streets/andorra.osm.pbf")      # skipped when the file exists
    osm = OsmData(pbf, bbox=(south, west, north, east), cache="cache/osm/coverage_check")
    osm.routes        # DataFrame: id, route, ref, name, operator, network, interval, opening_hours, from_, to, minx, miny, maxx, maxy (bounds)
    osm.stations      # railway=station/halt/tram_stop, public_transport=station+train: osm_id, name, railway, station, network, lat, lon
    osm.bus_stops     # lat, lon (highway=bus_stop or public_transport=platform+bus=yes)
    osm.places        # place=city/town/suburb/district with a name: name, place, lat, lon

Implementation: `pyosmium` (`pip install osmium`), two streaming passes over the file: (1) every `type=route` relation of a public-transport mode
with its tags and members, (2) the coordinates of the member stops / ways and of the stations, bus stops and places. A relation is kept when at least one
member lies inside the bbox. (The GDAL OSM driver was tried first and silently drops most route relations -- 19 of 253 bus relations in Guadalajara --
because it only builds relations whose ways it can assemble; do not use it for this.) A country extract takes about a minute; the result for the bbox
is cached under `cache`, keyed by file size/mtime and bbox, so reruns are instant.
"""
from __future__ import annotations

import hashlib
import pickle
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

GEOFABRIK = "https://download.geofabrik.de/"
ROUTE_TYPES = ("bus", "trolleybus", "tram", "subway", "light_rail", "train", "monorail", "ferry", "share_taxi")
_KV = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')


def parse_tags(other_tags: Optional[str]) -> Dict[str, str]:
    """GDAL OSM `other_tags` hstore text (`"ref"=>"1","route"=>"bus"`) -> dict.

    Args:
        other_tags: The ``other_tags`` text of a GDAL OSM layer.

    Returns:
        ``{key: value}``.
    """
    if not isinstance(other_tags, str) or not other_tags:
        return {}
    return {k: v.replace('\\"', '"') for k, v in _KV.findall(other_tags)}


def download_extract(geofabrik_path: str, dest, chunk: int = 1 << 20) -> Path:
    """Download `https://download.geofabrik.de/<geofabrik_path>` to `dest` unless it already exists (streamed, atomic rename).

    Args:
        geofabrik_path: Path on download.geofabrik.de, e.g. ``europe/spain-latest.osm.pbf``.
        dest: Local file.
        chunk: Download chunk size in bytes.

    Returns:
        The local file.
    """
    import requests

    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(GEOFABRIK + geofabrik_path.lstrip("/"), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for part in r.iter_content(chunk):
                fh.write(part)
    tmp.replace(dest)
    return dest


class OsmData:
    """Public-transport objects of a PBF extract inside `bbox` = (south, west, north, east). See the module docstring."""

    def __init__(self, pbf, bbox: Sequence[float], cache=None):
        self.pbf, self.bbox = Path(pbf), tuple(float(x) for x in bbox)
        if not self.pbf.is_file():
            raise FileNotFoundError(f"OSM extract not found: {self.pbf} (download_extract() fetches Geofabrik files)")
        st = self.pbf.stat()
        key = hashlib.md5(f"{self.pbf.resolve()}|{st.st_size}|{st.st_mtime_ns}|{self.bbox}".encode()).hexdigest()
        cfile = Path(cache) / f"osm_{key}.pkl" if cache else None
        if cfile and cfile.is_file():
            self.routes, self.stations, self.bus_stops, self.places = pickle.loads(cfile.read_bytes())
            return
        self.routes, self.stations, self.bus_stops, self.places = self._read()
        if cfile:
            cfile.parent.mkdir(parents=True, exist_ok=True)
            cfile.write_bytes(pickle.dumps((self.routes, self.stations, self.bus_stops, self.places)))

    # -- readers ------------------------------------------------------------------------------------------------------
    def _read(self):
        import osmium

        s, w, n, e = self.bbox
        inside = lambda lat, lon: s <= lat <= n and w <= lon <= e  # noqa: E731
        # pass 1: route relations (all of them; filtered by position after pass 2)
        rels = []
        want_nodes, want_ways = set(), set()
        for o in osmium.FileProcessor(str(self.pbf), osmium.osm.RELATION):
            if o.tags.get("type") == "route" and o.tags.get("route") in ROUTE_TYPES:
                nodes = [m.ref for m in o.members if m.type == "n"]
                ways = [m.ref for m in o.members if m.type == "w"]
                rels.append((o.id, {k: o.tags.get(k, "") for k in ("route", "ref", "name", "operator", "network", "interval", "opening_hours", "from", "to")}, nodes, ways))
                want_nodes.update(nodes)
                want_ways.update(ways[:3])  # a few ways are enough to place a relation that has no stop nodes
        # pass 2: coordinates + stations / bus stops / places
        coord, wcoord, st, bus, pl = {}, {}, [], [], []
        idx = tempfile.mktemp(suffix=".nodes", dir=tempfile.gettempdir())
        try:
            fp = osmium.FileProcessor(str(self.pbf), osmium.osm.NODE | osmium.osm.WAY).with_locations("sparse_file_array," + idx)
            for o in fp:
                if o.is_node():
                    if not o.location.valid():
                        continue
                    lat, lon = o.location.lat, o.location.lon
                    if o.id in want_nodes:
                        coord[o.id] = (lat, lon)
                    t = o.tags
                    if not len(t) or not inside(lat, lon):
                        continue
                    if t.get("railway") in ("station", "halt", "tram_stop") or (t.get("public_transport") == "station" and t.get("train") == "yes"):
                        st.append(dict(osm_id=o.id, name=t.get("name", ""), railway=t.get("railway", ""), station=t.get("station", ""), network=t.get("network", ""), lat=lat, lon=lon))
                    elif t.get("highway") == "bus_stop" or (t.get("public_transport") == "platform" and t.get("bus") == "yes"):
                        bus.append((lat, lon))
                    if t.get("place") in ("city", "town", "suburb", "district") and t.get("name"):
                        pl.append(dict(name=t["name"], place=t["place"], lat=lat, lon=lon))
                elif o.id in want_ways:
                    for nd in o.nodes:
                        if nd.location.valid():
                            wcoord[o.id] = (nd.location.lat, nd.location.lon)
                            break
        finally:
            Path(idx).unlink(missing_ok=True)
        rows = []
        for rid, tg, nodes, ways in rels:
            pts = [coord[x] for x in nodes if x in coord] or [wcoord[x] for x in ways[:3] if x in wcoord]
            if not any(inside(la, lo) for la, lo in pts):
                continue
            la, lo = zip(*pts)
            rows.append(dict(id=rid, route=tg["route"], ref=tg["ref"], name=tg["name"], operator=tg["operator"], network=tg["network"], interval=tg["interval"],
                             opening_hours=tg["opening_hours"], from_=tg["from"], to=tg["to"], minx=min(lo), miny=min(la), maxx=max(lo), maxy=max(la)))
        routes = pd.DataFrame(rows, columns=["id", "route", "ref", "name", "operator", "network", "interval", "opening_hours", "from_", "to", "minx", "miny", "maxx", "maxy"])
        return (routes, pd.DataFrame(st, columns=["osm_id", "name", "railway", "station", "network", "lat", "lon"]),
                np.array(bus, dtype=float).reshape(-1, 2), pd.DataFrame(pl, columns=["name", "place", "lat", "lon"]))

    # -- queries inside a cluster bbox -------------------------------------------------------------------------------
    def inside(self, bbox: Sequence[float]) -> dict:
        """Counts / names of OSM objects inside `bbox` = (south, west, north, east): bus stops, stations, places.

        Args:
            bbox: ``(south, west, north, east)``.

        Returns:
            ``osm_bus_stops``, ``osm_stations``, ``osm_route_relations``, ``osm_route_refs`` and ``places``.
        """
        s, w, n, e = bbox
        bs = self.bus_stops
        nb = int(((bs[:, 0] >= s) & (bs[:, 0] <= n) & (bs[:, 1] >= w) & (bs[:, 1] <= e)).sum()) if len(bs) else 0
        stn = self.stations[(self.stations.lat >= s) & (self.stations.lat <= n) & (self.stations.lon >= w) & (self.stations.lon <= e)]
        pl = self.places[(self.places.lat >= s) & (self.places.lat <= n) & (self.places.lon >= w) & (self.places.lon <= e)]
        r = self.routes
        rin = r[(r.maxy >= s) & (r.miny <= n) & (r.maxx >= w) & (r.minx <= e)]
        return dict(osm_bus_stops=nb, osm_stations=list(stn.name.replace("", np.nan).dropna())[:10], osm_route_relations=int(len(rin)),
                    osm_route_refs=sorted({x for x in rin.ref if x})[:25], places=list(pl.name)[:6])
