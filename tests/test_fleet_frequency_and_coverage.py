import csv
from pathlib import Path

import pandas as pd

from pyGTFSHandler.downloaders.utils import fleet_frequency as ff
from pyGTFSHandler.downloaders.utils import coverage_check as cc


def test_headway_from_fleet_matches_documented_example():
    # Guadalajara Line 5: 41 buses, 2 x 34.7 km at 25 km/h -> about 5.4 min
    assert abs(ff.headway_from_fleet(41, 34.7, 25) - 5.38) < 0.05


def test_split_fleet_conserves_vehicles():
    out = ff.split_fleet(100, {"a": {"round_trip_min": 60}, "b": {"round_trip_min": 120}})
    assert abs(sum(v["vehicles"] for v in out.values()) - ff.vehicles_in_service(100)) < 1e-9
    assert out["a"]["headway_min"] == out["b"]["headway_min"]
    w = ff.split_fleet(100, {"a": {"round_trip_min": 60, "weight": 3}, "b": {"round_trip_min": 60, "weight": 1}}, method="weighted")
    assert w["a"]["vehicles"] == 3 * w["b"]["vehicles"]


def test_rescale_headways_hits_target():
    hw, k = ff.rescale_headways({"a": 10, "b": 20}, {"a": 60, "b": 60}, 12)
    assert abs(sum(60 / h for h in hw.values()) - 12) < 1e-9 and k == 0.75


def _feed(tmp: Path, headway_s: int, rtype: int = 3) -> Path:
    def w(name, rows):
        with open(tmp / name, "w", newline="") as fh:
            c = csv.writer(fh)
            c.writerows(rows)
    w("routes.txt", [["route_id", "route_short_name", "route_long_name", "route_type"], ["R1", "1", "One", rtype]])
    w("trips.txt", [["route_id", "service_id", "trip_id", "direction_id"], ["R1", "S", "T1", "0"]])
    w("stop_times.txt", [["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], ["T1", "00:00:00", "00:00:00", "A", "1"], ["T1", "00:10:00", "00:10:00", "B", "2"]])
    w("calendar.txt", [["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "start_date", "end_date"], ["S", 1, 1, 1, 1, 1, 0, 0, "20200101", "20301231"]])
    ff.write_frequencies(tmp, ff.frequency_rows("T1", [("05:00", "22:00", headway_s / 60)]))
    return tmp


def test_service_table_and_flags(tmp_path):
    t = cc.flag_service_table(cc.route_service_table([_feed(tmp_path, 1200, rtype=1)]))  # metro every 20 min
    assert t.iloc[0].peak_headway_min == 20 and "slow_metro" in t.iloc[0]["issues"]
    t2 = cc.flag_service_table(cc.route_service_table([_feed(tmp_path, 300, rtype=1)]))
    assert t2.iloc[0]["issues"] == []


def test_stagger_is_idempotent(tmp_path):
    f = _feed(tmp_path, 600)
    n1 = ff.stagger_feed(f)
    first = (f / "frequencies.txt").read_text()
    ff.stagger_feed(f)
    assert n1 == 1 and (f / "frequencies.txt").read_text() == first


def test_missing_osm_routes():
    table = pd.DataFrame({"short_name": ["1"], "route_id": ["R1"], "long_name": ["One"]})
    osm = pd.DataFrame([dict(id=1, route="bus", ref="1", name="a", operator="", network="", interval="", opening_hours="", from_="", to=""),
                        dict(id=2, route="bus", ref="2", name="b", operator="", network="", interval="", opening_hours="", from_="", to="")])
    m = cc.missing_osm_routes(table, osm)
    assert list(m["key"]) == ["2"]


def test_parse_tags_and_nan():
    from pyGTFSHandler.downloaders.utils.osm_extract import parse_tags
    assert parse_tags('"ref"=>"N 22","route"=>"bus"') == {"ref": "N 22", "route": "bus"}
    assert parse_tags(float("nan")) == {} and parse_tags(None) == {}
