"""MobilityDatabaseCatalog: feeds of an area from the cached catalog file, no network."""

import geopandas as gpd
from shapely.geometry import box

from pyGTFSHandler.downloaders import MobilityDatabaseCatalog

HEADER = ("id,data_type,status,is_official,name,provider,location.country_code,urls.latest,redirect.id,"
          "location.bounding_box.minimum_latitude,location.bounding_box.maximum_latitude,"
          "location.bounding_box.minimum_longitude,location.bounding_box.maximum_longitude")
ROWS = [
    "a,gtfs,active,True,City bus,City,US,https://x/a.zip,,42.2,42.4,-71.2,-71.0",       # inside
    "b,gtfs,active,True,Far bus,Far,US,https://x/b.zip,,10,11,10,11",                    # elsewhere
    "c,gtfs,deprecated,True,Old bus,City,US,https://x/c.zip,,42.2,42.4,-71.2,-71.0",     # deprecated
    "d,gtfs,active,True,Moved,City,US,https://x/d.zip,a,42.2,42.4,-71.2,-71.0",          # redirected to another feed
    "e,gbfs,active,True,Bikes,City,US,https://x/e.zip,,42.2,42.4,-71.2,-71.0",           # not GTFS
    "f,gtfs,active,False,Regional,Region,US,https://x/f.zip,,41.0,43.0,-72.0,-70.5",     # bigger box that covers the area
]


def _catalog(tmp_path):
    (tmp_path / "feeds_v2.csv").write_text("\n".join([HEADER, *ROWS]) + "\n")
    return MobilityDatabaseCatalog(cache_dir=str(tmp_path), max_age_days=1e9)


def test_search_keeps_active_gtfs_feeds_overlapping_the_area(tmp_path):
    aoi = gpd.GeoDataFrame(geometry=[box(-71.1, 42.3, -71.05, 42.35)], crs=4326)
    feeds = _catalog(tmp_path).search_feeds(aoi)
    assert {f.id for f in feeds} == {"a", "f"}
    assert all(f.source == "mobility_database_catalog" for f in feeds)
    assert {f.download_url for f in feeds} == {"https://x/a.zip", "https://x/f.zip"}


def test_official_only_and_country(tmp_path):
    cat = _catalog(tmp_path)
    aoi = box(-71.1, 42.3, -71.05, 42.35)
    assert [f.id for f in cat.search_feeds(aoi, official_only=True)] == ["a"]
    assert cat.search_feeds(aoi, country_code="CA") == []


def test_no_key_needed(tmp_path, monkeypatch):
    monkeypatch.delenv("MOBILITY_DATABASE_REFRESH_TOKEN", raising=False)
    assert _catalog(tmp_path).api_key is None
