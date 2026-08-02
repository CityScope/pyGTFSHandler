"""Tests for `pyGTFSHandler.downloaders.utils.aoi`."""

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from pyGTFSHandler.downloaders.utils.aoi import bbox_from_aoi

SQUARE = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])


def test_bbox_from_polygon():
    assert bbox_from_aoi(SQUARE) == (0.0, 0.0, 1.0, 1.0)


def test_bbox_from_multipolygon():
    other = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
    mp = MultiPolygon([SQUARE, other])
    assert bbox_from_aoi(mp) == (0.0, 0.0, 3.0, 3.0)


def test_bbox_from_geodataframe():
    gdf = gpd.GeoDataFrame({"geometry": [SQUARE]}, crs="EPSG:4326")
    assert bbox_from_aoi(gdf) == (0.0, 0.0, 1.0, 1.0)


def test_bbox_from_geodataframe_reprojects():
    gdf = gpd.GeoDataFrame({"geometry": [SQUARE]}, crs="EPSG:4326").to_crs(3857)
    min_lon, min_lat, max_lon, max_lat = bbox_from_aoi(gdf)
    assert min_lon == pytest.approx(0.0, abs=1e-6)
    assert max_lon == pytest.approx(1.0, abs=1e-6)


def test_bbox_from_empty_geodataframe_raises():
    gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    with pytest.raises(ValueError):
        bbox_from_aoi(gdf)


def test_bbox_from_invalid_type_raises():
    with pytest.raises(TypeError):
        bbox_from_aoi("not an aoi")
