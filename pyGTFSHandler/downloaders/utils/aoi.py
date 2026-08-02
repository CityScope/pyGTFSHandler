# -*- coding: utf-8 -*-
"""Area-of-interest (AOI) handling shared by every downloader.

Downloaders accept an AOI in whatever form is convenient for the caller --
a Shapely `Polygon`/`MultiPolygon`, or a GeoPandas `GeoDataFrame`/
`GeoSeries` with one or more geometries -- but the GTFS catalog APIs they
call (Mobility Database, Transitland, ...) only understand a plain
bounding box in WGS84 (EPSG:4326) longitude/latitude. `bbox_from_aoi`
performs that normalization once so every downloader filters spatially the
same way.

`suggest_place_filters_from_aoi` complements this for catalogs (such as
Transitland) that filter by place name/country rather than by bounding
box: it reuses the reverse-geocoding helper already available in
`pyGTFSHandler.utils.geocoding` instead of reimplementing geocoding here.
"""

from typing import Dict, List, Tuple, Union

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from ...utils import geocoding

AOIType = Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries]


def bbox_from_aoi(aoi: AOIType) -> Tuple[float, float, float, float]:
    """Compute a WGS84 bounding box from an area-of-interest object.

    Args:
        aoi: The area of interest, as a Shapely `Polygon`/`MultiPolygon`,
            or a GeoPandas `GeoDataFrame`/`GeoSeries`. GeoPandas inputs are
            reprojected to EPSG:4326 if needed.

    Returns:
        A tuple `(min_lon, min_lat, max_lon, max_lat)`.

    Raises:
        ValueError: If a GeoDataFrame/GeoSeries AOI is empty.
        TypeError: If `aoi` is not one of the supported types.
    """
    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if aoi.empty:
            raise ValueError("AOI is empty.")
        geometry = aoi.to_crs(4326).union_all() if isinstance(aoi, gpd.GeoSeries) else aoi.geometry.to_crs(4326).union_all()
    elif isinstance(aoi, (Polygon, MultiPolygon)):
        geometry = aoi
    else:
        raise TypeError(
            f"AOI must be a Polygon, MultiPolygon, GeoDataFrame, or GeoSeries; got {type(aoi)}."
        )

    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    return min_lon, min_lat, max_lon, max_lat


def suggest_place_filters_from_aoi(
    aoi: AOIType, num_points: int = 1
) -> Dict[str, List[str]]:
    """Suggest country/subdivision/municipality names covering an AOI.

    Some catalog APIs (e.g. Transitland) filter by place name rather than
    by bounding box. This reverse-geocodes one or more points within the
    AOI via `pyGTFSHandler.utils.geocoding.get_geographic_suggestions_from_aoi`
    to suggest values for such filters.

    Args:
        aoi: The area of interest, as a Shapely `Polygon`/`MultiPolygon`,
            or a GeoPandas `GeoDataFrame`/`GeoSeries`.
        num_points: Number of points sampled within the AOI for reverse
            geocoding. Use more than 1 for large or irregular AOIs.

    Returns:
        A dict with keys `"country_codes"`, `"subdivision_names"`, and
        `"municipalities"`, each mapping to a sorted list of suggested
        values.
    """
    return geocoding.get_geographic_suggestions_from_aoi(aoi, num_points=num_points)
