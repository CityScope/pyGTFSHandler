# -*- coding: utf-8 -*-
"""GTFS feeds of any area from the Mobility Database's public catalog file -- no account or API key needed.

The Mobility Database publishes its whole catalog as one CSV (`feeds_v2.csv`, about 3 MB) with each feed's bounding box and a stable
link to its latest copy (`files.mobilitydatabase.org/<id>/latest.zip`). This client reads that file instead of the API, so it works
without the refresh token `MobilityDatabaseDownloader` needs:

    from pyGTFSHandler.downloaders import MobilityDatabaseCatalog
    cat = MobilityDatabaseCatalog()
    feeds = cat.search_feeds(aoi)                     # active GTFS feeds whose bounding box overlaps the area
    paths = cat.download_feeds(feeds, "gtfs/")        # one unzipped folder per feed

The catalog is cached (`cache_dir`, default the pyGTFSHandler cache) and refreshed after `max_age_days`.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import requests

from .base import BaseGTFSDownloader
from .utils.aoi import AOIType, bbox_from_aoi
from .utils.models import GTFSFeedMetadata

CATALOG_URL = "https://files.mobilitydatabase.org/feeds_v2.csv"


def _default_cache_dir() -> Path:
    return Path(os.environ.get("PYGTFSHANDLER_CACHE_DIR") or Path(tempfile.gettempdir()) / "pygtfshandler-cache") / "mobility_database"


class MobilityDatabaseCatalog(BaseGTFSDownloader):
    """Search and download feeds from the Mobility Database catalog file (no API key)."""

    SOURCE_NAME = "mobility_database_catalog"
    API_KEY_ENV_VAR = ""

    def __init__(self, cache_dir: Optional[str] = None, max_age_days: float = 7.0):
        """Args:
            cache_dir: Folder for the cached catalog (`feeds_v2.csv`). Default: `$PYGTFSHANDLER_CACHE_DIR/mobility_database`.
            max_age_days: Download the catalog again when the cached copy is older than this.
        """
        self.api_key = None
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.max_age_days = max_age_days

    def catalog(self) -> pd.DataFrame:
        """The whole catalog (one row per feed), downloaded when missing or older than `max_age_days`."""
        path = self.cache_dir / "feeds_v2.csv"
        if not path.is_file() or time.time() - path.stat().st_mtime > self.max_age_days * 86400:
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(CATALOG_URL, timeout=120)
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(path)
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def search_feeds(
        self,
        aoi: Optional[AOIType] = None,
        country_code: Optional[str] = None,
        statuses: Sequence[str] = ("active", ""),
        official_only: bool = False,
    ) -> List[GTFSFeedMetadata]:
        """GTFS feeds whose bounding box overlaps `aoi` (any polygon / GeoDataFrame / GeoSeries).

        Args:
            aoi: Area of interest. None = every feed (then filter with `country_code`).
            country_code: Two-letter country code (e.g. "US") to keep.
            statuses: Feed statuses to keep ("" = no status given). Deprecated / inactive feeds are left out by default.
            official_only: Keep only feeds the catalog marks as official.

        Returns:
            The feeds, largest overlap with the area first.
        """
        d = self.catalog()
        d = d[(d["data_type"] == "gtfs") & d["status"].isin(statuses) & (d["urls.latest"] != "") & (d["redirect.id"] == "")]
        if country_code:
            d = d[d["location.country_code"].str.upper() == country_code.upper()]
        if official_only:
            d = d[d["is_official"].str.lower() == "true"]
        box = ["location.bounding_box.minimum_longitude", "location.bounding_box.minimum_latitude",
               "location.bounding_box.maximum_longitude", "location.bounding_box.maximum_latitude"]
        b = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce") for c in box}, index=d.index, dtype=float)
        d = d[b.notna().all(axis=1)].copy()
        b = b.loc[d.index]
        if aoi is not None:
            minx, miny, maxx, maxy = bbox_from_aoi(aoi)
            ix = (b[box[2]].clip(upper=maxx) - b[box[0]].clip(lower=minx)).clip(lower=0)
            iy = (b[box[3]].clip(upper=maxy) - b[box[1]].clip(lower=miny)).clip(lower=0)
            d["_overlap"] = ix * iy
            d = d[d["_overlap"] > 0].sort_values("_overlap", ascending=False)
            d = self._keep_overlapping_shape(d, b.loc[d.index], aoi)
        return [
            GTFSFeedMetadata(
                id=r["id"], download_url=r["urls.latest"], name=r["name"] or None, provider=r["provider"] or None,
                country_code=r["location.country_code"] or None, source=self.SOURCE_NAME,
                raw={k: v for k, v in r.items() if not k.startswith("_")},
            )
            for r in d.to_dict("records")
        ]

    @staticmethod
    def _keep_overlapping_shape(d: pd.DataFrame, b: pd.DataFrame, aoi) -> pd.DataFrame:
        """Drop feeds whose bounding box overlaps the area's bounding box but not the area itself."""
        import geopandas as gpd
        from shapely.geometry import box as _box

        if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geom = aoi.to_crs(4326).union_all()
        else:
            geom = aoi
        keep = [geom.intersects(_box(*row)) for row in b.itertuples(index=False)]
        return d[keep]
