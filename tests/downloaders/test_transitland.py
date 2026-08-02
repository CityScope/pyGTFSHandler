"""Tests for `pyGTFSHandler.downloaders.transitland`.

All HTTP calls are mocked; no real Transitland API key is used.
"""

from unittest.mock import patch

import pytest
import requests

from pyGTFSHandler.downloaders.transitland import TransitLandDownloader


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("TRANSITLAND_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TransitLandDownloader()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TRANSITLAND_API_KEY", "env-key")
    client = TransitLandDownloader()
    assert client.api_key == "env-key"


def test_lat_lon_must_be_given_together():
    client = TransitLandDownloader(api_key="k")
    with pytest.raises(ValueError):
        client.search_feeds(lat=1.0)
    with pytest.raises(ValueError):
        client.search_feeds(lon=1.0)


def test_get_sends_api_key_and_drops_none_params():
    client = TransitLandDownloader(api_key="k")
    with patch("pyGTFSHandler.downloaders.transitland.request_json") as mock_rj:
        mock_rj.return_value = {"feeds": []}
        client._get("https://transit.land/api/v2/rest/feeds", {"search": None, "spec": "gtfs"})

    args, kwargs = mock_rj.call_args
    assert kwargs["params"]["apikey"] == "k"
    assert "search" not in kwargs["params"]
    assert kwargs["params"]["spec"] == "gtfs"


def test_to_feed_metadata_skips_feed_without_static_url():
    client = TransitLandDownloader(api_key="k")
    raw = {"onestop_id": "f-abc", "urls": {}}
    assert client._to_feed_metadata(raw) is None


def test_to_feed_metadata_maps_fields():
    client = TransitLandDownloader(api_key="k")
    raw = {
        "onestop_id": "f-abc",
        "name": "Metro Feed",
        "urls": {"static_current": "https://example.com/feed.zip"},
        "operators": [{"name": "City Transit"}],
    }
    feed = client._to_feed_metadata(raw)

    assert feed.id == "f-abc"
    assert feed.name == "Metro Feed"
    assert feed.provider == "City Transit"
    assert feed.download_url == "https://example.com/feed.zip"
    assert feed.source == "transitland"
    assert feed.raw is raw


def test_to_feed_metadata_falls_back_to_onestop_id_for_name():
    client = TransitLandDownloader(api_key="k")
    raw = {"onestop_id": "f-abc", "urls": {"static_current": "https://example.com/f.zip"}}
    feed = client._to_feed_metadata(raw)
    assert feed.name == "f-abc"
    assert feed.provider is None


def test_paginated_search_follows_cursor_until_limit():
    client = TransitLandDownloader(api_key="k")
    page1 = {
        "feeds": [{"onestop_id": f"f-{i}"} for i in range(5)],
        "meta": {"next": "https://transit.land/api/v2/rest/feeds?after=5"},
    }
    page2 = {"feeds": [{"onestop_id": "f-5"}], "meta": {}}

    with patch.object(client, "_get", side_effect=[page1, page2]) as mock_get:
        results = client._paginated_search({"spec": "gtfs"}, limit=None)

    assert mock_get.call_count == 2
    assert [r["onestop_id"] for r in results] == [f"f-{i}" for i in range(6)]


def test_paginated_search_stops_early_once_limit_reached():
    client = TransitLandDownloader(api_key="k")
    page1 = {
        "feeds": [{"onestop_id": f"f-{i}"} for i in range(5)],
        "meta": {"next": "https://transit.land/api/v2/rest/feeds?after=5"},
    }

    with patch.object(client, "_get", return_value=page1) as mock_get:
        results = client._paginated_search({"spec": "gtfs"}, limit=3)

    assert mock_get.call_count == 1
    assert len(results) == 3


def test_paginated_search_handles_request_exception_gracefully():
    client = TransitLandDownloader(api_key="k")
    with patch.object(client, "_get", side_effect=requests.exceptions.RequestException("boom")):
        results = client._paginated_search({"spec": "gtfs"}, limit=None)
    assert results == []


def test_search_feeds_builds_bbox_from_aoi():
    from shapely.geometry import Polygon

    client = TransitLandDownloader(api_key="k")
    aoi = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    captured_params = {}

    def fake_paginated_search(params, limit):
        captured_params.update(params)
        return []

    with patch.object(client, "_paginated_search", side_effect=fake_paginated_search):
        client.search_feeds(aoi=aoi)

    assert captured_params["bbox"] == "0.0,0.0,1.0,1.0"


def test_search_feeds_maps_country_and_city_params():
    client = TransitLandDownloader(api_key="k")
    captured_params = {}

    def fake_paginated_search(params, limit):
        captured_params.update(params)
        return []

    with patch.object(client, "_paginated_search", side_effect=fake_paginated_search):
        client.search_feeds(country_code="ES", city="Sevilla", search="tussam")

    assert captured_params["adm0_iso"] == "ES"
    assert captured_params["city_name"] == "Sevilla"
    assert captured_params["search"] == "tussam"
    assert captured_params["spec"] == "gtfs"
