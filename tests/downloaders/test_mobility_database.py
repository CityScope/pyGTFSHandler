"""Tests for `pyGTFSHandler.downloaders.mobility_database`.

No real API key is available in CI, so every HTTP call is mocked via
`unittest.mock.patch`. These tests cover: token refresh, error handling on
failed requests, multi-value/None filter preparation, and mapping raw API
feed dicts to `GTFSFeedMetadata`.
"""

from unittest.mock import patch

import pytest
import requests

from pyGTFSHandler.downloaders.mobility_database import MobilityDatabaseDownloader


def _patch_token(mock_request_json, expires_in=3600):
    mock_request_json.return_value = {"access_token": "tok", "expires_in": expires_in}


def make_client(**kwargs):
    with patch("pyGTFSHandler.downloaders.mobility_database.request_json") as mock_rj:
        _patch_token(mock_rj)
        client = MobilityDatabaseDownloader(api_key="refresh-token", **kwargs)
    return client


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MOBILITY_DATABASE_REFRESH_TOKEN", raising=False)
    with pytest.raises(ValueError):
        MobilityDatabaseDownloader()


def test_obtains_access_token_on_init():
    with patch("pyGTFSHandler.downloaders.mobility_database.request_json") as mock_rj:
        _patch_token(mock_rj)
        client = MobilityDatabaseDownloader(api_key="refresh-token")

    assert client._access_token == "tok"
    mock_rj.assert_called_once()
    args, kwargs = mock_rj.call_args
    assert args[0] == "POST"
    assert args[1] == MobilityDatabaseDownloader.TOKEN_ENDPOINT


def test_token_request_failure_raises_connection_error():
    with patch("pyGTFSHandler.downloaders.mobility_database.request_json") as mock_rj:
        mock_rj.side_effect = requests.exceptions.ConnectionError("down")
        with pytest.raises(ConnectionError):
            MobilityDatabaseDownloader(api_key="refresh-token")


def test_token_response_missing_access_token_raises_value_error():
    with patch("pyGTFSHandler.downloaders.mobility_database.request_json") as mock_rj:
        mock_rj.return_value = {"expires_in": 3600}
        with pytest.raises(ValueError):
            MobilityDatabaseDownloader(api_key="refresh-token")


def test_token_not_refreshed_when_still_valid():
    client = make_client()
    with patch("pyGTFSHandler.downloaders.mobility_database.request_json") as mock_rj:
        client._ensure_access_token()
    mock_rj.assert_not_called()


def test_prepare_list_param_single_string():
    values, has_none = MobilityDatabaseDownloader._prepare_list_param("provider", "Acme")
    assert values == ["Acme"]
    assert has_none is False


def test_prepare_list_param_list_with_none():
    values, has_none = MobilityDatabaseDownloader._prepare_list_param("provider", ["A", None, "B"])
    assert values == ["A", "B"]
    assert has_none is True


def test_prepare_list_param_none_value():
    values, has_none = MobilityDatabaseDownloader._prepare_list_param("provider", None)
    assert values is None
    assert has_none is False


def test_prepare_list_param_rejects_bad_types():
    with pytest.raises(TypeError):
        MobilityDatabaseDownloader._prepare_list_param("provider", 123)


def test_prepare_list_param_rejects_bad_item_types():
    with pytest.raises(TypeError):
        MobilityDatabaseDownloader._prepare_list_param("provider", [1, 2])


def test_to_feed_metadata_skips_feed_without_hosted_url():
    client = make_client()
    raw = {"id": "abc", "latest_dataset": {}}
    assert client._to_feed_metadata(raw) is None


def test_to_feed_metadata_maps_fields():
    client = make_client()
    raw = {
        "id": "mdb-1",
        "feed_name": "Metro",
        "provider": "City Transit",
        "latest_dataset": {"hosted_url": "https://example.com/feed.zip"},
        "locations": [{"country_code": "US"}],
    }
    feed = client._to_feed_metadata(raw)

    assert feed.id == "mdb-1"
    assert feed.name == "Metro"
    assert feed.provider == "City Transit"
    assert feed.download_url == "https://example.com/feed.zip"
    assert feed.country_code == "US"
    assert feed.source == "mobility_database"
    assert feed.raw is raw


def test_search_feeds_page_request_failure_is_skipped_not_raised():
    client = make_client()
    with patch.object(client, "_authorized_request", side_effect=requests.exceptions.RequestException("boom")):
        results = client._search_feeds_page(limit=10)
    assert results == []


def test_search_feeds_merges_and_dedupes_results():
    client = make_client()
    raw_feed = {
        "id": "mdb-1",
        "feed_name": "Metro",
        "provider": "City Transit",
        "latest_dataset": {"hosted_url": "https://example.com/feed.zip"},
        "locations": [{"country_code": "US"}],
    }
    with patch.object(client, "_authorized_request", return_value=[raw_feed]):
        feeds = client.search_feeds(country_code=["US", "CA"], limit=50)

    # Same feed id returned for both country codes; should be deduped.
    assert len(feeds) == 1
    assert feeds[0].id == "mdb-1"


def test_search_feeds_paginates_beyond_max_page():
    client = make_client()

    page1 = [
        {
            "id": f"mdb-{i}",
            "latest_dataset": {"hosted_url": f"https://example.com/{i}.zip"},
        }
        for i in range(200)
    ]
    page2 = [
        {
            "id": "mdb-200",
            "latest_dataset": {"hosted_url": "https://example.com/200.zip"},
        }
    ]

    with patch.object(client, "_search_feeds_page", side_effect=[page1, page2]) as mock_page:
        feeds = client.search_feeds(limit=201)

    assert mock_page.call_count == 2
    assert len(feeds) == 201


def test_invalid_bounding_filter_method_raises():
    client = make_client()
    from shapely.geometry import Polygon

    aoi = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    with pytest.raises(ValueError):
        client._search_feeds_page(aoi=aoi, bounding_filter_method="not_a_real_method")


def test_negative_offset_raises():
    client = make_client()
    with pytest.raises(ValueError):
        client._search_feeds_page(offset=-1)
