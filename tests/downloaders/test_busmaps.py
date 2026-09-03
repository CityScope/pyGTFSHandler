"""Tests for `pyGTFSHandler.downloaders.busmaps`.

Most tests mock all HTTP calls; no real BusMaps API key is used. One test
is marked `network` and hits the real BusMaps API using whatever key is
resolved via the normal `get_api_key` lookup (env var or local
`api_keys.json`) -- it's skipped automatically if no key is configured.
"""

from unittest.mock import patch

import pytest
import requests

from pyGTFSHandler.downloaders.busmaps import BusMapsDownloader
from pyGTFSHandler.downloaders.utils.config import get_api_key


def test_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("BUSMAPS_API_KEY", raising=False)
    monkeypatch.setenv("PYGTFSHANDLER_API_KEYS_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError):
        BusMapsDownloader()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("BUSMAPS_API_KEY", "env-key")
    client = BusMapsDownloader()
    assert client.api_key == "env-key"


def test_headers_use_bearer_key_and_default_host():
    client = BusMapsDownloader(api_key="k")
    headers = client._headers()
    assert headers["capi-key"] == "Bearer k"
    assert headers["capi-host"] == "busmaps.com"


def test_headers_respect_custom_capi_host():
    client = BusMapsDownloader(api_key="k", capi_host="wikiroutes.info")
    assert client._headers()["capi-host"] == "wikiroutes.info"


def test_get_drops_none_params_and_sends_headers():
    client = BusMapsDownloader(api_key="k")
    with patch("pyGTFSHandler.downloaders.busmaps.request_json") as mock_rj:
        mock_rj.return_value = []
        client._get({"countryIso": "CHL", "feedName": None})

    args, kwargs = mock_rj.call_args
    assert kwargs["params"] == {"countryIso": "CHL"}
    assert kwargs["headers"]["capi-key"] == "Bearer k"
    assert kwargs["headers"]["capi-host"] == "busmaps.com"
    assert args[0] == "GET"
    assert args[1] == BusMapsDownloader.GTFS_FEEDS_ENDPOINT


def test_pick_derivative_prefers_improved_over_source():
    derivatives = [
        {"fileName": "source_gtfs.zip", "path": "https://x/source.zip"},
        {"fileName": "improved_gtfs.zip", "path": "https://x/improved.zip"},
    ]
    picked = BusMapsDownloader._pick_derivative(derivatives, ("improved_gtfs.zip", "source_gtfs.zip"))
    assert picked["path"] == "https://x/improved.zip"


def test_pick_derivative_falls_back_to_source_when_no_improved():
    derivatives = [{"fileName": "source_gtfs.zip", "path": "https://x/source.zip"}]
    picked = BusMapsDownloader._pick_derivative(derivatives, ("improved_gtfs.zip", "source_gtfs.zip"))
    assert picked["path"] == "https://x/source.zip"


def test_pick_derivative_returns_none_when_no_match():
    derivatives = [{"fileName": "geojson_data", "path": "https://x/geo.zip"}]
    assert BusMapsDownloader._pick_derivative(derivatives, ("improved_gtfs.zip",)) is None


def test_search_feeds_parses_sample_response_shape():
    client = BusMapsDownloader(api_key="k")
    sample = [
        {
            "countryUrl": "chl",
            "countryRegion": "South America",
            "countryName": "Chile",
            "countryIso": "CHL",
            "feeds": [
                {
                    "feedId": 101279,
                    "feedName": "gran-concepcion",
                    "derivatives": [
                        {
                            "type": "processed_data",
                            "fileName": "improved_gtfs.zip",
                            "path": "https://example.com/gran-concepcion.zip",
                            "downloadFileName": "improved-gtfs-gran-concepcion.zip",
                        },
                        {
                            "type": "source_data",
                            "fileName": "source_gtfs.zip",
                            "path": "https://example.com/source.zip",
                        },
                    ],
                }
            ],
        }
    ]
    with patch.object(client, "_get", return_value=sample) as mock_get:
        feeds = client.search_feeds(country_iso="CHL")

    mock_get.assert_called_once_with({"countryIso": "CHL", "feedName": None})
    assert len(feeds) == 1
    feed = feeds[0]
    assert feed.id == "101279"
    assert feed.name == "gran-concepcion"
    assert feed.provider == "Chile"
    assert feed.country_code == "CHL"
    assert feed.source == "busmaps"
    assert feed.download_url == "https://example.com/gran-concepcion.zip"
    assert feed.raw["derivative"]["fileName"] == "improved_gtfs.zip"


def test_search_feeds_skips_feed_with_no_usable_derivative():
    client = BusMapsDownloader(api_key="k")
    sample = [
        {
            "countryIso": "CHL",
            "countryName": "Chile",
            "feeds": [{"feedId": 1, "feedName": "no-gtfs", "derivatives": [{"fileName": "geojson_data"}]}],
        }
    ]
    with patch.object(client, "_get", return_value=sample):
        feeds = client.search_feeds(country_iso="CHL")
    assert feeds == []


def test_search_feeds_handles_empty_response():
    client = BusMapsDownloader(api_key="k")
    with patch.object(client, "_get", return_value=[]):
        assert client.search_feeds(country_iso="ZZZ") == []


# ---------------------------------------------------------------------
# Real network test -- skipped unless a BusMaps API key is configured.
# ---------------------------------------------------------------------

_real_api_key = get_api_key("busmaps", None, "BUSMAPS_API_KEY")


@pytest.mark.network
@pytest.mark.skipif(_real_api_key is None, reason="No BusMaps API key configured.")
def test_real_search_feeds_finds_gran_concepcion():
    # `tests/downloaders/conftest.py`'s autouse `_isolate_downloader_secrets`
    # fixture deliberately blanks the env var and chdirs away from the
    # local `api_keys.json` for every test in this package (so a "no key"
    # test never accidentally becomes a live call) -- so the key resolved
    # once at collection time (`_real_api_key`, above) must be passed
    # explicitly here rather than left for `BusMapsDownloader()` to
    # re-discover.
    client = BusMapsDownloader(api_key=_real_api_key)
    try:
        feeds = client.search_feeds(country_iso="CHL", feed_name="gran-concepcion")
    except requests.exceptions.RequestException as e:
        pytest.skip(f"BusMaps API unreachable: {e}")

    assert len(feeds) == 1
    feed = feeds[0]
    assert feed.name == "gran-concepcion"
    assert feed.source == "busmaps"
    assert feed.download_url.startswith("https://")
