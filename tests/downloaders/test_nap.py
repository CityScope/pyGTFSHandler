"""Tests for `pyGTFSHandler.downloaders.spain.nap`.

All HTTP calls are mocked; no real NAP API key is used.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pyGTFSHandler.downloaders.spain.nap import NAPDownloader, _resolve_alias


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("NAP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        NAPDownloader()


def test_sets_api_key_header():
    client = NAPDownloader(api_key="secret")
    assert client.headers["ApiKey"] == "secret"


def test_resolve_alias_matches_index_and_alias():
    aliases = {"Municipio": {"3", "municipio", "municipality"}}
    assert _resolve_alias(3, aliases) == "Municipio"
    assert _resolve_alias("Municipality", aliases) == "Municipio"


def test_resolve_alias_raises_for_unknown_value():
    aliases = {"Municipio": {"3", "municipio"}}
    with pytest.raises(ValueError):
        _resolve_alias("not-a-region-type", aliases)


def _response(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else []
    resp.text = text
    return resp


def test_get_region_id_fuzzy_matches_closest_name():
    client = NAPDownloader(api_key="k")
    regions = [
        {"tipoNombre": "Municipio", "nombre": "Sevilla", "regionId": 1},
        {"tipoNombre": "Municipio", "nombre": "Madrid", "regionId": 2},
    ]
    with patch.object(client, "_get", return_value=_response(json_body=regions)):
        region_id = client.get_region_id("sevila", region_type="municipio")
    assert region_id == 1


def test_get_region_id_returns_none_on_http_error():
    client = NAPDownloader(api_key="k")
    with patch.object(client, "_get", return_value=_response(status_code=500)):
        assert client.get_region_id("Sevilla") is None


def test_get_region_id_returns_none_when_no_regions_of_type():
    client = NAPDownloader(api_key="k")
    with patch.object(client, "_get", return_value=_response(json_body=[])):
        assert client.get_region_id("Sevilla", region_type="municipio") is None


def test_get_transport_type_id_resolves_alias():
    client = NAPDownloader(api_key="k")
    types = [{"nombre": "Autobus", "tipoTransporteId": 7}]
    with patch.object(client, "_get", return_value=_response(json_body=types)):
        assert client.get_transport_type_id("bus") == 7


def test_get_transport_type_id_not_found_returns_none():
    client = NAPDownloader(api_key="k")
    with patch.object(client, "_get", return_value=_response(json_body=[])):
        assert client.get_transport_type_id("bus") is None


def test_get_file_type_id_matches_name():
    client = NAPDownloader(api_key="k")
    types = [{"nombre": "GTFS", "tipoFicheroId": 1}]
    with patch.object(client, "_get", return_value=_response(json_body=types)):
        assert client.get_file_type_id("gtfs") == 1


def test_get_organization_id_fuzzy_matches():
    client = NAPDownloader(api_key="k")
    orgs = [{"nombre": "Ayuntamiento de Sevilla", "organizacionId": 9}]
    with patch.object(client, "_get", return_value=_response(json_body=orgs)):
        org_id = client.get_organization_id("ayuntamiento sevilla")
    assert org_id == 9


def test_filter_by_dates_keeps_newest_matching_file():
    datasets = [
        {
            "ficherosDto": [
                {
                    "fechaDesde": "2024-01-01T00:00:00",
                    "fechaHasta": "2024-12-31T00:00:00",
                    "fechaActualizacion": "2024-01-01T00:00:00",
                    "id": "old",
                },
                {
                    "fechaDesde": "2024-01-01T00:00:00",
                    "fechaHasta": "2024-12-31T00:00:00",
                    "fechaActualizacion": "2024-06-01T00:00:00",
                    "id": "new",
                },
            ]
        }
    ]
    start = datetime(2024, 3, 1)
    end = datetime(2024, 3, 2)

    result = NAPDownloader._filter_by_dates(datasets, start, end, keep="newest")

    assert len(result) == 1
    assert [f["id"] for f in result[0]["ficherosDto"]] == ["new"]


def test_filter_by_dates_drops_datasets_with_no_match():
    datasets = [
        {
            "ficherosDto": [
                {
                    "fechaDesde": "2024-01-01T00:00:00",
                    "fechaHasta": "2024-01-31T00:00:00",
                    "fechaActualizacion": "2024-01-01T00:00:00",
                    "id": "old",
                }
            ]
        }
    ]
    start = datetime(2024, 3, 1)
    end = datetime(2024, 3, 2)

    result = NAPDownloader._filter_by_dates(datasets, start, end)
    assert result == []


def test_search_feeds_maps_one_feed_per_file():
    client = NAPDownloader(api_key="k")
    datasets = [
        {
            "nombre": "Consorcio Metro",
            "ficherosDto": [{"ficheroId": 111}, {"ficheroId": 222}],
        }
    ]
    with patch.object(client, "find_datasets", return_value=datasets):
        feeds = client.search_feeds(region="Sevilla")

    assert len(feeds) == 2
    assert feeds[0].id == "111"
    assert feeds[0].download_url == f"{NAPDownloader.BASE_URL}/Fichero/download/111"
    assert feeds[0].name == "Consorcio Metro"
    assert feeds[1].name == "Consorcio Metro_2"
    assert feeds[0].country_code == "ES"
    assert feeds[0].source == "nap_es"


def test_find_datasets_returns_empty_list_on_filter_error():
    client = NAPDownloader(api_key="k")
    with patch(
        "pyGTFSHandler.downloaders.spain.nap.requests.post",
        return_value=_response(status_code=500),
    ):
        assert client.find_datasets() == []


def test_find_datasets_returns_empty_when_no_files():
    client = NAPDownloader(api_key="k")
    with patch(
        "pyGTFSHandler.downloaders.spain.nap.requests.post",
        return_value=_response(json_body={"filesNum": 0}),
    ):
        assert client.find_datasets() == []


def test_download_feeds_passes_api_key_header(tmp_path):
    client = NAPDownloader(api_key="secret")
    with patch("pyGTFSHandler.downloaders.spain.nap.download_feeds") as mock_download:
        mock_download.return_value = ["/some/path"]
        result = client.download_feeds([], str(tmp_path))

    args, kwargs = mock_download.call_args
    assert kwargs["headers"] == {"ApiKey": "secret", "accept": "application/json"}
    assert result == ["/some/path"]
