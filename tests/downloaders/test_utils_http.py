"""Tests for `pyGTFSHandler.downloaders.utils.http`.

Uses `unittest.mock` to stand in for `requests` responses so these tests
run without any network access or API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from pyGTFSHandler.downloaders.utils.http import request_json


def _mock_response(json_body=None, status_ok=True, json_error=False):
    response = MagicMock()
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("boom")
    if json_error:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = json_body
    return response


@patch("pyGTFSHandler.downloaders.utils.http.requests.request")
def test_request_json_returns_decoded_body(mock_request):
    mock_request.return_value = _mock_response(json_body={"hello": "world"})

    result = request_json("GET", "https://example.com/api")

    assert result == {"hello": "world"}
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["timeout"] == 30


@patch("pyGTFSHandler.downloaders.utils.http.requests.request")
def test_request_json_passes_through_headers_and_params(mock_request):
    mock_request.return_value = _mock_response(json_body=[])

    request_json(
        "GET",
        "https://example.com/api",
        headers={"Authorization": "Bearer x"},
        params={"a": "1"},
        timeout=5,
    )

    args, kwargs = mock_request.call_args
    assert args[0] == "GET"
    assert args[1] == "https://example.com/api"
    assert kwargs["headers"] == {"Authorization": "Bearer x"}
    assert kwargs["params"] == {"a": "1"}
    assert kwargs["timeout"] == 5


@patch("pyGTFSHandler.downloaders.utils.http.requests.request")
def test_request_json_raises_on_http_error(mock_request):
    error_response = MagicMock()
    error_response.text = "server error"
    http_error = requests.exceptions.HTTPError("boom")
    http_error.response = error_response

    response = MagicMock()
    response.raise_for_status.side_effect = http_error
    mock_request.return_value = response

    with pytest.raises(requests.exceptions.RequestException):
        request_json("GET", "https://example.com/api")


@patch("pyGTFSHandler.downloaders.utils.http.requests.request")
def test_request_json_raises_on_connection_error(mock_request):
    mock_request.side_effect = requests.exceptions.ConnectionError("no network")

    with pytest.raises(requests.exceptions.RequestException):
        request_json("GET", "https://example.com/api")


@patch("pyGTFSHandler.downloaders.utils.http.requests.request")
def test_request_json_raises_valueerror_on_invalid_json(mock_request):
    mock_request.return_value = _mock_response(json_error=True)

    with pytest.raises(ValueError):
        request_json("GET", "https://example.com/api")
