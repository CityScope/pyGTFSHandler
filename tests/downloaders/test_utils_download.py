"""Tests for `pyGTFSHandler.downloaders.utils.download.download_feeds`.

`requests.get` is mocked throughout so these tests exercise the file
handling logic (skip/overwrite, missing URLs, network errors) without any
real network access.
"""

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import requests

from pyGTFSHandler.downloaders.utils.download import download_feeds
from pyGTFSHandler.downloaders.utils.models import GTFSFeedMetadata


def _feed(feed_id="f1", name="Name", provider="Prov", download_url="https://example.com/f1.zip"):
    return GTFSFeedMetadata(id=feed_id, name=name, provider=provider, download_url=download_url)


@contextmanager
def _mock_streaming_get(content=b"PK\x03\x04zipbytes"):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [content]
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    with patch("pyGTFSHandler.downloaders.utils.download.requests.get", return_value=response) as mock_get:
        yield mock_get


def test_download_feeds_writes_zip_and_skips_unzip(tmp_path):
    with _mock_streaming_get():
        paths = download_feeds([_feed()], str(tmp_path), unzip=False)

    assert len(paths) == 1
    assert paths[0].endswith(".zip")
    assert os.path.isfile(paths[0])


def test_download_feeds_skips_missing_download_url(tmp_path):
    feed = _feed(download_url="")
    with _mock_streaming_get() as mock_get:
        paths = download_feeds([feed], str(tmp_path), unzip=False)

    assert paths == []
    mock_get.assert_not_called()


def test_download_feeds_skips_feed_with_no_usable_name(tmp_path):
    feed = _feed(feed_id="", name="", provider="")
    with _mock_streaming_get() as mock_get:
        paths = download_feeds([feed], str(tmp_path), unzip=False)

    assert paths == []
    mock_get.assert_not_called()


def test_download_feeds_skips_existing_file_by_default(tmp_path):
    with _mock_streaming_get() as mock_get:
        download_feeds([_feed()], str(tmp_path), unzip=False)
        assert mock_get.call_count == 1

        # Second call: file already exists, should skip the network request.
        paths = download_feeds([_feed()], str(tmp_path), unzip=False)
        assert mock_get.call_count == 1

    assert len(paths) == 1


def test_download_feeds_overwrite_redownloads(tmp_path):
    with _mock_streaming_get() as mock_get:
        download_feeds([_feed()], str(tmp_path), unzip=False)
        download_feeds([_feed()], str(tmp_path), unzip=False, overwrite=True)

    assert mock_get.call_count == 2


def test_download_feeds_handles_request_exception(tmp_path):
    with patch(
        "pyGTFSHandler.downloaders.utils.download.requests.get",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ):
        paths = download_feeds([_feed()], str(tmp_path), unzip=False)

    assert paths == []
    assert os.listdir(str(tmp_path)) == []


def test_download_feeds_passes_custom_headers(tmp_path):
    with _mock_streaming_get() as mock_get:
        download_feeds([_feed()], str(tmp_path), unzip=False, headers={"ApiKey": "secret"})

    _, kwargs = mock_get.call_args
    assert kwargs["headers"] == {"ApiKey": "secret"}


def test_download_feeds_unzips_and_removes_zip(tmp_path):
    import zipfile

    zip_bytes_path = tmp_path / "src.zip"
    with zipfile.ZipFile(zip_bytes_path, "w") as zf:
        zf.writestr("stops.txt", "stop_id\n1\n")
    content = zip_bytes_path.read_bytes()

    download_folder = tmp_path / "out"
    with _mock_streaming_get(content=content):
        paths = download_feeds([_feed()], str(download_folder), unzip=True)

    assert len(paths) == 1
    assert os.path.isdir(paths[0])
    assert os.path.isfile(os.path.join(paths[0], "stops.txt"))
    assert all(not f.endswith(".zip") for f in os.listdir(str(download_folder)))
