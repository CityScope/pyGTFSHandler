"""Tests for `pyGTFSHandler.downloaders.base.BaseGTFSDownloader`."""

from unittest.mock import patch

import pytest

from pyGTFSHandler.downloaders.base import BaseGTFSDownloader
from pyGTFSHandler.downloaders.utils.models import GTFSFeedMetadata


class _DummyDownloader(BaseGTFSDownloader):
    API_KEY_ENV_VAR = "DUMMY_API_KEY"
    SOURCE_NAME = "dummy"

    def search_feeds(self, *args, **kwargs):
        return []


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        BaseGTFSDownloader()


def test_api_key_from_argument():
    downloader = _DummyDownloader(api_key="explicit-key")
    assert downloader.api_key == "explicit-key"


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("DUMMY_API_KEY", "env-key")
    downloader = _DummyDownloader()
    assert downloader.api_key == "env-key"


def test_api_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv("DUMMY_API_KEY", raising=False)
    downloader = _DummyDownloader()
    assert downloader.api_key is None


def test_download_feeds_delegates_to_shared_helper(tmp_path):
    downloader = _DummyDownloader(api_key="k")
    feeds = [GTFSFeedMetadata(id="1", download_url="https://example.com/1.zip")]

    with patch("pyGTFSHandler.downloaders.base.download_feeds") as mock_download:
        mock_download.return_value = ["/some/path"]
        result = downloader.download_feeds(feeds, str(tmp_path), overwrite=True, unzip=False)

    mock_download.assert_called_once_with(feeds, str(tmp_path), overwrite=True, unzip=False)
    assert result == ["/some/path"]
