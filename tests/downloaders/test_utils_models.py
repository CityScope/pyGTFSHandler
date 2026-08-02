"""Tests for `pyGTFSHandler.downloaders.utils.models.GTFSFeedMetadata`."""

from pyGTFSHandler.downloaders.utils.models import GTFSFeedMetadata


def test_defaults():
    feed = GTFSFeedMetadata(id="1", download_url="https://example.com/1.zip")
    assert feed.name is None
    assert feed.provider is None
    assert feed.country_code is None
    assert feed.source == ""
    assert feed.raw == {}


def test_raw_default_is_not_shared_between_instances():
    a = GTFSFeedMetadata(id="1", download_url="u")
    b = GTFSFeedMetadata(id="2", download_url="u")
    a.raw["x"] = 1
    assert b.raw == {}
