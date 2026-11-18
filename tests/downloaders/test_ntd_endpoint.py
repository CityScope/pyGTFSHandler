"""NTDDownloader pages the SODA2 endpoint without an app token and the v3 endpoint with one."""

from pyGTFSHandler.downloaders.usa import ntd


def test_endpoint_by_token(monkeypatch):
    calls = []

    def fake(method, url, params=None, **kw):
        calls.append((url, dict(params)))
        return []

    monkeypatch.setattr(ntd, "request_json", fake)
    monkeypatch.delenv("SOCRATA_APP_TOKEN", raising=False)
    d = ntd.NTDDownloader.__new__(ntd.NTDDownloader)
    d.api_key = None
    d._get_page(2)
    d.api_key = "tok"
    d._get_page(1)
    assert calls[0][0] == ntd.NTDDownloader.RESOURCE_URL and calls[0][1]["$offset"] == ntd.NTDDownloader.PAGE_SIZE
    assert calls[1][0] == ntd.NTDDownloader.BASE_URL and calls[1][1]["$$app_token"] == "tok"
