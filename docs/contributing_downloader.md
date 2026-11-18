# Adding a GTFS downloader

A downloader finds the GTFS feeds of an area in one catalog or portal and downloads them. Every downloader has the same
interface, so studies can use any of them in turn (most local source first).

## Where it goes

| Kind of source | Folder |
|---|---|
| Worldwide catalog (Mobility Database, Transitland) | `pyGTFSHandler/downloaders/<source>.py` |
| A country's (or region's) portal | `pyGTFSHandler/downloaders/<country>/<source>.py`, with helpers specific to that country in `<country>/utils.py` |

Helpers more than one downloader needs (area to bounding box, file names, the download loop, HTTP retries, API keys) are in
`downloaders/utils/`: use them, do not copy them.

## What it must have

Subclass `downloaders.base.BaseGTFSDownloader`:

```python
from typing import List, Optional

from ..base import BaseGTFSDownloader
from ..utils.aoi import AOIType, bbox_from_aoi
from ..utils.models import GTFSFeedMetadata


class MyPortalDownloader(BaseGTFSDownloader):
    """One line: what the portal is and its URL."""

    SOURCE_NAME = "my_portal"          # key name in the API keys file, and GTFSFeedMetadata.source
    API_KEY_ENV_VAR = "MY_PORTAL_KEY"  # "" when the portal needs no key

    def search_feeds(self, aoi: Optional[AOIType] = None, **filters) -> List[GTFSFeedMetadata]:
        """The feeds of the area (a polygon, GeoDataFrame or GeoSeries)."""
        ...
        return [GTFSFeedMetadata(id=..., download_url=..., name=..., provider=..., country_code=..., source=self.SOURCE_NAME, raw=row)]
```

* `search_feeds` returns `GTFSFeedMetadata` (`id`, `download_url`, `name`, `provider`, `country_code`, `source`, `raw` with
  the portal's own record). Accept `aoi` whenever the portal can filter by area; turn it into a bounding box with
  `bbox_from_aoi`.
* `download_feeds(feeds, download_folder)` is inherited (one unzipped folder per feed). Override it only when the portal
  needs something special (authentication in the request, several files per dataset); keep the same signature and return
  the folders.
* Keep only current feeds by default (not deprecated / expired ones) and say so in the docstring.

## API keys

The constructor resolves the key with `downloaders.utils.config.get_api_key`: the `api_key` argument, then the
`API_KEY_ENV_VAR` variable, then the keys file `~/.config/ptlos/api_keys.json` under `SOURCE_NAME`. Never put a key in the
code. Add the key name to `api_keys.example.json` with where to sign up, and to the API-keys table of
`docs/methodology/new_area_gtfs_agent_guide.md`. A downloader that needs no key sets `API_KEY_ENV_VAR = ""` and
`self.api_key = None`.

## Register, test, document

1. Export it in `downloaders/__init__.py` (worldwide) or `downloaders/<country>/__init__.py`, and list it in the module
   docstring of `downloaders/__init__.py`.
2. Tests in `tests/downloaders/test_<source>.py` that run **without network**: feed `search_feeds` a saved or hand-made
   answer (monkeypatch the HTTP call, or a small catalog file in `tmp_path`) and check the feeds it keeps and drops.
3. A docs page `docs/api/downloaders/<source>.md` (`::: pyGTFSHandler.downloaders.<path>.<Class>`) and its line in the
   `Downloaders` section of `mkdocs.yml`; `mkdocs build --strict` must pass.
4. If the portal is the official one for its country, add it to the table of sources in
   `docs/methodology/new_area_gtfs_agent_guide.md`.
