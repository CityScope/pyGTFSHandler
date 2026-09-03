# Implementing a new `pyGTFSHandler` GTFS downloader

This is a how-to for adding a new GTFS *source* (data provider / catalog
API) to `pyGTFSHandler/pyGTFSHandler/downloaders/`. It is written so a
future AI session with **no prior context on this repo** can pick it up
and implement a correct, complete, tested downloader unassisted.

It documents the *real* architecture already built and shipping in this
package (`transitland.py`, `mobility_database.py`, `busmaps.py`) — every
claim below is grounded in those files, not invented. Read this whole
document before writing any code.

---

## 1. Ready-to-use task brief

Hand this to a fresh AI session, filling in `<SOURCE_NAME>` and whatever
you know about the target API (name, base URL, docs link):

> You are adding a new GTFS downloader to `pyGTFSHandler` for the data
> source **`<SOURCE_NAME>`**. Read
> `pyGTFSHandler/IMPLEMENTING_A_DOWNLOADER.md` first — it documents the
> exact architecture you must follow and the testing/API-key conventions
> already established by the other downloaders in this package.
>
> Do NOT guess at `<SOURCE_NAME>`'s API shape from memory or from how
> other GTFS catalogs typically work. Find and read `<SOURCE_NAME>`'s
> real, current API documentation first. Then make real, live test
> requests against the real API (using a real or trial API key if one is
> required) and inspect the real JSON/XML responses before writing any
> integration code — endpoint paths, auth headers/params, field names,
> and pagination style must all come from a response you actually saw,
> not from assumption. This mirrors the mistake this project has hit
> before with fabricated data in another package (`pyCensus`): a
> plausible-looking integration built without verifying against a real
> response is worse than no integration, because it fails silently or
> produces wrong data. Concretely: BusMaps' downloader in this package was
> built only after confirming `search_feeds(country_iso="CHL")` really
> returned 14 Chilean feeds, and a downloaded feed really parsed into a
> `pyGTFSHandler.feed.Feed` with sane, geographically-correct route/stop/
> trip counts — that standard applies to `<SOURCE_NAME>` too.
>
> Follow the existing `BaseGTFSDownloader` architecture exactly — do not
> invent a different class shape, method name, or return type. Your new
> downloader must:
> - Subclass `pyGTFSHandler.downloaders.base.BaseGTFSDownloader`.
> - Implement `search_feeds(...)` returning a `List[GTFSFeedMetadata]`
>   (from `downloaders.utils.models`), mapping `<SOURCE_NAME>`'s own
>   response shape into that common shape. Do not implement
>   `download_feeds()` yourself — it's inherited for free.
> - Set `SOURCE_NAME` (a short lowercase identifier, e.g. `"busmaps"`)
>   and `API_KEY_ENV_VAR` (e.g. `"BUSMAPS_API_KEY"`) as class attributes.
> - Produce standard-format GTFS output on disk (the normal GTFS text
>   files: `agency.txt`, `routes.txt`, `stops.txt`, `stop_times.txt`,
>   `trips.txt`, `calendar.txt`/`calendar_dates.txt`, `shapes.txt`,
>   `feed_info.txt`, etc.) — you get this for free from the shared
>   `download_feeds()` loop as long as `download_url` in each
>   `GTFSFeedMetadata` points at a real GTFS static ZIP; do not
>   post-process or reshape the feed's internals.
> - Resolve its API key/secret via the existing
>   `downloaders.utils.config.get_api_key` mechanism (explicit arg → env
>   var → `api_keys.json`) — never hardcode a key, and never invent a new
>   secrets-storage mechanism.
>
> Add tests following `pyGTFSHandler/tests/downloaders/test_busmaps.py`'s
> convention exactly: mocked-HTTP unit tests for the bulk of coverage,
> plus exactly one `@pytest.mark.network`-marked test that hits the real
> API and auto-skips when no credentials are configured. Register the new
> `network` behavior is already available as a marker in `pyproject.toml`
> — do not add a second marker.
>
> When you're done: register the new class in
> `pyGTFSHandler/pyGTFSHandler/downloaders/__init__.py`, add its API key
> entry to `api_keys.json` (and validate the file is still syntactically
> valid JSON afterward — a single trailing comma silently breaks key
> lookup for *every* downloader, not just yours), and run the full test
> suite to confirm zero regressions.

---

## 2. The `BaseGTFSDownloader` interface

Defined in `pyGTFSHandler/pyGTFSHandler/downloaders/base.py`. Every real
downloader (`TransitLandDownloader`, `MobilityDatabaseDownloader`,
`BusMapsDownloader`, `spain.NAPDownloader`) subclasses it.

```python
class BaseGTFSDownloader(ABC):
    API_KEY_ENV_VAR: str = ""   # override, e.g. "NAP_API_KEY"
    SOURCE_NAME: str = ""       # override, e.g. "nap_es"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = get_api_key(self.SOURCE_NAME, api_key, self.API_KEY_ENV_VAR)

    @abstractmethod
    def search_feeds(self, *args, **kwargs) -> List[GTFSFeedMetadata]:
        raise NotImplementedError

    def download_feeds(self, feeds, download_folder, overwrite=False, unzip=True) -> List[str]:
        return download_feeds(feeds, download_folder, overwrite=overwrite, unzip=unzip)
```

What a new downloader must actually write:

- **Class attributes**: `API_KEY_ENV_VAR` and `SOURCE_NAME`. `SOURCE_NAME`
  is both the key used to look the secret up in `api_keys.json` *and* the
  value stamped into every `GTFSFeedMetadata.source` this downloader
  produces — keep it short, lowercase, stable (e.g. `"busmaps"`,
  `"transitland"`, `"mobility_database"`, `"nap_es"`).
- **`__init__`**: call `super().__init__(api_key=api_key)`, then — if the
  source requires auth (all four existing downloaders do) — check
  `self.api_key` is truthy and raise `ValueError` with a clear message if
  not (see `BusMapsDownloader.__init__`, `TransitLandDownloader.__init__`,
  `MobilityDatabaseDownloader.__init__` for the exact wording pattern).
  If the source needs extra setup beyond a static key (Mobility Database
  exchanges its refresh token for a short-lived OAuth access token at
  construction time — `_ensure_access_token()`), do that here too.
- **`search_feeds(...)`**: the only abstract method. Its *signature* is
  intentionally source-specific — Transitland's takes `aoi`/`lat`/`lon`/
  `radius`/`country_code`/`state`/`city`/`search`/`operator_onestop_id`/
  `limit`/`spec`; Mobility Database's takes `aoi`/`provider`/
  `producer_url`/`country_code`/`subdivision_name`/`municipality`/
  `bounding_filter_method`/`is_official`/pagination; BusMaps' takes just
  `country_iso`/`feed_name`/`derivative_names`. Design the new
  downloader's filters around what its real API actually supports —
  don't force it to match another downloader's parameter names if the
  underlying API doesn't have an equivalent concept. What's fixed is only
  the **return type**: `List[GTFSFeedMetadata]`.
- **`download_feeds`**: do NOT override this unless the source's download
  links need special handling (e.g. authenticated download URLs — see
  `download_feeds(..., headers=...)` support in
  `downloaders/utils/download.py`, used by Spain's NAP). The shared
  implementation streams each `GTFSFeedMetadata.download_url` to a ZIP,
  skips/overwrites existing files, and unzips — this is what turns your
  metadata list into actual GTFS files on disk.

Optional, source-specific extras seen in the existing downloaders (add
only if the source actually supports/needs them — don't add speculative
scaffolding): `download_historic_stack(...)` for sources whose catalog
keeps multiple historical dataset versions per feed (Transitland,
Mobility Database both implement this — see their modules for the
pattern of `find_*_history()` + `select_versions_covering_range` +
`historic_stack` stitching). A brand-new downloader almost certainly
does **not** need this on day one.

## 3. `GTFSFeedMetadata`

Defined in `pyGTFSHandler/pyGTFSHandler/downloaders/utils/models.py`:

```python
@dataclass
class GTFSFeedMetadata:
    id: str
    download_url: str
    name: Optional[str] = None
    provider: Optional[str] = None
    country_code: Optional[str] = None
    source: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
```

How `search_feeds()` should populate it, per the real precedent in all
three source modules:

- `id`: a stable identifier *within that source's own catalog* (BusMaps:
  `str(feed["feedId"])`, falling back to `feedName`; Transitland: the
  `onestop_id`; Mobility Database: the `mdb-NNN`-style feed id). Must be
  unique enough that `downloaders.utils.naming.build_feed_filename` (used
  by the shared `download_feeds()`) produces a distinct filename per feed.
- `download_url`: a direct, unauthenticated-or-already-credentialed URL to
  the feed's GTFS static ZIP. **If the source has no usable download URL
  for a given feed, skip that feed with a `logger.warning(...)` rather
  than emitting a `GTFSFeedMetadata` with an empty/`None` URL** — every
  existing downloader does this (see `BusMapsDownloader.search_feeds`'s
  `if derivative is None: ... continue`, `TransitLandDownloader.
  _to_feed_metadata`'s `if not download_url: ... return None`,
  `MobilityDatabaseDownloader._to_feed_metadata`'s equivalent).
- `name`: a human-readable feed name if the API provides one; fall back to
  the `id` if not (Transitland: `raw.get("name") or raw.get("onestop_id")`).
- `provider`: the operating agency/publisher name if available (not
  required — Transitland leaves it `None` when there's no operator).
- `country_code`: ISO country code if the API surfaces one per feed;
  `None` is acceptable when it doesn't (Transitland leaves this `None`
  since its `/feeds` response has no per-feed country field it maps
  cleanly).
- `source`: always `self.SOURCE_NAME` — never hardcode a string literal
  here.
- `raw`: the *entire* original API payload for that feed (and any
  sibling context needed later, e.g. BusMaps stores `{"feed":, "country":,
  "derivative":}` together) — this is the escape hatch for anything a
  caller needs beyond the common fields; never trim it down.

## 4. Output GTFS format on disk

`download_feeds()` (shared, in `downloaders/utils/download.py`) writes
each feed as `{download_folder}/{filename}.zip`, then — if `unzip=True`
(the default) — extracts it to `{download_folder}/{filename}/` and
deletes the ZIP. `filename` comes from
`downloaders.utils.naming.build_feed_filename(feed.id, feed.name, feed.provider)`.

The extracted folder must be a **standard GTFS static feed**: plain-text
CSV files directly inside that folder, not nested further. Real example
on disk, Concepción's Gran Concepción feed downloaded via
`BusMapsDownloader` this session
(`TransitLOSStudies/CS_transitLOS/concepcion/gtfs/gran_concepcion/`):

```
agency.txt
calendar.txt
feed_info.txt
routes.txt
shapes.txt
stops.txt
stop_times.txt
trips.txt
```

(`calendar_dates.txt`, `frequencies.txt`, `transfers.txt`, `fare_*.txt`
etc. also show up in other real feeds when the source provides them —
whatever GTFS files the upstream ZIP legitimately contains is what should
end up on disk; **do not** filter, rename, or restructure files inside
the feed.) This is what lets Boston's 60-feed re-download and
Concepción's feed both load cleanly through `pyGTFSHandler.feed.Feed(...)`
with no code changes — the downloader's only job is to get a genuine
GTFS ZIP (whatever internal shape the source publishes) onto disk and
unzipped; `pyGTFSHandler`'s feed-loading code (`pyGTFSHandler/feed.py`)
is what actually parses GTFS structure, and it expects exactly this
standard layout. If a new source's "GTFS" isn't already packaged this
way (e.g. it hands back individual files, or a non-standard archive
layout), the downloader is responsible for reassembling a conformant
GTFS ZIP before `download_feeds` unzips it — don't push a translation
problem downstream into feed-loading code.

## 5. Registering the downloader

Add to `pyGTFSHandler/pyGTFSHandler/downloaders/__init__.py`:

```python
from .newsource import NewSourceDownloader

__all__ = [
    "BaseGTFSDownloader",
    "BusMapsDownloader",
    "MobilityDatabaseDownloader",
    "NewSourceDownloader",   # add, alphabetized with the rest
    "TransitLandDownloader",
    "spain",
]
```

Also add a one-line bullet to the module docstring's "Available
downloaders" list (mirror the existing bullets for `busmaps`/
`transitland`/`mobility_database`), stating what the source covers and
its main endpoint, so future readers of `__init__.py` alone get an
accurate map of what's available — don't skip this, it's the first place
a future session will look.

If the source is inherently country/region-specific rather than global
(like Spain's NAP), follow the `spain/` package pattern instead of a flat
module: put it under `downloaders/<country_or_region>/`, with its own
`__init__.py` exposing the downloader class(es), and any source-specific
helpers in `downloaders/<country_or_region>/utils/` rather than polluting
the shared `downloaders/utils/`.

## 6. API keys: `api_keys.json`

Resolution order, implemented in
`pyGTFSHandler/pyGTFSHandler/downloaders/utils/config.py`'s
`get_api_key()`: explicit `api_key` constructor arg → the
`API_KEY_ENV_VAR` environment variable → the `SOURCE_NAME` entry of a
local JSON file. That file is looked up at (in priority order):
`$PYGTFSHANDLER_API_KEYS_FILE` if set, `api_keys.json` in the CWD or any
parent directory, then `~/.pygtfshandler/api_keys.json`. The real
`pyGTFSHandler/api_keys.json` at the repo root is gitignored; it currently
has one entry per registered source, keyed by each downloader's
`SOURCE_NAME` (confirmed: `mobility_database`, `transitland`, `nap_es`,
`busmaps`).

To add a new source's key:

1. Open `pyGTFSHandler/api_keys.json`.
2. Add `"newsource": "<the real key value>"` as a new entry, keeping it
   valid JSON — i.e. every entry except the last needs a trailing comma,
   and the *last* entry must NOT have one.
3. **Validate the file after editing it.** This session hit exactly this
   bug: a trailing comma left after the BusMaps entry was added made the
   whole file invalid JSON, which made `get_api_key()`'s file-based
   lookup silently fail for *every* source, not just BusMaps (`json.load`
   raises `json.JSONDecodeError`, which `_load_api_keys_file()` catches
   and only logs a warning for — so the failure mode is "my key isn't
   found" with no obvious error, not a crash). After any edit, run:
   ```bash
   python -c "import json; json.load(open('api_keys.json'))"
   ```
   from the `pyGTFSHandler/` directory and confirm it prints nothing /
   doesn't raise.
4. Never print, log, or commit the literal key value anywhere outside
   this file.

## 7. Rate limits

Respect whatever the real target API documents — don't assume it's
unlimited or match another source's limits. Document the real limits in
the new module's docstring, the way `busmaps.py` does:

> Rate limits (free tier, per the API docs): `getGtfsFeedsDownloads` is
> capped at 50 requests/day and 1,000/month.

Design `search_feeds()`'s defaults to respect this: BusMaps' catalog
endpoint returns an entire country's (or, with no filter, the *entire
world's*) feed list in a single call, so `search_feeds()` doesn't
paginate — callers are expected to pass `country_iso` to scope requests
and stay well under 50/day. If the target API's limits are tighter or
shaped differently (per-minute throttling, concurrent-request caps,
etc.), the downloader should fail clearly when a request is rejected for
exceeding quota — surface the real HTTP status/error message via a
specific exception or a clear log line — rather than silently retrying or
hammering the endpoint. Do not implement blind retry loops against a
rate-limited endpoint.

## 8. Testing conventions

Follow `pyGTFSHandler/tests/downloaders/test_busmaps.py` exactly (12
tests there, one network-marked):

- **Mocked-HTTP unit tests** (the bulk of coverage) — patch the
  downloader's low-level request method (e.g.
  `patch("pyGTFSHandler.downloaders.busmaps.request_json")` or
  `patch.object(client, "_get", return_value=...)`), and cover:
  - constructor requires an API key (`pytest.raises(ValueError)` when no
    key is resolvable — use `monkeypatch.delenv` + a redirected
    `PYGTFSHANDLER_API_KEYS_FILE` pointed at a nonexistent path, per
    `test_requires_api_key`).
  - constructor picks up the key from the env var
    (`monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")`).
  - request-building details specific to the source's auth style
    (headers, query params — whatever the real API needs).
  - `search_feeds()` against a **realistic sample response shape** (copy
    the actual JSON structure you saw in your real verification request,
    trimmed to one or two feeds) — assert the resulting `GTFSFeedMetadata`
    fields are correctly mapped.
  - `search_feeds()` behavior on edge cases: a feed with no usable
    download URL is skipped (not error), an empty response returns `[]`.
- **Exactly one real-network test**, marked `@pytest.mark.network` and
  guarded to skip when no credentials exist:
  ```python
  _real_api_key = get_api_key("newsource", None, "NEWSOURCE_API_KEY")

  @pytest.mark.network
  @pytest.mark.skipif(_real_api_key is None, reason="No NewSource API key configured.")
  def test_real_search_feeds_finds_something():
      client = NewSourceDownloader(api_key=_real_api_key)
      ...
  ```
  Note the key is resolved once at **module collection time** (outside
  any test function) and passed explicitly to the constructor — this
  matters because `tests/downloaders/conftest.py`'s autouse
  `_isolate_downloader_secrets` fixture deliberately blanks every known
  `*_API_KEY_ENV_VAR` and chdirs into an empty temp directory before
  *every* test in this package, precisely so a "no key" test can never
  accidentally become a live call. Add your new env var name to that
  fixture's `_API_KEY_ENV_VARS` list in
  `pyGTFSHandler/tests/downloaders/conftest.py` — if you don't, your
  mocked tests risk picking up a real ambient key on a developer's
  machine.
  Also handle real transient network failures gracefully inside the test
  itself (`except requests.exceptions.RequestException: pytest.skip(...)`,
  as `test_real_search_feeds_finds_gran_concepcion` does) — the network
  test should fail on a real integration bug, not on the API being
  briefly unreachable.
- **No new pytest marker needed.** `network` (and `slow`) are already
  registered in `pyGTFSHandler/pyproject.toml`:
  ```toml
  markers = [
      "slow: exercises the full real-world Sevilla feeds; skip with -m 'not slow' for a fast run",
      "network: hits a real external API; skip with -m 'not network' for a fully offline run",
  ]
  ```
  Reuse `network` as-is.
- After adding tests, run the **full** suite (not just the new file) to
  confirm zero regressions, e.g. `pytest pyGTFSHandler/tests -q`, and
  separately `pytest pyGTFSHandler/tests -m network -q` if you have a
  live key to confirm the real-network test passes too.

## 9. How a downloader gets used by a city's pipeline

`CS_transitLOS/<city>/run.py` files do **not** call a downloader
directly as part of the normal pipeline run — GTFS acquisition is a
separate, manual step that produces the `<city>/gtfs/` directory ahead of
time, and `run.py`'s `download_and_prepare_stops` (or equivalent) simply
reads whatever feeds already exist under `<city>/gtfs/`. Concretely, per
this session's real work:

- **Concepción**: `BusMapsDownloader().search_feeds(country_iso="CHL")`
  was run ad hoc (not from `run.py`, which doesn't exist yet for this
  city) to locate the `"gran-concepcion"` feed (BusMaps feedId 101279),
  then `download_feeds([...], "concepcion/gtfs/gran_concepcion")` fetched
  and unzipped it directly into the city's `gtfs/` tree.
- **Boston**: re-downloaded via
  `MobilityDatabaseDownloader().search_feeds(country_code="US",
  subdivision_name="Massachusetts")` (60 feeds, replacing an
  under-scoped 17-feed set from a tighter AOI-bbox search), then
  `download_feeds(...)` into `boston/gtfs/`, with the old set preserved
  at `boston/gtfs_backup_pre_ma_search/` rather than deleted.

So: a new downloader's job stops at producing a correct `<city>/gtfs/`
directory (or `<city>/gtfs/<feed_name>/` per-feed subdirectories) on
disk; wiring it into a specific city's automated pipeline (if that's ever
done) is a separate, later task and out of scope for "implement a
downloader."

## 10. Verify before you code — non-negotiable

This project has already been burned once (in the sibling `pyCensus`
package) by an integration built from assumed-plausible API shapes rather
than a real, inspected response. Do not repeat that here.

Before writing `search_feeds()`'s parsing logic:

1. Find the target source's real, current API documentation (not a
   memory of "how these APIs usually work"). If it exposes an MCP server
   or interactive docs page, use those to see live example responses.
2. Make a real HTTP request against the real endpoint, with a real (or
   real trial) credential, and print/inspect the actual JSON/XML you get
   back. Confirm: the real base URL and endpoint path, the real
   authentication mechanism (header vs. query param vs. OAuth — these
   differ across the three existing downloaders and cannot be guessed;
   BusMaps uses `capi-key`/`capi-host` headers, Transitland uses an
   `apikey` query param, Mobility Database exchanges a refresh token for
   a bearer token), the real field names for feed id / name / provider /
   country / download URL, and the real pagination style if any.
3. Only after that, write `GTFSFeedMetadata`-mapping code against the
   *actual* observed shape, and copy a trimmed real sample response into
   your mocked unit tests (see §8).
4. Before considering the downloader done, actually download one real
   feed end-to-end and load it through `pyGTFSHandler.feed.Feed(...)`,
   confirming it parses with no structural errors and that route/stop/
   trip counts and stop coordinates look geographically sane for that
   feed (not just "some rows exist"). This is exactly what was done for
   BusMaps this session: `search_feeds(country_iso="CHL")` was confirmed
   to return 14 real Chilean feeds; the Gran Concepción feed was then
   downloaded and confirmed to contain 133 routes / 1,982 stops / 22,306
   trips with stop coordinates spanning lat -37.02..-36.64 / lon
   -73.17..-72.92 — a real bounding box matching Gran Concepción's actual
   geography, not a plausible-looking but fabricated result. Treat that
   level of verification as the bar for "the downloader is done," not
   "the code compiles and the mocked tests pass."
