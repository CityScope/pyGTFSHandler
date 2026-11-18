# GTFS for a new city or country (guide for AI agents and people)

Goal: every regular public-transport line of the study area in GTFS, with a believable **frequency, speed and span**.
This page covers the whole path: search for official feeds, rebuild old ones, infer feeds when none exist, and add a
**downloader** to pyGTFSHandler when the data can be fetched by code. Once feeds are loaded, section 7 finds and fills the gaps and section 8 turns a fleet count into headways.

## 0. Rules

1. **Never invent.** Every stop, line and headway comes from a source you opened (URL and date recorded). An estimate is
   allowed only when it is derived from a stated source and marked as an estimate.
2. **pyGTFSHandler holds code only.** No hand-made feeds, lists, notes or calibration files under
   `pyGTFSHandler/downloaders/`. A dataset belongs in the package only when **code** builds it from a public source.
   Feeds you assemble by hand, their `ASSUMED.md` and hand-checked lists go to the study repo
   (`TransitLOSStudies/city_science_network/<city>/gtfs/assumed_*`, `usa_study/feed_lists/`); working notes stay local (`*.local.md`).
3. **One feed per service** in a city's `gtfs/` folder: no backups or old copies (the pipeline loads every folder there).
4. **API keys:** when a source needs a key you do not have, **stop and tell the user** (what to say and where the key goes:
   section 5). Never use a key found online.

## 1. Search for official feeds (in this order)

| Where | How |
|---|---|
| Mobility Database | `MobilityDatabaseCatalog().search_feeds(area)` (the public catalogue file, no key; `urls.latest` is a public copy of each feed), or `MobilityDatabaseDownloader().search_feeds(aoi=area)` (API, refresh token) |
| Transitland | `TransitLandDownloader().search_feeds(aoi=area)`; also lists operators that have no feed |
| National access points | EU NAPs (Spain `downloaders/spain/nap.py`, Germany DELFI / gtfs.de, France transport.data.gouv.fr, Italy, Netherlands OVapi...), US NTD (`downloaders/usa/ntd.py`) and state DOTs, Canada city portals, Taiwan TDX, Israel MOT, Chile DTPR, Mexico state portals |
| Operators and authorities | the operator's site ("datos abiertos", "open data", "developers", "GTFS", "horarios descarga"), the regional transport authority, tender documents |
| Archives | Wayback Machine copies of dead portals; Mobility Database and Transitland keep every past version |
| Code hosts and data repositories | GitHub / GitLab ("<city> gtfs"), Zenodo, Figshare, national open-data portals |

Search terms, in the local language first: `"<city> GTFS"`, `"<city> transit feed"`, `"datos abiertos transporte <ciudad> GTFS"`,
`"GTFS <Stadt> Download"`, `"horaires GTFS <ville>"`, `"<城市> GTFS"`, `"<city> open data bus routes timetable"`, the operator's name
with `"GTFS"`, and the national access point name (`"punto de acceso nacional transporte"`, `"NAP transport <country>"`).

For each candidate: download it, open it (stops, routes, trips, calendar), check it covers the study area and runs on a
weekday in its calendar, and note who publishes it. Several copies of one network: keep one
(`pyGTFSHandler.utils.feed_dedupe` finds duplicates; it keeps the copy that runs on the analysis date).

## 2. Rebuild a feed that existed

* **Expired feed**: an old feed is still right for lines that did not change; the scorer reads each feed on its own
  calendar (a typical weekday of that feed).
* **Past versions**: Mobility Database and Transitland list every published version with its service dates;
  `downloaders/utils/historic.py` downloads several versions and stitches them into one feed with consecutive calendars
  (Spain's NAP has its own variant in `downloaders/spain/nap.py`).
* **Broken file**: `Feed(..., check_files=True)` repairs the usual errors (quotes, missing columns, bad times) and logs
  what it changed (`pyGTFSHandler.utils.io`).

## 3. When no feed exists: infer one

Use the first that works, and say which one in the feed's `ASSUMED.md`:

1. **Published timetables** (PDF, Excel, HTML, apps): lines, stops, first / last departure, headways. Text extraction
   first (`pdfplumber`, `pdftotext -layout`), OCR only for scans; read every sheet of a spreadsheet.
2. **An open network dataset** (routes, stops, geometry without times), e.g. CPTOND for Chinese cities
   (`TransitLOSStudies/city_science_network/china/GTFS_PLAYBOOK.md`): times come from distance and a researched speed,
   headways from timetables or fleets.
3. **OpenStreetMap route relations** (stop order, geometry, `ref`, sometimes `interval` and `opening_hours`): read the
   downloaded extract with `downloaders/utils/osm_extract.OsmData`, not Overpass.
4. **Fleets** (registers, tenders, press notes): headway = round trip / vehicles in service
   (`downloaders/utils/fleet_frequency.py`, method in section 8).
5. **Aggregators** (Moovit, treep, Rome2Rio, Wikipedia line tables): names, stops, intervals when nothing official
   exists; say so.

Write such a feed as **frequency-based GTFS** (one representative trip per direction, relative stop times,
`frequencies.txt` with `exact_times=0`), mark everything `ASSUMED` (folder `assumed_<name>`, agency and `route_desc`
starting with `ASSUMED`, an `ASSUMED.md` with sources and the basis of each headway), run `stagger_feed`, and keep it in
the study repo. A service you cannot find a count for is not modelled: list it under "Known limits".

## 4. Add a downloader (when code can fetch or build the data)

A downloader is worth writing when the source has an API or a fixed download pattern that code can follow for any
area (a national access point, an authority's portal, a public bucket), or when a public dataset can be turned into GTFS
by rules alone.

1. **Where**: `pyGTFSHandler/downloaders/<country>/<source>.py` (country-level) or `pyGTFSHandler/downloaders/<source>.py`
   (international catalogue). Export the class from the folder's `__init__.py`.
2. **Interface**: subclass `BaseGTFSDownloader`; set `SOURCE_NAME` (short id) and `API_KEY_ENV_VAR` (if a key is needed);
   implement `search_feeds(**filters) -> List[GTFSFeedMetadata]` (one entry per downloadable file, with its URL and the
   service dates when the source gives them). `download_feeds()` is inherited (`downloaders/utils/download.py`: resumable,
   browser user agent, refuses HTML pages, unzips).
3. **Keys**: read them only through `get_api_key(SOURCE_NAME, api_key, API_KEY_ENV_VAR)` (argument, then environment
   variable, then the local gitignored keys file). Raise a clear error naming the key and the sign-up page when it is
   missing.
4. **Building a feed by code** (e.g. GTFS from an authority's line API): put the builder next to the downloader, write
   the output only to the folder the caller gives (never into the package), and make every number traceable to the
   source (no constants typed in by hand except documented rules such as the fleet shares).
5. **Tests**: `tests/downloaders/test_<source>.py` with the network mocked (recorded small responses); check the
   metadata fields and that a missing key raises.
6. **Docs**: a page under `docs/api/` (or let `mkdocs` pick up the module) and one line in the source table of section 1.

Hand-checked lists that complete a source (dead links replaced by hand, partial feeds to skip) are **inputs from the
study**: the downloader takes their path as an argument (see `downloaders/usa/us_feeds.py`, whose lists live in
`TransitLOSStudies/usa_study/feed_lists/`).

## 5. API keys (the one standard place)

Every key of every package of the project lives in **one file outside every repository**: `~/.config/ptlos/api_keys.json`,
a JSON object `{"<key name>": "<key>"}` (permissions 600). Both pyCensus (`pycensus.api_keys.resolve_api_key`) and pyGTFSHandler
(`pyGTFSHandler.downloaders.utils.config.get_api_key`) read it. Order: a key passed as an argument, then the source's
environment variable, then the file (`API_KEYS_FILE=<path>` points to another file). The key names are listed in
`api_keys.example.json` at the root of each package. Never put a key in a repository, a notebook, a log or a message to anyone
but the user.

**When a source needs a key you do not have, stop and tell the user**, in one short message:

1. which source needs it and what it would add;
2. where to sign up (the registration page) and whether it is free;
3. what to do with the key: *either* paste it to you in the chat, *or* add it themselves to `~/.config/ptlos/api_keys.json`
   as `"<key name>": "<key>"`.

When the user gives you a key, write it into `~/.config/ptlos/api_keys.json` under its key name (create the file with
permissions 600 if it is missing, keep the other entries), never echo it back or print it, and check it works with one small
request. Then continue. Meanwhile do what needs no key and leave the module raising its credentials error.

Keys in use today:

| Source | Key name | Environment variable | Sign up |
|---|---|---|---|
| Mobility Database API | `mobility_database` (refresh token) | `MOBILITY_DATABASE_REFRESH_TOKEN` | https://mobilitydatabase.org (account, then "refresh token") |
| Transitland API | `transitland` | `TRANSITLAND_API_KEY` | https://www.transit.land (account, API key) |
| Spain NAP | `nap_es` | `NAP_API_KEY` | https://nap.transportes.gob.es (register) |
| US NTD (Socrata) | `ntd` | `SOCRATA_APP_TOKEN` | optional; https://data.transportation.gov (app token, raises the rate limit) |

The Mobility Database public catalogue CSV and its `urls.latest` copies need no key.

## 6. Check before you stop

- [ ] every line of the area is in a feed or listed under "Known limits" with what was searched
- [ ] `coverage_check` run; flagged routes explained (section 7)
- [ ] no hand-made file added under `pyGTFSHandler/`
- [ ] each hand-made feed's source, date and licence written in its `ASSUMED.md`
- [ ] never `git commit` / `git push` for the user

## 7. Finding and filling missing routes and frequencies (guide for people and AI agents)
Goal: every real, regularly running public-transport line of a study area (metro, tram, light rail, BRT, suburban/regional/intercity rail, bus, and the frequent minibus/colectivo
systems that behave like lines) exists in the GTFS data **with a believable frequency, speed and span**. The score depends almost only on those three. This guide says how to find what is
missing, where to look, how to build it, and how to record it. Never invent data: an estimate is allowed only if it is derived from a stated source and marked as an estimate.

### 0. Tools

| Tool | What it does |
|---|---|
| `pyGTFSHandler.downloaders.utils.coverage_check` (`python -m pyGTFSHandler.downloaders.utils.coverage_check --feeds F1 F2 --bbox S W N E --out report.json`) | the automatic check: flagged routes, OSM routes missing from the feeds, OSM stations without a stop, OSM `interval` conflicts. OSM comes from a **downloaded extract** (`--osm region.osm.pbf`), parsed once per file and bbox and cached. |
| `TransitLOSStudies/city_science_network/code/check_missing_transit.py` (`python -m code.check_missing_transit <city> ...`) | the same per city-science city, plus **underserved clusters** from the last run's scores and the OSM context of each. Writes `<city>/checks/missing_transit.json`. |
| `downloaders/utils/osm_extract.py` | `download_extract()` (Geofabrik `.osm.pbf`, skipped if present) and `OsmData` (route relations, stations, bus stops, places of a bbox; needs `pip install osmium`, ~2.5 min for a 650 MB country extract, then cached). |
| `downloaders/utils/fleet_frequency.py` | fleet -> headway, fleet split, rescale, `frequencies.txt`, phase (see `section 8`). |
| `downloaders.MobilityDatabaseDownloader`, `TransitlandDownloader`, `downloaders/spain/nap.py`, `downloaders/usa/*` | find official feeds. |

Download the region's `.osm.pbf` once (`download_extract`), then run the check after every rebuild of a city and whenever a feed is added. Work the report from the top.

### 1. Reading the report

`missing_transit.json` / `report.json` keys:

* `flagged_routes` -- routes with an `issues` list: `no_service` (defined, no weekday trips: calendar mismatch or empty route), `few_departures` (1-2 trips a day: a cut or partial timetable),
  `no_peak_service`, `slow_metro|slow_tram|slow_rail|slow_bus` (peak headway above 10 / 15 / 30 / 60 min -- metro, tram and rail rarely run that rarely in a city: **confirm against the operator's timetable**;
  a real rural line at 120 min is fine, a metro at 25 min is almost surely wrong or a partial service).
* `osm_routes_missing_from_feeds` -- OSM route relations (bus, tram, subway, light rail, train, ...) whose `ref` (or name when it has no ref) matches no `route_short_name`, `route_id` or long name in any feed,
  grouped per line, most relations first. Each is a *candidate*: it may be a renamed line, a discontinued one, a school/tourist service, or a real gap.
* `osm_stations_missing_from_feeds` -- OSM `railway=station|halt|tram_stop` with no feed stop within 300 m. A new station, or a whole line missing.
* `osm_interval_conflicts` -- OSM `interval` tag at least 2x shorter than the feed's headway for the same ref. Volunteer data: a reason to find the timetable, not a truth.
* `underserved_clusters` (city-science wrapper) -- large connected groups of populated cells with score < 25. Each has `population`, `mean_score`, `bbox`, `feed_stops` (stops of all feeds inside) and `osm`
  (OSM bus stops, stations, route refs, places there). Interpretation:
  * many `osm_bus_stops` or `osm_route_refs`, few `feed_stops` -> **a network missing from the feeds** (find the operator's feed / build the lines);
  * many `feed_stops`, low score -> **frequency too low**: research the operator's real timetable or fleet;
  * few of both and rural -> probably genuinely unserved; confirm with the regulator's list of services (and record that you checked).

### 2. Where to look (in this order)

1. **Official feeds** not yet loaded: Mobility Database catalogue (`MobilityDatabaseCatalog().search_feeds(area)`; `urls.latest` public copy), Transitland (operators by bbox), national access points (Spain NAP, US NTD/state DOTs,
   Germany DELFI/GTFS.de, Switzerland, Canada city open-data portals, Taiwan TDX, Israel MOT, Chile DTPR, Mexico state portals), operator websites ("datos abiertos", "developers", "GTFS"), regional transit authorities
   and the Wayback Machine for portals that are down. A feed older than the service may still be right for lines that did not change; the scorer uses each feed on its own calendar.
2. **Timetables as documents** (PDF, Excel, CSV, HTML tables, apps): operator line pages, regulator service lists, tender documents (fleet, vehicles-per-line, headways, first/last trip, speeds),
   press notes on a new line. For PDFs use text extraction first (`pdfplumber`, `pdftotext -layout`), OCR only for scans; for spreadsheets read **every sheet** (fleet registers hide the data on sheet 2).
   Record the exact URL, the date of the document and what you took from it.
3. **Third-party aggregators** (treep.app, Moovit, Rome2Rio, Citymapper pages, Wikipedia line tables): acceptable for names, stops, first/last departure and intervals when no official source exists; say so, and
   prefer a value from the operator when both exist.
4. **OpenStreetMap**: route relations give stop order, geometry, `ref`, sometimes `interval`, `opening_hours`, `operator`. Use `type=route` relations (not `route_master`), both directions, `out geom` for the path.
   **Do not use Overpass for metro-area queries**: read the downloaded extract (`osm_extract.OsmData`; the city-science streets folder already holds the Geofabrik `.osm.pbf` of every city). For a one-off look at a single line, the OSM website or a very small Overpass query is fine.
5. **Rail census**: for every study area list the OSM rail relations and stations and compare with Wikipedia's list of lines/stations -- conventional, suburban and intercity rail are the lines most often forgotten.
   Cached examples: `TransitLOSStudies/cache/osm/shanghai_rail/`.

### 3. Building the missing line

The common steps (the builder is `TransitLOSStudies/city_science_network/assumed_feeds/build_assumed_feeds.py`):

1. existence and name (a published list, relation or table); 2. stops (published coordinates, else OSM stops within 35-40 m of the path, else interpolated every ~700 m where a gap exceeds 1 km, named "(interpolated)");
3. path (shortest path on the OSM road/rail network through the stops in order; drop an anchor that makes the path double back; reject a line whose implied speed is outside 10-40 km/h for buses -- rail and BRT have their own ranges);
4. frequency by the order of preference in `section 8` (timetable > published headway > fleet > documented cautious value; never a median of the feed);
5. checks: not a duplicate of an official line (stop overlap with the best official route), stops within a few tens of metres of the path, speed plausible.

Write the feed as a **frequency-based GTFS** (one representative trip per direction, relative stop times, `frequencies.txt` with `exact_times=0`), mark it as assumed -- folder `assumed_<name>`, `feed_info.txt` publisher and
`agency.txt` name starting with `ASSUMED`, `route_id` starting `assumed_`, `route_desc` starting `ASSUMED:`, an `ASSUMED.md` with sources and `HEADWAY_BASIS` -- add it to `assumed_feeds/registry.json` and run `stagger_feed`.

**Rail and rapid transit**: the number of stations and the end-to-end time are usually published (Wikipedia, operator); derive speed from them, headway from the timetable of the busiest sections or from the
number of daily trains (trains per day / operating hours). Conventional lines with only "N trains a day" are modelled as that many departures spread over the operating span. A service you cannot find a count for is
**not** modelled (document it under "Known limits"), it is not guessed.

**Rural / low-demand buses**: do not assume urban headways. Use floors (inter-town 40/60 min peak/off-peak, village 60/90 min unless the source says otherwise) and say they are floors.

### 4. When the line is missing in OpenStreetMap too

The feed is built from the other sources (operator stop list, road-network path between known stops). Then:

* put `OSM_MISSING: yes` in the line's block of `ASSUMED.md` and list in an `OSM_TODO.md` next to it what an OSM mapper would need: `type=route, route=bus, ref=, name=, operator=, network=, from=, to=, interval=, opening_hours=`, the stops, and the source URL;
* do not upload to OSM automatically. A human reviews `OSM_TODO.md`; once a line is in OSM, the next run of `missing_osm_routes` stops reporting it.
* if the line is in OSM but with the wrong `ref` (the check reports it as missing), fix the match in the builder (an alias table) rather than adding a duplicate.

### 5. After filling

1. Re-run the check; the finding must disappear or be explained (renamed, discontinued, school service, out of area).
2. Rebuild the city (`run_all.py --full <city>` when stops/lines changed; refresh otherwise) and compare the score of the affected area.
3. Append to the feed's `ASSUMED.md` (committed with the feed): what was added, sources (URL + date), estimates and their basis, **known limits** (what you looked for and did not find).
4. Never run `git commit` / `git push` for the user; leave changes in the working tree.

### 6. Rules that always apply

* Real, line-specific frequencies. Never feed medians. Fleet-derived headways: 8% maintenance + 9% spare + 10% layover.
* One feed per service: never put backups, old versions or duplicates in `<city>/gtfs/` (the pipeline loads every folder there).
* Quote sources with URL and date; flag every estimate; document what could not be found.
* The check is a lead generator. A flagged route can be right (night lines, school services, genuinely rare rural lines): decide with evidence, and leave a one-line note in the feed's `ASSUMED.md` when you decide to keep a flagged route.

## 8. From a fleet to frequencies
Many operators publish **how many vehicles they run** (a fleet register, a tender, a press note, a regulator's Excel sheet) but no timetable. This page is the method for turning that
number into `frequencies.txt` headways. The code is `pyGTFSHandler.downloaders.utils.fleet_frequency`; the worked examples are the Gran Concepción and Guadalajara feeds
(each feed's basis is in its `ASSUMED.md`).

### 1. Order of preference for a line's headway

Use the first that exists, and write which one you used in the feed's `ASSUMED.md` (`HEADWAY_BASIS:`):

1. the line's **own published timetable** (first / last departure, interval) -> use it, no estimate;
2. a **published line-specific headway** (regulator guide, operator website, press) -> use it;
3. the **fleet of the line** (this page);
4. a **documented cautious value** (say what it is and why, e.g. "20 min: the figure the state gave for the one comparable line");
5. never: the median or mean of the other lines of the same feed. That hides exactly the differences the score is about.

### 2. The formula

```
vehicles in service  = fleet x (1 - 8% maintenance - 9% spare)            = 83% of the fleet
round trip (min)     = 2 x one-way km / commercial speed (km/h) x 60 x (1 + 10% layover)
headway (min)        = round trip / vehicles in service on that line
```

* **8% maintenance, 9% spare, 10% layover** are the study's defaults (`MAINTENANCE_SHARE`, `SPARE_SHARE`, `LAYOVER_SHARE`); change them only with a source and say so.
* **Commercial speed** includes stops and traffic. Typical values used in the study: urban bus in dense areas 14-20 km/h by stop spacing, 28-34 km/h for 1-4 km spacing,
  40 km/h inter-urban, BRT floor 24 km/h, tram 24-27 km/h, and the measured speed of the operator's own feed whenever one exists. Reject a line whose implied speed is
  outside 10-40 km/h (the geometry or the fleet is wrong).
* **Loops** run once around: pass `loop=True` (one leg instead of two).
* The result is clamped to 2-60 min (`headway_from_fleet(min_headway=, max_headway=)`); if the clamp is hit, state it.
* Check: Guadalajara Line 5, 41 buses, 34.7 km, 25 km/h -> `headway_from_fleet(41, 34.7, 25)` = 5.4 min.

```python
from pyGTFSHandler.downloaders.utils.fleet_frequency import headway_from_fleet
headway_from_fleet(fleet=41, one_way_km=34.7, speed_kmh=25)   # 5.38
```

### 3. One fleet, several lines (`split_fleet`)

Operators usually quote the fleet of the *company*, which runs several lines. Divide the vehicles in service among the lines:

* `method="equal_headway"` (default): all lines get the same headway; longer lines get more vehicles (`h = sum(round trips) / vehicles`). Right for a branch network run as a
  system, e.g. the four Coronel-Lota operators (Sotral 44, Nueva Takora 39, Nueva Ruta 160 43, Playa Colcura 28 = 154 buses over the shared OSM branches: one headway for all branches).
* `method="weighted"`: vehicles in proportion to a weight you can justify (passengers, trips, buses the operator says it assigns to each line). Right when a line is known to be the
  trunk (Santa Juana: 80% of the 23 buses on the Concepción trunk, the rest on local branches).
* Several **service patterns on one corridor** (all-stops, express, short-turn): `share_between_services(vehicles, {name: (round_trip_min, relative_frequency)})`.

Always check that the split adds up: `sum(vehicles) == vehicles_in_service(fleet)`.

### 4. A fleet that is known only in total, with generic headways already in the feed (`rescale_headways`)

When a feed has plausible but generic headways (e.g. built from a stop-spacing model) and a source gives the fleet of the whole city/operator, scale **all free headways by one factor k**
so the vehicles they imply (round trip / headway, at the 08:00 weekday peak) equal the in-service fleet. Lines whose headway is a published fact are `fixed`: keep them and subtract their
vehicles first. `k` is limited to [0.5, 4] because a fleet figure often includes vehicles outside the scored area. The Shanghai-area calibration
(`china/CPTOND-2025/code/calibrate_other_city_buses.py`) does this city by city with line classes (BRT-like 0.45, fast 0.75, trunk 0.85) and tier weights; read `bus_calibration.json` in the feed folder
for the resulting k, the target and the buses implied.

### 5. Writing `frequencies.txt`

```python
from pyGTFSHandler.downloaders.utils.fleet_frequency import frequency_rows, write_frequencies, stagger_feed
rows = frequency_rows("T_line5_out", [("03:30", "06:00", 8), ("06:00", "21:00", 5.4), ("21:00", "22:30", 10)])
write_frequencies(feed_dir, rows)      # one representative trip per direction + one row per time band
stagger_feed(feed_dir)                 # last step, always
```

* Keep one **representative trip** per route and direction in `trips.txt` with relative `stop_times.txt` (first stop 00:00:00, later stops from distance / speed).
* Use time bands: peak (7-9, 17-19), midday, evening, early morning, with the headways the source implies; a single headway all day is acceptable only if that is what the source says.
* `exact_times = 0` (headway-based).
* **Phase**: `stagger_feed` moves each window start by a fixed pseudo-random offset smaller than the headway (md5 of the trip id) so independent lines sharing a station do not depart on the same seconds
  (the scoring counts simultaneous departures once). Metro (route_type 1) is left alone. It is idempotent (`frequencies.nostagger.txt` keeps the original).

### 6. What to record

In the feed's `ASSUMED.md`: the fleet and its source (document, date), the shares used, the commercial speed and why, the length and where it comes from, the split method, the resulting headway per line, and
every clamp or cap that was applied. Fleet figures are estimates of an estimate; the file must let the next person redo them.
