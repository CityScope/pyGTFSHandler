# pyGTFSHandler

**A Python package to download, load, and pre-process GTFS public transport timetable files, built on [Polars](https://pola.rs) for speed.**

`pyGTFSHandler` loads one or more [GTFS](https://gtfs.org/) (General Transit Feed Specification) static feeds into a single, denormalized `Feed` object and gives you:

- Fast, lazy loading of large GTFS feeds using Polars `LazyFrame`s.
- Geographic filtering of stops/trips to an Area of Interest (AOI).
- Date-range, time-range, `route_type`, and id-based filtering.
- Expansion of `frequencies.txt`-defined trips into concrete departures.
- Geometry-derived `direction_id` assignment and stop/edge-level speed and headway analysis.
- Interactive Leaflet route maps and conflict maps.
- Downloaders for the [Mobility Database](https://mobilitydatabase.org/), [TransitLand](https://www.transit.land/), and Spain's NAP open-data portal (with room to add more countries).

Full docs (API reference, methodology notes, runnable example notebooks): **https://CityScope.github.io/pyGTFSHandler/**

## Installation

```bash
pip install git+https://github.com/CityScope/pyGTFSHandler.git
```

Optional extras:

```bash
pip install "pyGTFSHandler[plot]"      # matplotlib/folium/plotly/streamlit map & plotting helpers
pip install "pyGTFSHandler[geocoding]" # geopy-based geocoding/reverse-geocoding helpers
pip install "pyGTFSHandler[docs]"      # build the documentation site locally
```

This project uses [uv](https://docs.astral.sh/uv/) for development; from a clone of the repo:

```bash
uv sync --extra docs --extra plot --extra dev
```

## Quickstart

```python
from pyGTFSHandler import Feed

# Load one or more GTFS directories/zips into a single Feed
feed = Feed(
    gtfs_dirs="path/to/gtfs_folder",   # or a list of paths, one per feed
    stop_group_distance=20,             # merge stops within 20m into one parent_station
)

# Interactive route/timetable map for a given service date
from pyGTFSHandler.maps import route_map
m = route_map(feed, date="2024-05-06")

# Headway (mean interval between trips) at every stop, grouped by geometric direction
headway = feed.get_headway_at_stops(
    date="2024-05-06",
    by="shape_direction",
)
```

`feed.lf` is a single Polars `LazyFrame` joining `stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt`, `calendar.txt`/`calendar_dates.txt` and `shapes.txt` into one denormalized schedule table, ready for your own analysis.

## Downloading GTFS feeds

```python
from pyGTFSHandler.downloaders import MobilityDatabaseDownloader

downloader = MobilityDatabaseDownloader()  # or api_key=..., falls back to env var
feeds = downloader.search_feeds(country_code="ES")
downloader.download_feeds(feeds, download_folder="path/to/download")
```

`TransitLandDownloader` and Spain's `downloaders.spain.NAPDownloader` follow the same interface. See the [downloaders API reference](https://CityScope.github.io/pyGTFSHandler/api/downloaders/base/) for details.

## Documentation

- **[Examples](https://CityScope.github.io/pyGTFSHandler/examples/)** — runnable notebooks, including a full Cambridge, MA case study and a walkthrough of the direction/headway methodology.
- **[API Reference](https://CityScope.github.io/pyGTFSHandler/api/)** — full class/method documentation.
- **[Methodology](https://CityScope.github.io/pyGTFSHandler/methodology/)** — the math behind calendar resolution, frequency expansion, and the direction/headway algorithms.

To build the docs locally:

```bash
uv sync --extra docs
uv run mkdocs serve   # live-reloading dev server at http://127.0.0.1:8000
```

## Related project

[`UrbanAccessAnalyzer`](https://github.com/CityScope/UrbanAccessAnalyzer) builds on `pyGTFSHandler` to compute full accessibility indicators (isochrones, POI access, public-transport quality scores) from GTFS, OSM, and population data.

## License

[GNU General Public License v3.0](LICENSE)
