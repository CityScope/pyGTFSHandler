# pyGTFSHandler

**A Python package to download, load, and pre-process GTFS public transport timetable files, built on Polars for speed.**

`pyGTFSHandler` loads one or more [GTFS](gtfs-format.md) (General Transit Feed Specification) static feeds into a single, denormalized `Feed` object and gives you:

- Fast, lazy loading of large GTFS feeds using [Polars](https://pola.rs) `LazyFrame`s.
- Geographic filtering of stops/trips to an Area of Interest (AOI).
- Date-range, time-range, `route_type`, and id-based filtering.
- Expansion of `frequencies.txt`-defined trips into concrete departures.
- Geometry-derived `direction_id` assignment and stop/edge-level speed and headway analysis (see [Direction & Headway methodology](methodology/direction-headway.md)).
- Interactive Leaflet route maps and conflict maps.
- Downloaders for the [Mobility Database](https://mobilitydatabase.org/), [TransitLand](https://www.transit.land/), and Spain's NAP open-data portal.

## Installation

```bash
pip install git+https://github.com/GeomaticsCaminosUPM/pyGTFSHandler.git
```

Optional extras (matching `pyproject.toml`'s `[project.optional-dependencies]`):

```bash
pip install "pyGTFSHandler[plot]"      # matplotlib/folium/plotly/streamlit map & plotting helpers
pip install "pyGTFSHandler[geocoding]" # geopy-based geocoding/reverse-geocoding helpers
pip install "pyGTFSHandler[docs]"      # build this documentation site locally
```

This project also uses [uv](https://docs.astral.sh/uv/) for development; from a clone of the repo:

```bash
uv sync --extra docs
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

See the [Examples](examples/index.md) section for complete, runnable notebooks, the [API Reference](api/index.md) for the full class/method documentation, and [Methodology](methodology/index.md) for the math behind the non-obvious parts of the library (calendar resolution, frequency expansion, and the direction/headway algorithms).
