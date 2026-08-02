# -*- coding: utf-8 -*-
"""Helper/utility modules, grouped here (as opposed to `models/`, which holds
the GTFS-file-specific domain classes) because none of them are tied to one
particular GTFS file:

- `io`: file-system-level IO -- locating files in a folder/zip, unzipping,
  lazily scanning CSVs into polars (`read_csv_lazy`/`read_csv_list`), and the
  slower row-tolerant CSV parser used when `check_files=True`.
- `gtfs_checker`: per-value parsing/normalization rules (dates, times, GTFS
  standard vs. extended route types, id-string normalization) that `io.py`
  and the row-tolerant parser rely on -- no file/path awareness of its own.
- `geo_polars`: polars-native great-circle distance and grid-bucketing
  helpers (shared by `models/stops.py` clustering and `models/shapes.py`
  real shape geometry), plus general-purpose polars expression helpers that
  don't belong anywhere more specific (`filter_by_id_column`, `mean_angle`,
  `max_separation_angle`).
- `time_parsing`: GTFS time-string parsing (`normalize_time_expr`, used by
  both `models/stop_times.py` and `models/frequencies.py`),
  `datetime`/`time` -> seconds-since-midnight conversion, and along-trip
  time/distance interpolation (`time_displacement`).
- `date_parsing`: the epoch-day integer convention used throughout
  `models/calendar.py` for GTFS dates, and public-holiday lookups.
- `geocoding`: place-name/AOI -> country/subdivision/municipality
  resolution, including the one geocoding call on the core `Feed` loading
  path (`get_country_region`, used for holiday lookups).
- `hashing`: content-hashing a GTFS file/folder/zip, for change detection.
- `colors`: deterministic hex-color helpers (`route_id_to_color`,
  `contrasting_text_color`) shared by `models/routes.py` (filling in missing
  `route_color`/`route_text_color`) and `maps/style.py` (badge/icon styling).
- `processing_helpers`/`plot_helpers`: downstream analysis-output
  post-processing, static-matplotlib plotting (`service_intensity`), and the
  simpler geopandas-`.explore()`-based folium helper (`general_map`).
- The interactive, clickable `route_map` (a self-contained per-stop/route/trip
  Leaflet map for a given service date) lives in the top-level
  `pyGTFSHandler.maps` subpackage instead of here -- it needs its own JS/CSS
  assets (`pyGTFSHandler/maps/static/`) and Feed-specific data-prep code that
  didn't fit the "no file/path awareness" theme of this package.
- `stack_gtfs`: a standalone, manually-invoked tool for merging successive
  GTFS publications from the same agency into one feed before handing it to
  `Feed` (`Feed` itself treats every path it's given as an independent feed).
"""
