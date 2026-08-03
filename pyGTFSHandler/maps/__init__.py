# -*- coding: utf-8 -*-
"""Interactive folium/Leaflet map(s) for a `Feed`.

Split out of `utils/plot_helpers.py` (which keeps the matplotlib-only
`service_intensity` chart and the simpler geopandas-`.explore()`-based
`general_map` helper) because the main deliverable
here -- `route_map` -- is not expressible with folium/geopandas alone: it
needs click-driven, client-side-recomputed popups (stop -> timetable,
timetable row -> full trip itinerary, mode-filter checkboxes) that folium's
built-in layer/marker API can't produce. It works by serializing the
relevant slice of GTFS data to JSON and injecting it, plus hand-written
vanilla JS/CSS (kept in `maps/static/` rather than inline Python strings so
it stays readable/editable), into the map via a folium `MacroElement`.
"""

from .route_map import route_map
from .conflict_map import conflict_map

__all__ = ["route_map", "conflict_map"]
