# API Reference

Auto-generated from docstrings in the source code via [mkdocstrings](https://mkdocstrings.github.io/).

- **[Feed](feed.md)** — the main user-facing entry point: loads one or more GTFS feeds into a single denormalized Polars-backed object and exposes filtering/analysis methods.
- **Models** — one class/module per GTFS table, handling parsing, normalization, and (for [Calendar](models/calendar.md) and [Frequencies](models/frequencies.md)) expansion logic.
- **Analysis** — [edge](analysis/edges.md) construction, geometric [direction/stop](analysis/stops.md) analysis, and feed-wide [filtering](analysis/filtering.md).
- **Downloaders** — fetch GTFS feeds from the [Mobility Database](downloaders/mobility_database.md), [TransitLand](downloaders/transitland.md), or [Spain's NAP portal](downloaders/spain.md).
- **Maps** — Leaflet-based [route maps](maps/route_map.md) and [conflict maps](maps/conflict_map.md).
- **[Utils](utils.md)** — geometry, time, date, and I/O helpers used throughout the package.
