# Examples

Runnable Jupyter notebooks from the [`examples/`](https://github.com/GeomaticsCaminosUPM/pyGTFSHandler/tree/main/examples)
directory, rendered in place below.

- **[Quickstart](quickstart.ipynb)** — loads small local GTFS feeds bundled
  in the repo (`examples/test_files/sevilla`), runs offline. Covers building
  a `Feed`, the interactive `route_map`, and stop/edge-level speed and
  headway analysis.
- **[Cambridge, Massachusetts, USA — full example](cambridge_massachusetts_usa_example.ipynb)** —
  an end-to-end workflow against real, downloaded GTFS data: searching and
  downloading feeds from the Mobility Database, building a `Feed`, picking a
  representative service day, route maps, speed/headway analysis at stops
  and edges, exporting to GIS-ready files, and a full GTFS DataFrame column
  reference.
- **[Direction clustering and headway: a Y-shaped worked example](direction_and_headway_methodology.ipynb)** —
  why `route_id`-based headway is unreliable and geometric,
  direction-based headway is not; how direction clustering works; the
  headway formula; and a real-feed (Sevilla) headway map. Companion to the
  [Direction & headway methodology](../methodology/direction-headway.md) docs page.
- **[Direction & headway methodology](direction_conflicts.ipynb)** — how
  `direction_id` is derived from geometry alone (reconciling forward/backward
  bearings at a stop, clustering bearings via the "widest gap" split) and
  how per-stop departures are summarized into a single headway number, with
  small synthetic examples.

!!! tip
    These notebooks are also runnable directly in
    [Google Colab](https://colab.research.google.com/) — each one has a
    "Open in Colab" badge as its first cell.
