# Edge & Filtering Logic

This page covers two mixins on `Feed`: [`FeedFilteringMixin`](../api/analysis/filtering.md)
(date/time/route-type/AOI filtering) and [`FeedEdgeAnalysisMixin`](../api/analysis/edges.md)
(stop-to-stop "edge" headway and speed).

## Date filtering: `day_offset` and nominal vs. real dates

GTFS allows `arrival_time`/`departure_time` values past `24:00:00` for trips
that run past midnight. pyGTFSHandler keeps such trips attached to their
original *nominal* service date and stores how many days a stop_time spills
into the next day as `day_offset` (0 for same-day, 1 for a trip that departs
after midnight relative to its `trip_id`'s nominal service date, etc.).

The *real* calendar date of a stop_time is therefore:

$$
\text{real_date} = \text{service_date} + \text{day_offset}
$$

`_max_day_offset` scans the data for the largest `day_offset` present so that
`_filter_by_date`/`_filter_by_date_range` know how many "nominal date = real
date − day_offset" candidate service dates to check — a stop_time can only
match a requested real date if its nominal service date is active *and* its
`day_offset` bridges the gap. This lets `filter_by_date` and
`filter_by_date_range` return exactly the stop_times that actually occur (in
wall-clock terms) on the requested date(s), including the tail end of
overnight trips, without duplicating rows for same-day trips.

## Time-of-day filtering and `frequencies.txt`

`_filter_by_time_range` filters to a `[start_time, end_time]` window of
seconds since midnight (via `time_parsing.time_to_seconds`). Trips defined by
`frequencies.txt` are handled differently from explicit `stop_times.txt`
trips: a frequency-based "trip" only has a start/end window and a headway,
not concrete departure times, so it is kept if its window overlaps
`[start_time, end_time]` at all (the actual departures are only materialized
later, during expansion — see [Frequency expansion](frequencies.md)) — rather
than requiring a specific departure timestamp to fall inside the window.

## Edge construction

Both `get_headway_at_edges` and `get_speed_at_edges` build "edges" (directed
stop-to-stop segments) the same way:

1. Filter the feed to the requested `date`/`start_time`/`end_time`/`route_types`.
2. Sort by `(trip_id, stop_sequence)` and pair each stop with the
   previous stop on the same trip (`stop_id_B = stop_id.shift(1).over("trip_id")`),
   producing one row per consecutive stop pair per trip.
3. Derive a direction-independent `edge_id` by ordering the two endpoint ids
   lexicographically (`{at}_A` vs `{at}_B`, where `at` is typically
   `"parent_station"`) — i.e. the same physical segment travelled in either
   direction gets the same `edge_id`, with `direction_id` (0 or 1) recording
   which way each row went relative to that canonical order.

## Headway per edge

`get_headway_at_edges` groups rows sharing an edge (and grouping key `by`,
e.g. `route_id` or `shape_direction`) and applies the same RMS-style headway
statistic used for per-stop headway (see
[Direction & headway methodology](direction-headway.md#4-headway-mean-interval-at-stops)):

$$
\text{headway} = \frac{\left(\sum_{i} \Delta t_i^{2}\right) + \text{initial_headway}^{2}}{t_{\text{end}} - t_{\text{start}}}
$$

where $\Delta t_i$ are consecutive differences between sorted departure
times on that edge/group, and `initial_headway` accounts for the gap before
the first and after the last departure within the requested time window
(so an edge with a single very early or very late trip is penalized rather
than scored as a perfect headway). The result is converted from seconds to
minutes before being returned.

When aggregating multiple routes/directions sharing a physical edge
(`how="add"`), pyGTFSHandler combines headways as combined *frequencies*
(reciprocal rates), matching how independent, roughly-Poisson arrival
processes combine:

$$
\text{headway}_{\text{combined}} = \left(\sum_{k} \frac{1}{\text{headway}_k}\right)^{-1}
$$

`how="best"`/`"max"`/`"min"` instead pick the extreme value across the group
(e.g. the shortest headway among candidate routes at that edge), keeping the
associated `route_ids`/`shape_direction`/endpoint columns from that row via
`sort_by("headway", ...).first()`/`.last()`.

Edges with fewer than `min_trips` trips are dropped from the result, since a
headway computed from 0–1 observed departures is not meaningful.

## Speed per edge

`get_speed_at_edges` derives, per consecutive stop pair on a trip:

- `distance_weight`: the difference in `shape_dist_traveled` between the two
  stops (meters along the shape geometry, not straight-line).
- `time_weight`: the difference in `departure_time` between the two stops
  (seconds).

and computes speed in km/h:

$$
\text{speed} = \frac{\text{distance_weight} / 1000}{\text{time_weight} / 3600}
$$

Aggregation across trips sharing an edge (`how="mean"`) is a weighted mean —
distance and time are separately summed weighted by `n_trips` per row, and
speed is recomputed from the aggregated totals, rather than averaging
per-trip speeds directly:

$$
\text{speed}_{\text{mean}} = \frac{\sum_k n_k \cdot d_k \, / \, 1000}{\sum_k n_k \cdot t_k \, / \, 3600}
$$

This avoids over-weighting infrequent trips and matches how a rider
experiences "typical" travel time better than an unweighted average of
speeds would. `how="max"`/`"min"` instead pick the fastest/slowest single
observed trip on that edge (with `+inf`/`-inf` sentinels used during sorting
so `null` speeds — e.g. zero time_weight — sort last, then converted back to
`null` afterward).
