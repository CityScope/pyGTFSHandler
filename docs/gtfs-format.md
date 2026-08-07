# GTFS Format Reference

GTFS (General Transit Feed Specification) static describes a public transport
network as a set of comma-separated `.txt` files inside a single folder or
`.zip` archive. This page summarizes each file's mandatory/optional status
and columns from the [GTFS spec](https://gtfs.org/schedule/reference/)
(general background, written from memory rather than fetched), and notes
which of those pyGTFSHandler actually reads or relies on (grepped from
`pyGTFSHandler/utils/gtfs_checker.py`, `pyGTFSHandler/feed.py`, and
`pyGTFSHandler/models/*.py`).

!!! note "Reading this page"
    Each file section has a **Used by pyGTFSHandler** line. Columns marked
    **(used)** are read and relied on by the library today; everything else
    is general GTFS background the library currently ignores or passes
    through unmodified.

## File overview

| File | Presence | Used by pyGTFSHandler |
|---|---|---|
| `agency.txt` | Mandatory | Structurally required; only `agency_id` is read |
| `stops.txt` | Mandatory | Yes — core |
| `routes.txt` | Mandatory | Yes — core |
| `trips.txt` | Mandatory | Yes — core |
| `stop_times.txt` | Mandatory | Yes — core |
| `calendar.txt` | Conditionally mandatory* | Yes — core |
| `calendar_dates.txt` | Conditionally mandatory* | Yes — core |
| `fare_attributes.txt` | Optional | Not read |
| `fare_rules.txt` | Optional | Not read |
| `shapes.txt` | Optional | Yes — real route geometry |
| `frequencies.txt` | Optional | Yes — expanded to discrete trips |
| `transfers.txt` | Optional | Not read |
| `pathways.txt` | Optional | Not read |
| `levels.txt` | Optional | Not read |
| `feed_info.txt` | Optional | Not read |
| `translations.txt` | Optional | Not read |
| `attributions.txt` | Optional | Not read |

\* At least one of `calendar.txt` / `calendar_dates.txt` must be present, per the GTFS spec and `pyGTFSHandler.utils.gtfs_checker`'s `MANDATORY_FILES`/`FILE_PAIRS` structural check.

---

## agency.txt

One row per transit agency operating services in the feed.

**Mandatory columns**

| Column | Description |
|---|---|
| `agency_name` | Full published name of the agency |
| `agency_url` | URL of the agency |
| `agency_timezone` | Timezone name (tz database), e.g. `Europe/Madrid` |

**Optional columns**

| Column | Description |
|---|---|
| `agency_id` | Unique agency identifier; required if the feed has more than one agency |
| `agency_lang` | Primary language used by the agency |
| `agency_phone` | Voice telephone number |
| `agency_fare_url` | URL for purchasing tickets online |
| `agency_email` | Customer service contact email |

**Used by pyGTFSHandler**: only structurally checked for presence; `agency_id` **(used)** is read as the sole schema column (`gtfs_checker.get_df_schema_dict`), and is the file's mandatory column for validation. `routes.txt`'s optional `agency_id` foreign key is not cross-checked against it.

---

## stops.txt

One row per stop, station, station entrance, or other physical/virtual
location referenced by `stop_times.txt`.

**Mandatory columns**

| Column | Description |
|---|---|
| `stop_id` | Unique identifier for a stop, station, or station entrance |

**Conditionally required**

| Column | Description |
|---|---|
| `stop_lat` | Latitude of the stop/station (WGS84) |
| `stop_lon` | Longitude of the stop/station (WGS84) |
| `stop_name` | Public-facing name of the stop |

**Optional columns**

| Column | Description |
|---|---|
| `tts_stop_name` | Text-to-speech readable name |
| `stop_desc` | Description of the stop |
| `zone_id` | Fare zone |
| `stop_url` | URL of a page about the stop |
| `location_type` | `0`=stop/platform, `1`=station, `2`=entrance/exit, `3`=generic node, `4`=boarding area |
| `parent_station` | Identifies the station a platform/entrance belongs to |
| `stop_timezone` | Timezone of the stop, if different from the agency's |
| `wheelchair_boarding` | Wheelchair accessibility |
| `level_id` | Level of the stop, referencing `levels.txt` |
| `platform_code` | Platform identifier |

**Used by pyGTFSHandler**: `stop_id`, `stop_lat`, `stop_lon` **(used, mandatory in `gtfs_checker`)**. `parent_station` **(used)** — `Feed.build_lf` deduplicates consecutive stop-time rows sharing a `parent_station`, and `Stops.group_stops`/`_cluster_by_distance` can synthesize/override `parent_station` by geographic clustering (`stop_group_distance`). `stop_name` is read where present for map/label purposes (`maps/*`) but is not mandatory to pyGTFSHandler itself.

---

## routes.txt

One row per transit route (a group of trips displayed as a single line to riders).

**Mandatory columns**

| Column | Description |
|---|---|
| `route_id` | Unique identifier for a route |
| `route_type` | Mode of transport, e.g. `0`=tram, `1`=subway, `2`=rail, `3`=bus, `4`=ferry, `5`=cable tram, `6`=aerial lift, `7`=funicular, `11`=trolleybus, `12`=monorail |

**Conditionally required**

| Column | Description |
|---|---|
| `agency_id` | Agency operating the route; required if `agency.txt` has more than one agency |
| `route_short_name` | Short name (e.g. `"32"`, `"100X"`); required if `route_long_name` is empty |
| `route_long_name` | Full descriptive name; required if `route_short_name` is empty |

**Optional columns**

| Column | Description |
|---|---|
| `route_desc` | Description of the route |
| `route_url` | URL of a page about the route |
| `route_color` | Route color as a hex string |
| `route_text_color` | Legible text color for `route_color` |
| `route_sort_order` | Ordering hint for display |
| `continuous_pickup` / `continuous_drop_off` | Continuous stopping behavior |
| `network_id` | Network the route belongs to |

**Used by pyGTFSHandler**: `route_id` **(used, mandatory)**, `agency_id`, `route_short_name`, `route_long_name` are read into the schema; `route_type` **(used)** is normalized by `gtfs_checker.normalize_route_type`/`extended_to_standard_route_type` (accepting both standard GTFS codes and the wider [Transmodel "extended" `route_type` values](https://developers.google.com/transit/gtfs/reference/extended-route-types) some feeds use, plus human-readable names like `"bus"`) and is the field `Feed(route_types=...)` filters on. `Routes._fill_route_colors` derives missing `route_color` values. `route_id` is joined onto the integrated `Feed.lf` for `route_type`.

---

## trips.txt

One row per scheduled trip along a route.

**Mandatory columns**

| Column | Description |
|---|---|
| `route_id` | Route this trip belongs to |
| `service_id` | Calendar/service pattern this trip runs on |
| `trip_id` | Unique identifier for the trip |

**Optional columns**

| Column | Description |
|---|---|
| `trip_headsign` | Text shown to riders (destination) |
| `trip_short_name` | Short, rider-facing identifier (e.g. train number) |
| `direction_id` | `0`/`1` — distinguishes two directions of travel for the same route |
| `block_id` | Groups trips a single vehicle serves sequentially |
| `shape_id` | Geometry of the trip's path, referencing `shapes.txt` |
| `wheelchair_accessible` | Wheelchair accessibility of the vehicle |
| `bikes_allowed` | Bicycle accommodation |

**Used by pyGTFSHandler**: `route_id`, `service_id`, `trip_id` **(used, mandatory)**. `direction_id` **(used)** is carried through to `Feed.lf` and used as a fallback/comparison in the direction-analysis notebooks and `get_headway_at_stops(by="route_id")`, though pyGTFSHandler's own `Shapes.assign_direction_ids` derives a geometry-based `direction_id` independently since raw GTFS `direction_id` is frequently null or inconsistent across feeds (see [Direction & Headway](methodology/direction-headway.md)). `shape_id` **(used)** links a trip to its real polyline in `shapes.txt`; `trip_headsign` is read for map/timetable display.

---

## stop_times.txt

One row per stop visited by a trip, in order.

**Mandatory columns**

| Column | Description |
|---|---|
| `trip_id` | Trip this stop time belongs to |
| `stop_id` (or `location_group_id`/`location_id`) | Stop visited |
| `stop_sequence` | Order of the stop within the trip |

**Conditionally required**

| Column | Description |
|---|---|
| `arrival_time` | `HH:MM:SS` time of arrival, may exceed `24:00:00` for post-midnight trips; required at timepoints |
| `departure_time` | `HH:MM:SS` time of departure; required at timepoints |

**Optional columns**

| Column | Description |
|---|---|
| `stop_headsign` | Headsign override for this specific stop |
| `pickup_type` / `drop_off_type` | Pickup/drop-off availability |
| `continuous_pickup` / `continuous_drop_off` | Continuous stopping behavior |
| `shape_dist_traveled` | Distance traveled along the shape at this stop |
| `timepoint` | Whether the given time is exact (`1`) or approximate (`0`) |

**Used by pyGTFSHandler**: `trip_id`, `stop_id`, `arrival_time`, `departure_time` **(used, mandatory in `gtfs_checker`)**, `stop_sequence` **(used)** — `StopTimes._correct_sequence` revises inconsistent sequences. Times are normalized to seconds-since-midnight with a `next_day` flag for values `>= 24:00:00`. `shape_dist_traveled`, if present, is used directly; otherwise pyGTFSHandler generates its own `shape_dist_traveled` from geometry (`Shapes._generate_shape_dist_traveled_column`) and uses it to interpolate missing times (`Feed._fix_null_times`).

---

## calendar.txt

Weekly service pattern rows: which weekdays a `service_id` runs, within a date range.

**Mandatory columns**

| Column | Description |
|---|---|
| `service_id` | Identifier for a service pattern |
| `monday` … `sunday` | `1` if the service runs on that weekday, else `0` |
| `start_date` | First date the pattern applies (`YYYYMMDD`) |
| `end_date` | Last date the pattern applies (`YYYYMMDD`) |

**Used by pyGTFSHandler**: every column **(used, all mandatory)** — see [Calendar Handling](methodology/calendar.md) for how `Calendar.get_services_in_date`/`get_services_in_date_range` resolve which `service_id`s are active on a date by combining these weekday/date-range rows with `calendar_dates.txt` exceptions, without duplicating rows.

---

## calendar_dates.txt

Single-date exceptions (add or remove a service on a specific date), either
on top of `calendar.txt` or as the sole calendar source.

**Mandatory columns**

| Column | Description |
|---|---|
| `service_id` | Identifier for a service pattern |
| `date` | The exception date (`YYYYMMDD`) |
| `exception_type` | `1` = service added on this date, `2` = service removed on this date |

**Used by pyGTFSHandler**: all three columns **(used, all mandatory)**. `Calendar.get_services_in_date` unions `exception_type=1` additions and subtracts `exception_type=2` removals from the `calendar.txt`-derived weekday matches.

---

## fare_attributes.txt / fare_rules.txt

Legacy fare model: `fare_attributes.txt` defines a fare's price/currency/transfer rules; `fare_rules.txt` maps fares to routes/zones/origin-destination pairs.

**fare_attributes.txt mandatory columns**: `fare_id`, `price`, `currency_type`, `payment_method`, `transfers`.
**fare_rules.txt columns** (all conditionally required/optional): `fare_id`, `route_id`, `origin_id`, `destination_id`, `contains_id`.

**Used by pyGTFSHandler**: not read at all — pyGTFSHandler is focused on schedule/timetable analysis, not fares.

---

## shapes.txt

Ordered points describing a trip's physical path.

**Mandatory columns**

| Column | Description |
|---|---|
| `shape_id` | Identifier for a shape |
| `shape_pt_lat` | Latitude of a shape point |
| `shape_pt_lon` | Longitude of a shape point |
| `shape_pt_sequence` | Order of the point within the shape |

**Optional columns**

| Column | Description |
|---|---|
| `shape_dist_traveled` | Cumulative distance traveled at this point |

**Used by pyGTFSHandler**: all columns **(used)** when present. pyGTFSHandler builds its own *synthetic* `shape_id` per group of trips sharing an identical stop sequence + travel time (`StopTimes.generate_shape_ids`), independent of the feed's own `shape_id`; if `shapes.txt` is present and `Feed(load_shapes=True)` (default), the real polyline for the matching original `shape_id` is looked up and each synthetic shape's stops are inserted into it at their nearest-segment position (`Shapes._insert_stops_into_real_shapes`). Without `shapes.txt` (or with `load_shapes=False`), every shape falls back to a straight line between consecutive stops.

---

## frequencies.txt

Defines headway-based service (a trip template repeated every N seconds within a time window) instead of exact `stop_times.txt` departures per trip.

**Mandatory columns**

| Column | Description |
|---|---|
| `trip_id` | The trip template being repeated |
| `start_time` | Start of the frequency-based service window |
| `end_time` | End of the frequency-based service window |
| `headway_secs` | Seconds between departures within the window |

**Optional columns**

| Column | Description |
|---|---|
| `exact_times` | `0` = frequency-based (approximate), `1` = schedule-based (exact departures at the headway interval) |

**Used by pyGTFSHandler**: `trip_id`, `start_time`, `end_time`, `headway_secs` **(used, all mandatory)**. See [Frequency Expansion](methodology/frequencies.md) for the full headway-correction and trip-expansion algorithm (`models/frequencies.py`), including midnight-crossing windows and the heuristic that auto-detects headways mistakenly given in minutes instead of seconds.

---

## transfers.txt

Defines special transfer rules between pairs of stops/routes/trips beyond the default connection behavior.

**Mandatory columns**: `from_stop_id`, `to_stop_id` (or the newer `from_*`/`to_*` location fields), `transfer_type`.
**Optional columns**: `min_transfer_time`, `from_route_id`, `to_route_id`, `from_trip_id`, `to_trip_id`.

**Used by pyGTFSHandler**: not read.

---

## pathways.txt

Describes accessible paths (walkways, stairs, elevators, ramps) connecting locations inside `stops.txt`, typically for station navigation.

**Mandatory columns**: `pathway_id`, `from_stop_id`, `to_stop_id`, `pathway_mode`, `is_bidirectional`.
**Optional columns**: `length`, `traversal_time`, `stair_count`, `max_slope`, `min_width`, `signposted_as`, `reversed_signposted_as`.

**Used by pyGTFSHandler**: not read.

---

## levels.txt

Describes the levels/floors of a station, referenced by `stops.txt.level_id` and `pathways.txt`.

**Mandatory columns**: `level_id`, `level_index`.
**Optional columns**: `level_name`.

**Used by pyGTFSHandler**: not read.

---

## feed_info.txt

Metadata about the feed itself (publisher, version, validity window).

**Conditionally required columns**: `feed_publisher_name`, `feed_publisher_url`, `feed_lang`.
**Optional columns**: `default_lang`, `feed_start_date`, `feed_end_date`, `feed_version`, `feed_contact_email`, `feed_contact_url`.

**Used by pyGTFSHandler**: not read.

---

## translations.txt

Provides translated values for fields in other files, keyed by table/field/language.

**Mandatory columns**: `table_name`, `field_name`, `language`, `translation`.
**Optional columns**: `record_id`, `record_sub_id`, `field_value`.

**Used by pyGTFSHandler**: not read.

---

## attributions.txt

Attributes the feed's data/roles (data source, operator, authority) to organizations.

**Optional columns (all)**: `attribution_id`, `agency_id`, `route_id`, `trip_id`, `organization_name`, `is_producer`, `is_operator`, `is_authority`, `attribution_url`, `attribution_email`, `attribution_phone`.

**Used by pyGTFSHandler**: not read.
