# sample_gtfs_midnight

This GTFS sample dataset is built to test **trips that cross midnight**, with times exceeding 24:00:00.

---

## ✅ Purpose

To verify:
- Correct handling of `arrival_time`/`departure_time` > 24:00:00
- Suffixing `service_id` with `_night` for trips crossing days
- Accurate filtering by time and date

---

## 📂 Files

### `calendar.txt`
Defines weekday service `NIGHT` active from May 1–31, 2024.

### `calendar_dates.txt`
Adds `NIGHT` service on May 14, 2024.

### `routes.txt`
Defines route `R3` called "Night Line".

### `trips.txt`
Defines a single trip `N1` on route `R3`, using shape `S3`.

### `stop_times.txt`
Trip `N1` has stop times crossing midnight:
- Starts at 23:55
- Second stop at 24:05
- Last stop at 25:10 (01:10 AM next day)

### `shapes.txt`
Three-point shape used to allow distance interpolation.

---

## 🔬 Testing Ideas

- `Feed.lf` should replace `service_id` with `NIGHT_night` for trip `N1`.
- Stop times should still sort correctly.
- `get_services_in_date()` should recognize `NIGHT_night` as a distinct service.
- `get_service_intensity_in_date_range()` should count night services separately.
- Time filtering should still apply properly (e.g., 23:00–01:30 window).

---

## 🧪 Tip
In your test code, use:
```python
feed = Feed(gtfs_dirs=sample_gtfs_midnight)
```
Then assert that:
```python
assert any("_night" in sid for sid in feed.lf.select("service_id").unique().collect()["service_id"])
