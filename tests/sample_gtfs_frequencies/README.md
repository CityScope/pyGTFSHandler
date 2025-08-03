# sample_gtfs_frequencies

This GTFS sample dataset is designed to test **frequency-based trips** defined in `frequencies.txt`. It simulates a simple transit line with a repeating trip every 10 minutes.

---

## ✅ Purpose

To verify:
- Proper handling of `frequencies.txt`
- Generation of multiple trips from a single base trip (`F1`)
- Interpolation of missing `arrival_time`
- Correct filtering in time-based queries

---

## 📂 Files

### `calendar.txt`
Defines the weekday service `FREQ`, active May 1–31, 2024.

### `calendar_dates.txt`
Adds a special service instance on **May 13, 2024** (for direct testing).

### `routes.txt`
Defines a single route `R2`.

### `trips.txt`
Defines one base trip `F1` (used in `frequencies.txt`) on route `R2`, using shape `S2`.

### `stop_times.txt`
Defines 3 stops for trip `F1`, with a missing `arrival_time` at `S2` to test interpolation.

### `frequencies.txt`
Repeats `F1` every 600 seconds (10 minutes) from 08:00 to 09:00.

### `shapes.txt`
Defines a simple 3-point shape (`S2`) to enable shape-time interpolation.

---

## 🔬 Testing Ideas

- `feed._frequencies_to_stop_times()` correctly expands `F1` to 7 trips.
- `feed.filter_by_time_range()` restricts based on `start_time`, `end_time`.
- `feed.lf` includes proper `start_time`, `n_trips`, and interpolated `arrival_time`.
- `feed.get_service_intensity_in_date_range()` reflects repeated trips.

---

## 🧪 Tip
Use `test/sample_gtfs_frequencies` in your test suite via a fixture:

```python
@pytest.fixture
def sample_gtfs_frequencies():
    return Path("test/sample_gtfs_frequencies")
```

Then load it:
```python
feed = Feed(gtfs_dirs=sample_gtfs_frequencies)