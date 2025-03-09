# PyGTFSHandler

**A Python package to download, load, and pre-process GTFS public transport timetable files.**

`PyGTFSHandler` is a comprehensive Python library designed to handle GTFS (General Transit Feed Specification) data. It provides functionalities to download, load, and pre-process GTFS files.

## Features

- **Download and Load GTFS Files**: Easily download and load GTFS files into your Python environment.
- **Pre-process GTFS Data**: Clean and prepare GTFS data for analysis.
- **Geographic Filtering**: Select trips within specified geographic bounds.
- **Stop and Trip Grouping**: Cluster stops and trips based on distance and overlap.
- **Service Date and Time Filtering**: Filter trips based on service date and time. It can search the date with the maximum amount of services.
- **Route Filtering**: Filter routes based on custom criteria.

## Installation

To install `PyGTFSHandler`, use pip:

```bash
pip install git+https://github.com/MiguelUrenaPliego/pyGTFSHandler.git
```

## Usage

### Initialization

The `GTFS` object loads and pre-processes a list of uncompressed GTFS files.

```python
from pygtfshandler import GTFS

gtfs = GTFS(
    gtfs_dir='path/to/gtfs_files',  # List of paths pointing to uncompressed folders with .txt files in GTFS format
    service_date='YYYY-MM-DD',  # Date to select the trips. If 'max' select the date with the maximum amount of services.
    start_time='00:00:00',  # Start time for filtering trips
    end_time='00:00:00',  # End time for filtering trips
    bounds=None,  # Polygon with geographic bounds to select trips within the bounds
    strict_bounds=True,  # If True, delete stops outside the bounds
    stop_group_distance=0,  # Cluster all stops using this distance
    trip_group_distance=0,  # Trips with stops less than this distance apart are considered overlapping
    trip_group_overlap=0.75,  # Minimum percentage of the trip that has to overlap with another to be considered a branch of the same line
    correct_stop_sequence=True,  # Revise the stop_sequence column
    crs=4326,  # EPSG code for geographic coordinates
    route_filter=None,  # Filter routes if the trip contains
    all_stops=True  # If False, delete stops that do not have any trips
)
```

### Methods

- **`get_tph_by_line(trip_groups: bool = True)`**: Returns a DataFrame with the number of trips per hour for each `rep_trip_id`.

- **`get_tph_at_stops(stop_groups: bool = True)`**: Returns a DataFrame with the number of trips per hour for each `stop_id`.

- **`get_lines_gdf()`**: Returns a GeoDataFrame with line geometry for each `rep_trip_id`.

- **`get_line_stops_gdf(trip_groups: bool = True)`**: Returns a GeoDataFrame with records for each stop for each `rep_trip_id`.

- **`get_line_time(trip_groups: bool = True)`**: Returns a DataFrame with the total service time for each `rep_trip_id`.

- **`get_service_hours_by_line()`**: Returns a DataFrame with the number of service hours for each `rep_trip_id`.

- **`get_routes_by_stops(stop_groups: bool = True)`**: Returns a DataFrame with a list of stops for each `rep_trip_id`.

- **`get_total_trips_by_line(trip_groups: bool = True)`**: Returns a DataFrame with the total number of trips for each `rep_trip_id`.

- **`get_schedule_symmetry(trip_groups: bool = True)`**: Returns a number between 0 and 1 indicating how symmetric the timetable is.

- **`get_cph_by_line(trip_groups: bool = True, stop_groups: bool = True, agg: str = 'max', exclude_first_stop: bool = False)`**: Returns a DataFrame with the number of trips per hour for each `rep_trip_id`.
  - **`agg`**: The aggregation method used to determine the frequency of trips. Options include:
    - **`'max'`**: Selects the line with the maximum frequency.
    - **`'sum'`**: Adds all the frequencies of all lines.
    - **`'agg_sum'`**: Sums all frequencies to the maximum frequency, multiplied by an aggregation factor (`agg_factor`).
  - **`exclude_first_stop`**: It is recommended to set this to True to avoid counting the same line twice.
  
- **`get_cph_at_stops(trip_groups: bool = True, stop_groups: bool = True, agg: str = 'max', agg_factor: float = 1, exclude_first_stop: bool = True)`**: Returns a DataFrame with the number of trips per hour for each `stop_id`.

- **`stop_service_quality(frequencies: list, start_time=None, end_time=None, agg='max', agg_factor: float = 1, exclude_first_stop: bool = True)`**: Returns an integer for each stop indicating the service quality based on the frequency of trips.


