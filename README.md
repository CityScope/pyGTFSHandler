# PyGTFSHandler

**High-Performance GTFS Integration and Analysis using Polars.**

`PyGTFSHandler` is a comprehensive Python library designed for transit analysts and engineers to download, load, integrate, and perform advanced geospatial analysis on GTFS (General Transit Feed Specification) data.

Built on the high-performance Polars data manipulation framework, it transforms disparate GTFS files (`stops.txt`, `trips.txt`, `shapes.txt`, etc.) into a single, unified, analysis-ready lazy data structure, handling complexities like services crossing midnight, frequency interpolation, and spatial clustering automatically.

## Features

- **Polars-Powered Performance**: Uses Polars LazyFrames for highly efficient, memory-optimized, parallelized data processing.
- **Unified `Feed` Model**: Integrates all GTFS files into one denormalized data structure (`self.lf`) ready for complex queries.
- **Integrated Downloaders**: Easily search and download GTFS feeds from major repositories (e.g., Mobility Database).
- **Advanced Filtering at Initialization**: Filter data instantly by Area of Interest (AOI), date range, time range, specific routes, services, or stop groups.
- **Data Cleaning and Consistency**: Automatically corrects for common GTFS inconsistencies, including:
    - Accurate handling of trips that cross midnight.
    - Interpolation of stop times for trips defined in `frequencies.txt`.
    - Advanced, shape-based time interpolation for missing `stop_times`.
- **In-Depth Analysis Methods**: Specialized functions for calculating key public transport metrics:
    - Service Intensity (daily/hourly coverage).
    - Headway (Mean Interval) at Stops and Edges.
    - Mean Operating Speed at Stops and Edges.

## Installation

To install `PyGTFSHandler`, including the optional dependencies for geospatial plotting (`plot`) and OpenStreetMap utilities (`osm`), use pip:

```bash
pip install "pyGTFSHandler[osm,plot] @ git+https://github.com/CityScope/pyGTFSHandler.git"
```
Or, if you only need the core functionality:
```bash
git+https://github.com/CityScope/pyGTFSHandler.git
```

## Usage

The main entry point is the `Feed` object, which orchestrates the loading, filtering, and joining of all GTFS components.

### 0. Downloading GTFS Feeds (Optional)

`PyGTFSHandler` includes the `MobilityDatabaseClient` to search and download GTFS files directly from the Mobility Database API.

```python
from pygtfshandler.downloaders.mobility_database import MobilityDatabaseClient
from pygtfshandler.utils import get_city_geometry

# 1. Initialize the API Client
# Request your refresh token here: https://mobilitydatabase.org/
refresh_token = 'YOUR_REFRESH_TOKEN'
api = MobilityDatabaseClient(refresh_token)

city_name = "Cambridge, Massachusetts, USA"
aoi = get_city_geometry(city_name)
download_folder = "gtfs_orig_files"
revised_folder = "gtfs_revised_files"

# 2. Search for relevant feeds
feeds = api.search_gtfs_feeds(
    aoi=aoi, 
    country_code=['US'],
    is_official=True
)

# 3. Download the feeds
# This returns a list of paths to the downloaded ZIP files.
downloaded_paths = api.download_feeds(
    feeds=feeds,
    download_folder=download_folder,
    overwrite=False
)
print(f"Downloaded files: {downloaded_paths}")

# 4. Check for file validity and correct errors (Optional)
# This checks that the gtfs files are valid and corrects errors if possible. 
file_paths = []
for f in downloaded_paths:
    filename = os.path.splitext(os.path.basename(f))[0]
    if os.path.isdir(os.path.join(revised_folder,filename)):
        file_paths.append(os.path.join(revised_folder,filename))
    else:
        file_paths.append(gtfs_checker.preprocess_gtfs(f,download_folder))

print(f"Revised files: {file_paths}")
```

### 1. Initialization and Filtering

The `Feed` object takes a list of paths (or a single path) pointing to uncompressed GTFS directories or ZIP files.

```python
from pygtfshandler.feed import Feed
from datetime import datetime, date
import geopandas as gpd

# Example AOI (Area of Interest - a GeoDataFrame polygon)
aoi = gpd.read_file('path/to/my_city_boundary.geojson') 

# Initialize and load the feed, applying filters immediately
gtfs = Feed(
    gtfs_dirs=downloaded_paths,  # Use the list of paths from the download step
    aoi=aoi,  # Geospatial filter to only keep stops within this polygon
    stop_group_distance=100,  # Group stops less than 100 meters apart into a single parent_station
    start_date=date(2025, 11, 1),
    end_date=date(2025, 11, 30),
    route_types=['bus', 'subway'], # Only load these route types
    check_files=True
)

# The resulting integrated data is available as a Polars LazyFrame
print(gtfs.lf) 
# <LazyFrame... columns: ['trip_id', 'stop_id', 'departure_time', 'arrival_time', ...]>
```

### 2. Analysis Methods

Once the `Feed` object is initialized, you can use its methods to calculate specific transit metrics.

#### A. Service Intensity

Calculate the total number of scheduled stop times (service intensity) across a date range.

```python
from datetime import date

service_intensity_df = gtfs.get_service_intensity_in_date_range(
    start_date=date(2025, 11, 1),
    end_date=date(2025, 11, 7),
    date_type='weekday', # Optional: filter by specific days (e.g., 'monday', 'holiday')
    by_feed=True # Optional: break results down by original GTFS source file
).to_pandas()

print(service_intensity_df)
#       date  weekday  service_intensity  file_id
# 0 2025-11-03   Monday             120000        0
# 1 2025-11-04  Tuesday             125000        0
# ...
```

#### B. Mean Interval (Headway) at Stops

Calculate the average waiting time (headway) at stops, grouped by service direction or route. Results are in **minutes**.

```python
from datetime import time

# Analyze the best interval at stops during the morning peak
stop_interval_df = gtfs.get_mean_interval_at_stops(
    date=date(2025, 11, 5),
    start_time=time(hour=7),
    end_time=time(hour=9),
    at='parent_station',      # Spatial unit: use grouped stations
    by='shape_direction',    # Group services by their angular direction
    how='best',              # Return only the best interval found at that station
    n_divisions=2,           # Use 4 total directional clusters (2 forward, 2 backward)
).collect().to_pandas()

# Add human-readable names and coordinates for plotting
stop_interval_df = gtfs.add_stop_coords(stop_interval_df)
stop_interval_df = gtfs.add_route_names(stop_interval_df)
```

#### C. Mean Operating Speed at Edges

Calculate the average operating speed for segments (edges) between stops. Speeds are in **km/h**.

```python
edge_speed_df = gtfs.get_mean_speed_at_edges(
    date=date(2025, 11, 5),
    start_time=time(hour=7),
    end_time=time(hour=9),
    at='parent_station', # Defines the endpoints of the segments
    how='mean',          # Weighted average speed across all trips on that segment
    min_trips=5          # Only include segments with at least 5 observed trips
).collect().to_pandas()

# This method also adds a 'edge_linestring' column for geospatial mapping.
edge_speed_df = gtfs.add_stop_coords(edge_speed_df) 
```

### 3. General Data Filtering (`filter`)

Use the general `filter` method to extract a subset of the integrated data for custom analysis.

```python
filtered_data_lf = gtfs.filter(
    date=date(2025, 11, 5),
    start_time=time(hour=17),
    end_time=time(hour=19),
    frequencies=False, # Convert frequency-based trips into discrete stop times
    in_aoi=True        # Only include stops that were within the AOI boundary
)

# Convert to pandas/GeoPandas for final visualization or analysis
filtered_data_gdf = filtered_data_lf.collect().to_pandas()
# ... use pyGTFSHandler.plot_helper or your own plotting library
```

## GTFS Data Reference (The Integrated `lf` Columns)

When the `Feed` is initialized, it produces a master Polars LazyFrame (`gtfs.lf`) containing the denormalized data. The following is a reference of the key columns available for direct querying or analysis.

| Column | Description |
| :--- | :--- |
| **service\_id** | Unique identifier for a service pattern/calendar day. |
| **route\_id** | Unique identifier for the transit line (e.g., Red Line). |
| **trip\_id** | Unique identifier for the individual vehicle journey. |
| **stop\_id** | Unique identifier for each physical stop. |
| **parent\_station** | Identifier grouping related stops (e.g., platforms at one station). |
| **direction\_id** | Trip direction, usually `0` or `1`. |
| **shape\_id** | Unique identifier for the trip line geometry. |
| **route\_type** | GTFS route type (e.g., `3` for Bus). |
| **gtfs\_name** | Name of the original GTFS source file. |
| **file\_id** | Index of the GTFS file used. |
| **isin\_aoi** | Boolean: whether the stop is inside the initialized AOI. |

### Stop Times & Sequence

| Column | Description |
| :--- | :--- |
| **departure\_time** | Departure time from stop, in seconds after midnight (can exceed 86400s). |
| **arrival\_time** | Arrival time at stop, in seconds after midnight. |
| **stop\_sequence** | Ordering of stops within a trip (1 = first stop). |

### Shape / Geospatial Details

| Column | Description |
| :--- | :--- |
| **shape\_dist\_traveled** | Distance from the start of the shape, in meters. |
| **shape\_total\_distance** | Total distance of the trip, in meters. |
| **shape\_direction** | Average forward direction of travel at this stop (angle 0[north]-360). |
| **shape\_direction\_backwards** | Average backward direction of travel at this stop (angle 0[north]-360). |

### Frequency Data (If applicable)

| Column | Description |
| :--- | :--- |
| **start\_time** | Start of the frequency window, in seconds after midnight. |
| **end\_time** | End of the frequency window, in seconds after midnight. |
| **headway\_secs** | Interval between repeated trips, in seconds. |
| **n\_trips** | Number of trips represented by the frequency block. |

