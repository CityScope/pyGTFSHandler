import sys 
sys.path.append("/home/miguel/Documents/Proyectos/PTLevelofService/gtfs/pyGTFSHandler")

import os
import argparse

from pyGTFSHandler.feed import Feed
import pyGTFSHandler.gtfs_checker as gtfs_checker
import pyGTFSHandler.processing_helper as processing_helper
from datetime import datetime, date, time
import polars as pl 
from shapely import wkt
import geopandas as gpd
import numpy as np
import json
import copy 
import warnings 

parser = argparse.ArgumentParser(description="Process GTFS feed.")

# Mandatory
parser.add_argument('--orig_file', required=False, help='Path to original GTFS file')

# Optional arguments with defaults
parser.add_argument('--processed_gtfs_folder', default=None, help='Folder for processed GTFS')
parser.add_argument('--aoi', type=str, default=None, help='Area of interest (as wkt string in epsg 4326)')
parser.add_argument('--start_time', type=str, default="08:00:00", help='Start time (HH:MM:SS)')
parser.add_argument('--end_time', type=str, default="20:00:00", help='End time (HH:MM:SS)')
parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
parser.add_argument('--date_type', type=str, default='businessday', help='businessday, holiday, non_businessday, weekend')
parser.add_argument('--stop_group_distance', type=int, default=100)
parser.add_argument('--route_type_mapping', type=str, default=json.dumps({
    'bus':'all',
    'tram':[0,1,2,4,5,6,7],
    'rail':[1,2]
}), help='JSON string mapping route types')
parser.add_argument('--route_speed_mapping', type=str, default=json.dumps([0,10,15,20,25,30]), help='JSON string mapping route types and speeds')
parser.add_argument('--time_step_speeds', type=int, default=5)
parser.add_argument('--speed_direction', type=str, default="both", help='Direction to compute speed: forward, backward, both')
parser.add_argument(
    '--fast_check',
    action='store_false',    # sets check_files = False when used
    dest='check_files',
    help='Skip full GTFS check. Default is to check files.'
)
parser.add_argument(
    '--overwrite',
    action='store_true',     # sets overwrite = True when used
    help='Overwrite all existing files. Default is False.'
)
parser.add_argument(
    '--params_file',
    type=str,
    default=None,
    help='Path to JSON file with all parameters (alternative to passing each argument individually)'
)
# Parse all arguments
args = parser.parse_args()

# If params_file is provided, override args with the JSON content
if args.params_file is not None:
    with open(args.params_file) as f:
        params = json.load(f)

    # Override args attributes
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)

# Remove required constraint for orig_file when using JSON
if not hasattr(args, 'orig_file') or args.orig_file is None:
    raise ValueError("Parameter 'orig_file' is required")

# -------------------------
# Convert to usable Python variables
# -------------------------
orig_file = args.orig_file
processed_gtfs_folder = args.processed_gtfs_folder

check_files = args.check_files
overwrite = args.overwrite

# aoi 
if args.aoi is None:
    aoi = None 
else:
    aoi = gpd.GeoDataFrame(
        geometry=[wkt.loads(args.aoi)],
        crs="EPSG:4326"
    )

# Time conversion
start_time = datetime.strptime(args.start_time, "%H:%M:%S").time() if args.start_time else time(hour=8)
end_time = datetime.strptime(args.end_time, "%H:%M:%S").time() if args.end_time else time(hour=20)

# Date conversion
start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

date_type = args.date_type
stop_group_distance = args.stop_group_distance

def ensure_dict(value):
    """Return a dict: load from JSON string if needed, otherwise return as-is."""
    if isinstance(value, dict):
        return value
    elif isinstance(value, str):
        return json.loads(value)
    else:
        raise TypeError(f"Expected dict or JSON string, got {type(value).__name__}")

def check_route_type_mapping(route_type_mapping):
    seen_values = set()
    new_mapping = {}

    for k, v in reversed(list(route_type_mapping.items())):
        v_tuple = tuple(v) if isinstance(v, list) else v
        if v_tuple not in seen_values:
            new_mapping[k] = v
            seen_values.add(v_tuple)

    # Reverse again to restore original order
    new_mapping = dict(reversed(list(new_mapping.items())))
    return new_mapping

# JSON strings to Python dicts
route_type_mapping = ensure_dict(args.route_type_mapping)
if route_type_mapping is None:
    route_type_mapping = {'any':'all'}

if not isinstance(route_type_mapping,dict):
    route_type_mapping = {'any':route_type_mapping}

for k in route_type_mapping:
    route_types = []
    if route_type_mapping[k] is None:
        route_types = 'all'
    elif route_type_mapping[k] == 'all':
        route_types = 'all'
    elif isinstance(route_type_mapping,(str,int)):
        route_types = int(route_type_mapping[k])
    else:
        for i in route_type_mapping[k]:
            if (i is None):
                route_types = 'all'
                break
            elif route_types == 'all':
                route_types = 'all'
                break  
            else:
                route_types.append(int(i))

    route_type_mapping[k] = route_types

route_type_mapping = check_route_type_mapping(route_type_mapping) 

all_route_types = []
for k in route_type_mapping:
    r = route_type_mapping[k]
    if not isinstance(r,list):
        r = [r]
    
    all_route_types += r 

if ('all' in all_route_types) or (None in all_route_types): 
    all_route_types = 'all'

route_speed_mapping = ensure_dict(args.route_speed_mapping)
if isinstance(route_speed_mapping,list):
    route_speed_mapping = {
        k:route_speed_mapping for k in route_type_mapping.keys()
    }

time_step_speeds = args.time_step_speeds
speed_direction = args.speed_direction


# -------------------------
# Process
# -------------------------

filename = os.path.splitext(os.path.basename(orig_file))[0]
if (not overwrite) and os.path.isdir(os.path.join(processed_gtfs_folder,filename)):
    file_path = os.path.join(processed_gtfs_folder,filename)
else:
    if check_files:
        file_path = gtfs_checker.preprocess_gtfs(orig_file,processed_gtfs_folder)
    else:
        file_path = gtfs_checker.unzip(orig_file,processed_gtfs_folder)

if (not overwrite) and os.path.isfile(os.path.join(file_path,"stop_intervals.gpkg")):
    warnings.warn(f"Skipping {file_path} as stop_intervals.gpkg already exists.") 
else:
    try:
        gtfs = Feed(
            file_path,
            stop_group_distance=stop_group_distance,
            start_date=start_date,
            end_date=end_date,
            route_types=all_route_types,
            check_files=not check_files
        )
    except Exception as e:
        error_msg = str(e)
        if (
            "No trips found in time range" in error_msg
            or "No routes found with filter" in error_msg
            or "No trips with your id filters and filters" in error_msg
            or "No stops found inside your aoi" in error_msg
        ):
            # Raise a clean exception with a message
            raise RuntimeError(f"Error loading feed {file_path}: {error_msg}") from None
        else:
            # Re-raise unexpected exceptions with full traceback
            raise

    selected_service_intensity = gtfs.get_service_intensity_in_date_range(
        start_date=None, # If None take the feed min date
        end_date=None, # If one take the feed max date
        date_type=date_type # Could be something like 'holiday', 'businessday', 'non_businessday', or 'monday' to only consider some dates from the range.
    )
    selected_service_intensity = selected_service_intensity.to_pandas()
    idx = processing_helper.most_frequent_row_index(selected_service_intensity['service_intensity'])
    selected_day = selected_service_intensity.iloc[idx]['date'].to_pydatetime()

    gtfs_lf = gtfs.filter(
            date=selected_day,
            start_time=start_time,
            end_time=end_time,
            frequencies=False,
            in_aoi=True,
            delete_last_stop=True
        )

    if route_speed_mapping is not None:
        stop_speed_df = gtfs.get_mean_speed_at_stops(
            date=selected_day,
            start_time=start_time,
            end_time=end_time,
            route_types = 'all',
            by = "route_id", # Speed is computed for every 'trip_id' and grouped by this column with the how method
            at = 'parent_station', # Compute speed for every 'parent_station' 'stop_id' or 'route_id'
            how="mean", # How to group individual trip speeds 'mean' 'max' or 'min'
            direction=speed_direction, # Compute speed in 'forward' 'backward' or 'both' directions (walking n_stops in direction)
            time_step=time_step_speeds, # Number of stops to pick to compute the speed
        )
        if isinstance(stop_speed_df,pl.DataFrame):
            stop_speed_df = stop_speed_df.lazy()

        gtfs_lf = gtfs_lf.join(stop_speed_df.select(['parent_station','route_id','speed']),on=['parent_station','route_id'],how='left')

    unique_route_types = gtfs.lf.select("route_type").unique().collect()['route_type'].to_list()
    invalid_keys  = []
    for k in route_type_mapping.keys():
        route_types = route_type_mapping[k]
        if (route_types is None):
            continue 

        intersection = set(route_types) & set(unique_route_types)
        if not intersection:
            invalid_keys.append(k)
        else:
            route_type_mapping[k] = sorted(list(intersection))
            
    for k in invalid_keys:
        del route_type_mapping[k] 
        
    route_type_mapping = check_route_type_mapping(route_type_mapping) 
    stop_interval_df = []
    route_type_simple_int = -1
    for route_type_simple in route_type_mapping.keys():
        route_type_simple_int += 1 
        route_types = route_type_mapping[route_type_simple]
        gtfs_lf_route_types = gtfs._filter_by_route_type(gtfs_lf,route_types)
        
        if route_speed_mapping is None: 
            stop_interval_df.append(
                gtfs._get_mean_interval_at_stops(
                    gtfs_lf_route_types,
                    date=selected_day,
                    start_time=start_time,
                    end_time=end_time,
                    route_types=None, 
                    by = "shape_direction", # Interval is computed for all 'trip_id' grouped by this column and sorted by 'departure_time'
                    at = "parent_station", # Where to compute the interval 'stop_id' 'parent_station'
                    how = "best", 
                    # 'best' pick the route with best interval, 
                    # 'mean' Combine all intervals of all routes, 
                    # 'all' return results per stop and route
                    n_divisions=1, # Number of divisions for by = 'shape_direction'
                ).with_columns(
                    pl.lit(route_type_simple).alias("route_type_simple"),
                    pl.lit(route_type_simple_int).alias("route_type_simple_int"),
                    pl.lit(0).alias("min_speed"),
                )
            )
        else:
            speeds = route_speed_mapping[route_type_simple] 
            for s in speeds: 
                if s > 0:
                    gtfs_lf_s = gtfs_lf_route_types.filter(pl.col("speed") > s)
                else:
                    gtfs_lf_s = copy.deepcopy(gtfs_lf_route_types)

                stop_interval_df.append(
                    gtfs._get_mean_interval_at_stops(
                        gtfs_lf_s,
                        date=selected_day,
                        start_time=start_time,
                        end_time=end_time,
                        route_types=route_types, 
                        by = "shape_direction", # Interval is computed for all 'trip_id' grouped by this column and sorted by 'departure_time'
                        at = "parent_station", # Where to compute the interval 'stop_id' 'parent_station'
                        how = "best", 
                        # 'best' pick the route with best interval, 
                        # 'mean' Combine all intervals of all routes, 
                        # 'all' return results per stop and route
                        n_divisions=1, # Number of divisions for by = 'shape_direction'
                    ).with_columns(
                        pl.lit(route_type_simple).alias("route_type_simple"),
                        pl.lit(route_type_simple_int).alias("route_type_simple_int"),
                        pl.lit(s).alias("min_speed"),
                    )
                )

    stop_interval_df = (
        pl.concat(stop_interval_df)
    )

    stop_interval_df = stop_interval_df.sort(
        ["parent_station","route_type_simple_int","min_speed"]
    ).unique(
        ["parent_station","route_type_simple_int", "mean_interval"],keep='last'
    ) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! You could take route_type_simple_int out of the unique to have only one route type if some less good types include better ones (bus includes rail)

    if route_speed_mapping is None: 
        stop_interval_df = stop_interval_df.drop("min_speed")
        
    stop_interval_df = stop_interval_df.join(
        gtfs.stops.lf.select(['stop_id', 'parent_station']).collect(),
        on='parent_station',
        how='right'
    )
    stop_interval_df = gtfs.add_stop_coords(stop_interval_df)
    stop_interval_df = gtfs.add_route_names(stop_interval_df)

    stop_interval_df = stop_interval_df.to_pandas()

    stop_interval_df = gpd.GeoDataFrame(
        stop_interval_df,
        geometry=gpd.points_from_xy(stop_interval_df['stop_lon'],y=stop_interval_df['stop_lat']),
        crs=4326
    )
    stop_interval_df = stop_interval_df[stop_interval_df.geometry.is_valid]
    stop_interval_df = stop_interval_df.sort_values("parent_station").reset_index(drop=True)
    stop_interval_df['date'] = selected_day

    stop_interval_df.to_file(os.path.join(file_path,"stop_intervals.gpkg"))

