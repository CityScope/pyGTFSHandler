import sys 
sys.path.append("/home/miguel/Documents/Proyectos/PTLevelofService/gtfs/pyGTFSHandler")

import os
import argparse

from pyGTFSHandler.feed import Feed
import pyGTFSHandler.gtfs_checker as gtfs_checker
import pyGTFSHandler.processing_helper as processing_helper
from datetime import datetime, date, time
import polars as pl 
import geopandas as gpd
import json
import copy 

parser = argparse.ArgumentParser(description="Process GTFS feed.")

# Mandatory
parser.add_argument('--orig_file', required=True, help='Path to original GTFS file')

# Optional arguments with defaults
parser.add_argument('--processed_gtfs_folder', default=None, help='Folder for processed GTFS')
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
parser.add_argument('--n_stops_speeds', type=int, default=5)
parser.add_argument('--speed_direction', type=str, default="both", help='Direction to compute speed: forward, backward, both')

# Parse all arguments
args = parser.parse_args()

# -------------------------
# Convert to usable Python variables
# -------------------------
orig_file = args.orig_file
processed_gtfs_folder = args.processed_gtfs_folder

# Time conversion
start_time = datetime.strptime(args.start_time, "%H:%M:%S").time() if args.start_time else time(hour=8)
end_time = datetime.strptime(args.end_time, "%H:%M:%S").time() if args.end_time else time(hour=20)

# Date conversion
start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

date_type = args.date_type
stop_group_distance = args.stop_group_distance

# JSON strings to Python dicts
route_type_mapping = json.loads(args.route_type_mapping)
if route_type_mapping is None:
    route_type_mapping = {'any':'all'}

route_speed_mapping = json.loads(args.route_speed_mapping)
if isinstance(route_speed_mapping,list):
    route_speed_mapping = {
        k:route_speed_mapping for k in route_type_mapping.keys()
    }

n_stops_speeds = args.n_stops_speeds
speed_direction = args.speed_direction

# -------------------------
# Process
# -------------------------

filename = os.path.splitext(os.path.basename(orig_file))[0]
if os.path.isdir(os.path.join(processed_gtfs_folder,filename)):
    file_path = os.path.join(processed_gtfs_folder,filename)
else:
    file_path = gtfs_checker.preprocess_gtfs(orig_file,processed_gtfs_folder)

gtfs = Feed(
    file_path,
    stop_group_distance=stop_group_distance, # Group stops into one that are less than x meters apart. This creates or updates the parent_station column
    start_date=start_date,
    end_date=end_date,
)

selected_service_intensity = gtfs.get_service_intensity_in_date_range(
    start_date=None, # If None take the feed min date
    end_date=None, # If None take the feed max date
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
        n_stops=n_stops_speeds, # Number of stops to pick to compute the speed
    )
    if isinstance(stop_speed_df,pl.DataFrame):
        stop_speed_df = stop_speed_df.lazy()

    gtfs_lf = gtfs_lf.join(stop_speed_df.select(['parent_station','route_id','speed']),on=['parent_station','route_id'],how='left')


stop_interval_df = []
for route_type_simple in route_type_mapping.keys():
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
            ).with_columns(pl.lit(route_type_simple).alias("route_type_simple"))
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
                    pl.lit(s).alias("min_speed"),
                )
            )

stop_interval_df = (
    pl.concat(stop_interval_df)
)

# Mapping dictionary
route_type_mapping = {k: i for i, k in enumerate(["bus", "tram", "train"])}

expr = None
for k, v in route_type_mapping.items():
    if expr is None:
        expr = pl.when(pl.col("route_type_simple") == k).then(v)
    else:
        expr = expr.when(pl.col("route_type_simple") == k).then(v)

stop_interval_df = stop_interval_df.with_columns(
    expr.alias("route_type_simple_int")
)

stop_interval_df = stop_interval_df.sort(
    ["parent_station","route_type_simple_int","min_speed"]
).unique(
    ["parent_station","mean_interval"],keep='last'
).drop("route_type_simple_int")

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

