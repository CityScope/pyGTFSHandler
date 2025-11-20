import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta, date, time
from . import gtfs_checker

ROUTE_TYPES_TRANSLATOR = {
    "bus": ["bus"],
    "tram": ["tram", "subway", "rail", "ferry", "cable car", "gondola", "funicular"],
    "rail": ["subway", "rail"],
}

ROUTE_TYPES = ["rail", "tram", "bus"]

SERVICE_MATRIX = pd.DataFrame(
    {
        "interval": [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 360],
        "bus": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "tram": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "rail": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
)

DISTANCE_MATRIX = pd.DataFrame(
    {
        "service_quality": range(1, 13),
        300: ["A1", "A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D", "E"],
        500: ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D", "D", "E"],
        750: ["A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D", "D", "D", "E"],
        1000: ["A3", "B1", "B2", "B3", "C1", "C2", "C3", "D", "D", "D", "E", "E"],
        1250: ["B3", "C1", "C2", "C3", "D", "D", "D", "E", "E", "E", "F", "F"],
        1500: ["C1", "C2", "C3", "D", "D", "D", "E", "E", "E", "F", "F", "F"],
        2000: ["D", "D", "D", "E", "E", "E", "F", "F", "F", "F", "F", "F"],
    }
)

LEVEL_OF_SERVICES = [
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "C3",
    "D",
    "E",
    "F",
]


def most_frequent_row_index(values, bins=5):
    """
    Returns the index (usable with df.iloc) of the row whose value is
    closest to the median of the most populated histogram bin.

    Parameters:
        values (array-like): Numeric values (e.g., from a column).
        bins (int): Number of histogram bins to use.

    Returns:
        int: Index (iloc-style) of the most typical row.
    """
    values = np.asarray(values)

    counts, bin_edges = np.histogram(values, bins=bins)

    # Most frequent bin
    max_bin_idx = np.argmax(counts)
    bin_start = bin_edges[max_bin_idx]
    bin_end = bin_edges[max_bin_idx + 1]

    # Mask for values inside the most frequent bin
    in_bin_mask = (values >= bin_start) & (values <= bin_end)
    in_bin_indices = np.where(in_bin_mask)[0]

    if len(in_bin_indices) == 0:
        return None

    # Median of values inside the bin
    median_in_bin = np.mean(values[in_bin_indices])

    # Find the index of the value closest to the median
    closest_idx_in_bin = np.argmin(np.abs(values[in_bin_indices] - median_in_bin))
    row_index = in_bin_indices[closest_idx_in_bin]

    return int(row_index)


def assign_service_quality_to_interval(interval, route_type, service_matrix=SERVICE_MATRIX):
    return service_matrix.loc[service_matrix['interval'] >= interval, route_type].iloc[0]
    
# def assign_service_quality_to_interval(mean_interval_df, service_matrix, route_type):
#     service_matrix_intervals = list(service_matrix["interval"])  # e.g., [10, 20, 30]
#     service_matrix_values = list(
#         service_matrix[route_type]
#     )  # e.g., [1, 2, 3] — must be same length
#     # Start with the first condition
#     expr = pl.when(pl.col("interval").is_null()).then(None)

#     # Add interval-based conditions
#     cond_expr = pl.when(pl.col("interval") < service_matrix_intervals[0]).then(
#         service_matrix_values[0]
#     )

#     for i in range(1, len(service_matrix_intervals)):
#         cond_expr = cond_expr.when(
#             pl.col("interval") < service_matrix_intervals[i]
#         ).then(service_matrix_values[i])

#     # Final fallback
#     cond_expr = cond_expr.otherwise(service_matrix_values[-1] + 1)

#     # Chain the null-check with the conditional logic
#     expr = expr.otherwise(cond_expr)

#     # Apply to DataFrame
#     mean_interval_df = mean_interval_df.with_columns(expr.alias("service_quality"))

#     return mean_interval_df


# def get_minutes_diff(start_time, end_time):
#     # If both are datetime.datetime, subtract directly
#     if isinstance(start_time, datetime) and isinstance(end_time, datetime):
#         diff = end_time - start_time
#     # If both are datetime.time, combine with arbitrary date (today)
#     elif isinstance(start_time, time) and isinstance(end_time, time):
#         dt_start = datetime.combine(date.today(), start_time)
#         dt_end = datetime.combine(date.today(), end_time)
#         diff = dt_end - dt_start
#         # Handle overnight time ranges (end before start)
#         if diff.total_seconds() < 0:
#             diff += timedelta(days=1)
#     else:
#         raise TypeError(
#             "start_time and end_time must both be datetime or both be time objects"
#         )

#     return diff.total_seconds() / 60  # minutes


# def get_single_service_quality(
#     gtfs,
#     date,
#     start_time,
#     end_time,
#     service_matrix=SERVICE_MATRIX,
#     route_types=ROUTE_TYPES,
#     route_types_translator=ROUTE_TYPES_TRANSLATOR,
# ):
#     # Build route_id → "short: long" mapping
#     route_long_replacement = (
#         gtfs.routes.lf.collect()
#         .with_columns(
#             pl.concat_str(
#                 [pl.col("route_id"), pl.lit(": "), pl.col("route_long_name")]
#             ).alias("label"),
#             pl.concat_str(
#                 [pl.col("route_id"), pl.lit("_file_id_"), pl.col("file_id")]
#             ).alias("route_id"),
#         )
#         .select(["route_id", "label"])
#         .to_dict(as_series=False)
#     )
#     route_long_replacement = dict(
#         zip(route_long_replacement["route_id"], route_long_replacement["label"])
#     )

#     route_short_replacement = (
#         gtfs.routes.lf.collect()
#         .with_columns(
#             (pl.col("route_short_name")).alias("label"),
#             pl.concat_str(
#                 [pl.col("route_id"), pl.lit("_file_id_"), pl.col("file_id")]
#             ).alias("route_id"),
#         )
#         .select(["route_id", "label"])
#         .to_dict(as_series=False)
#     )
#     route_short_replacement = dict(
#         zip(route_short_replacement["route_id"], route_short_replacement["label"])
#     )

#     all_service_dfs = []

#     for route_type in route_types:
#         gtfs_route_types = (
#             route_types_translator[route_type] if route_types_translator else route_type
#         )

#         # Get aggregated service at stops
#         service_df = gtfs.get_mean_intervall_at_stops(
#             date=date,
#             start_time=start_time,
#             end_time=end_time,
#             route_types=gtfs_route_types,
#             on="parent_station",
#             method="agg",
#             n_divisions=1,
#         )

#         # Enrich with readable route names and compute stats
#         service_df = service_df.with_columns(
#             [
#                 (pl.col("mean_interval") / 60).alias("interval"),
#                 pl.col("departure_times").list.len().alias("n_services"),
#                 (
#                     pl.col("route_ids")
#                     .list.eval(pl.element().replace(route_long_replacement))
#                     .list.unique()
#                 ).alias("route_long_names"),
#                 (
#                     pl.col("route_ids")
#                     .list.eval(pl.element().replace(route_short_replacement))
#                     .list.unique()
#                 ).alias("route_names"),
#             ]
#         )

#         interval_if_one_service = get_minutes_diff(start_time, end_time)

#         service_df = service_df.with_columns(
#             pl.when(pl.col("n_services") == 1)
#             .then(interval_if_one_service)
#             .otherwise(pl.col("interval"))
#             .alias("interval")
#         )

#         # Assign quality score
#         service_df = assign_service_quality_to_interval(
#             service_df, service_matrix, route_type
#         )

#         # Add route_type as column
#         service_df = service_df.with_columns(pl.lit(route_type).alias("route_type"))

#         # Select only needed columns
#         service_df = service_df.select(
#             [
#                 "parent_station",
#                 "shape_directions",
#                 "interval",
#                 "n_services",
#                 "route_names",
#                 "route_long_names",
#                 "service_quality",
#                 "route_type",
#             ]
#         )

#         all_service_dfs.append(service_df)

#     # Combine all route_type DataFrames
#     combined_df = pl.concat(all_service_dfs)

#     # For each shape_direction, keep the row with the best (min) service_quality
#     final_df = (
#         combined_df.sort("service_quality")
#         .group_by("parent_station")
#         .agg(
#             [
#                 pl.first("service_quality"),
#                 pl.first("interval"),
#                 pl.first("n_services"),
#                 pl.first("route_names"),
#                 pl.first("route_long_names"),
#                 pl.first("shape_directions"),
#                 pl.first("route_type"),
#             ]
#         )
#     )

#     return final_df


# def get_service_quality(
#     results_path,
#     gtfs,
#     dates,
#     times,
#     service_matrix=SERVICE_MATRIX,
#     route_types=ROUTE_TYPES,
#     route_types_translator=ROUTE_TYPES_TRANSLATOR,
# ):
#     if not isinstance(dates, list):
#         dates = [dates]

#     if not (isinstance(times, list) and all(isinstance(t, list) for t in times)):
#         times = [times]

#     results = []
#     for processing_date in dates:
#         date_str = processing_date.strftime("%Y%m%d")  # for filename
#         weekday_str = processing_date.strftime("%A").lower()
#         stops_gdf = gtfs.stops.gdf[["stop_id", "parent_station", "geometry"]].copy()
#         stops_gdf = stops_gdf.merge(
#             gtfs.stops.lf.select(["stop_id", "stop_name"]).collect().to_pandas(),
#             on="stop_id",
#             how="left",
#         )

#         for time_bounds in times:
#             start_int, end_int = time_bounds[0], time_bounds[1]
#             start_time = time(hour=time_bounds[0])
#             if time_bounds[1] == 24:
#                 end_time = time(hour=23, minute=59, second=59)
#             else:
#                 end_time = time(hour=time_bounds[1])

#             service_df = get_single_service_quality(
#                 gtfs,
#                 processing_date,
#                 start_time,
#                 end_time,
#                 service_matrix,
#                 route_types,
#                 route_types_translator,
#             )

#             service_df = service_df.rename(
#                 {
#                     "service_quality": f"service_quality_{start_int}h_{end_int}h",
#                     "interval": f"interval_{start_int}h_{end_int}h",
#                     "n_services": f"n_services_{start_int}h_{end_int}h",
#                     "route_names": f"route_names_{start_int}h_{end_int}h",
#                     "route_long_names": f"route_long_names_{start_int}h_{end_int}h",
#                     "shape_directions": f"shape_directions_{start_int}h_{end_int}h",
#                     "route_type": f"route_type_{start_int}h_{end_int}h",
#                 }
#             )

#             stops_gdf = stops_gdf.merge(
#                 service_df.to_pandas(), on="parent_station", how="left"
#             )

#         # List route priority
#         priority = route_types

#         # Get all columns with 'route_type' in their name
#         route_type_cols = [col for col in stops_gdf.columns if "route_type" in col]

#         def prioritize_route_type(row):
#             for route in priority:
#                 if any(route in str(row[col]) for col in route_type_cols):
#                     return route
#             return None

#         stops_gdf["route_type"] = stops_gdf.apply(prioritize_route_type, axis=1)
#         stops_gdf.to_file(
#             results_path + f"/stop_service_quality_{date_str}_{weekday_str}.gpkg"
#         )
#         results.append(
#             results_path + f"/stop_service_quality_{date_str}_{weekday_str}.gpkg"
#         )

#     if len(results) == 1:
#         results = results[0]

#     return results
