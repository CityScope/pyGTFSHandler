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

def most_frequent_row_range(values, bins=5):
    values = np.asarray(values)

    # Thresholding
    threshold = 0.5 * np.percentile(values, 90)
    cleaned = values.copy()
    cleaned[cleaned < threshold] = 0
    nonzero_mask = cleaned > 0
    nz_vals = cleaned[nonzero_mask]
    nz_idx = np.where(nonzero_mask)[0]

    if len(nz_vals) == 0:
        return []
    elif len(nz_vals) == 1:
        return [int(nz_idx[0])]  # single value

    # Histogram to find high-frequency bins
    counts, edges = np.histogram(nz_vals, bins=bins)
    max_count = counts.max()
    max_bins = np.where(counts == max_count)[0]

    # Include all tied max-count bins in the bounds
    start = edges[max_bins[0]]
    end = edges[max_bins[-1] + 1]

    in_bin = (nz_vals >= start) & (nz_vals <= end)
    vals_in = nz_vals[in_bin]
    idx_in = nz_idx[in_bin]

    if len(vals_in) == 0:
        return []

    # Compute mean and std of values in high-frequency bin
    mean_val = np.mean(vals_in)
    std_val = np.std(vals_in)

    # Define lower and upper bounds (mean ± 1 std)
    lower_bound = mean_val - std_val
    upper_bound = mean_val + std_val

    # Find all indices corresponding to values within this range
    in_range_mask = (vals_in >= lower_bound) & (vals_in <= upper_bound)
    indices_in_range = idx_in[in_range_mask]

    return indices_in_range.tolist()

def most_frequent_row_index(values, bins=5):
    if isinstance(values,pl.LazyFrame):
        values = values.collect()

    if isinstance(values,pl.DataFrame):
        values = values.to_pandas()
    
    if isinstance(values,pd.DataFrame):
        if "date" in values.columns:
            values['date'] = pd.to_datetime(values['date'])
            values = values.dropna(subset=['date']).sort_values('date')
        else:
            raise Exception("Column 'date' is mandatory in values if passing a DataFrame instead of a list.")
        
        if "service_intensity" in values.columns:
            values = values.dropna(subset=["service_intensity"])
            values['service_intensity'] = values['service_intensity'].astype(float)
        else:
            raise Exception("Column 'service_intensity' is mandatory in values if passing a DataFrame instead of a list.")
        
        values = values.sort_values("date")
        if "file_id" in values.columns:
            id_col = "file_id"
        elif "gtfS_name" in values.columns:
            id_col = "gtfs_name"
        else:
            id_col = None
            values = list(values['service_intensity'])
        
    else:
        id_col = None

    if id_col is None:
        indices = most_frequent_row_range(values, bins=bins)
        if indices is None:
            return None 
        # Compute mean of the selected values
        mean_val = np.mean(values[indices])
        # Compute absolute differences between all values and the mean
        diffs = np.abs(values - mean_val)
        if len(diffs) == 0:
            return None
        # Find the minimum difference
        min_diff = np.min(diffs)
        # Find all indices where difference equals minimum
        nearest_indices = np.where(diffs == min_diff)[0]
        # Return first nearest index or None
        if nearest_indices.size > 0:
            return int(nearest_indices[0])
        else:
            return None
    else:
        indices = []

        # Loop over each unique ID
        for uid in values[id_col].unique():
            # Get the service_intensity column for this ID
            intensity_series = values.loc[values[id_col] == uid, "service_intensity"]

            # Get list of row indices in the most frequent range
            indices_i = most_frequent_row_range(intensity_series, bins=bins)
        
            # Map back to the DataFrame's actual index
            if indices_i is not None:  # only add if non-empty
                indices += list(intensity_series.iloc[indices_i].index)

        if len(indices)==0:
            return None  # no valid indices found

        # From the selected rows, find the index of the max service_intensity
        max_idx = values.loc[indices, "service_intensity"].idxmax()
        return max_idx

def assign_service_quality_to_interval(interval, route_type, service_matrix=SERVICE_MATRIX):
    l = service_matrix.loc[service_matrix['interval'] >= interval, route_type]
    if len(l) > 0:
        return l.iloc[0]
    else: 
        return None
    
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
