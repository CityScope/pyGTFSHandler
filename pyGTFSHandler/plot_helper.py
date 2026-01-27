import matplotlib.pyplot as plt
import pandas as pd
from datetime import time, datetime
import matplotlib.patches as mpatches
import folium

def service_intensity(service_intensity):
    # Convert to pandas if it's Polars
    if not isinstance(service_intensity, pd.DataFrame):
        pdf = service_intensity.to_pandas()
    else:
        pdf = service_intensity.copy()

    pdf["date"] = pd.to_datetime(pdf["date"])

    plt.figure(figsize=(12, 6))
    bar_width = 0.8

    # Color and hatching setup
    weekend_hatch = "xx"
    holiday_hatch = "//"

    # --- CASE 1: file_id present ---
    if "file_id" in pdf.columns or "gtfs_name" in pdf.columns:
        name_col = "gtfs_name"
        if "gtfs_name" not in pdf.columns:
            name_col = "file_id"

        unique_files = sorted(pdf[name_col].unique())
        color_cycle = plt.cm.tab20.colors  # large qualitative palette
        color_map = {fid: color_cycle[i % len(color_cycle)] for i, fid in enumerate(unique_files)}

        grouped = pdf.groupby("date")
        for date, group in grouped:
            bottom = 0
            for _, row in group.iterrows():
                value = row["service_intensity"]
                fid = row[name_col]
                color = color_map[fid]

                # Determine hatch pattern (weekend or holiday)
                if row.get("holiday", False):
                    hatch = holiday_hatch
                elif row.get("weekend", False):
                    hatch = weekend_hatch
                else:
                    hatch = None

                plt.bar(
                    date,
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                )
                bottom += value

        # --- Legend setup ---
        file_handles = [
            plt.Line2D([], [], color=color_map[fid], label=f"File {fid}", linewidth=10)
            for fid in unique_files
        ]
        pattern_handles = [
            mpatches.Patch(facecolor="white", hatch=weekend_hatch, edgecolor="black", label="Weekend"),
            mpatches.Patch(facecolor="white", hatch=holiday_hatch, edgecolor="black", label="Holiday"),
        ]

        plt.legend(
            handles=file_handles + pattern_handles,
            title="Legend",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    # --- CASE 2: no file_id column ---
    else:
        base_color = "#a6c8ff"  # lighter blue
        for _, row in pdf.iterrows():
            row_date = row["date"]
            value = row["service_intensity"]

            if row.get("holiday", False):
                hatch = holiday_hatch
            elif row.get("weekend", False):
                hatch = weekend_hatch
            else:
                hatch = None

            plt.bar(
                row_date,
                value,
                width=bar_width,
                color=base_color,
                hatch=hatch,
                edgecolor="black",
            )

        pattern_handles = [
            plt.Line2D([], [], color=base_color, label="Weekday", linewidth=10),
            mpatches.Patch(facecolor=base_color, hatch=weekend_hatch, edgecolor="black", label="Weekend"),
            mpatches.Patch(facecolor=base_color, hatch=holiday_hatch, edgecolor="black", label="Holiday"),
        ]
        plt.legend(
            handles=pattern_handles,
            title="Legend",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    # --- Plot styling ---
    plt.xlabel("Date")
    plt.ylabel("Service Intensity")
    plt.title("Service Intensity Over Time")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()



def service_quality_map(stops_gdf, start_time, end_time, cmap="RdYlGn_r"):
    if isinstance(start_time, (datetime, time)):
        start_int = start_time.hour
    elif isinstance(start_time, str):
        start_int = int(start_time)
    else:
        start_int = int(start_time)

    if isinstance(end_time, (datetime, time)):
        end_int = end_time.hour
    elif isinstance(end_time, str):
        end_int = int(end_time)
    else:
        end_int = int(end_time)

    m = stops_gdf[
        [
            "stop_id",
            "parent_station",
            "stop_name",
            f"service_quality_{start_int}h_{end_int}h",
            f"interval_{start_int}h_{end_int}h",
            f"route_names_{start_int}h_{end_int}h",
            f"shape_directions_{start_int}h_{end_int}h",
            f"route_type_{start_int}h_{end_int}h",
            "route_type",
            "geometry",
        ]
    ].explore(
        column=f"service_quality_{start_int}h_{end_int}h",
        cmap=cmap,
        vmin=1,
        vmax=6,
        style_kwds={
            "color": "black",  # Border color
            "weight": 1,  # Border thickness
            "opacity": 1.0,  # Border opacity
            "fillOpacity": 1,
            "radius": 6,
        },
    )
    return m
