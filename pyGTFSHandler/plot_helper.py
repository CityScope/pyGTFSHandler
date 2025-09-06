import matplotlib.pyplot as plt
import pandas as pd
from datetime import time, datetime


def plot_service_intensity(service_intensity):
    # Convert to pandas and parse dates
    if isinstance(service_intensity, pd.DataFrame):
        pdf = service_intensity.copy()
    else:
        pdf = service_intensity.to_pandas()

    pdf["date"] = pd.to_datetime(pdf["date"])

    plt.figure(figsize=(12, 6))
    bar_width = 0.8

    # Plot each date individually, coloring by type based on holiday/weekend flags
    for _, row in pdf.iterrows():
        row_date = row["date"]
        value = row["service_intensity"]

        if row["holiday"]:
            color = "green"
            # label = "Holiday"
        elif row["weekend"]:
            color = "red"
            # label = "Weekend"
        else:
            color = "blue"
            # label = "Weekday"

        plt.bar(row_date, value, width=bar_width, color=color, align="center")

    # Avoid duplicate legend entries
    handles = [
        plt.Line2D([], [], color="blue", label="Weekday", linewidth=10),
        plt.Line2D([], [], color="red", label="Weekend", linewidth=10),
        plt.Line2D([], [], color="green", label="Holiday", linewidth=10),
    ]
    plt.legend(handles=handles)

    plt.xlabel("Date")
    plt.ylabel("Service Intensity")
    plt.title("Service Intensity Over Time (Weekday, Weekend, Holiday)")
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def service_quality_map(stops_gdf, start_time, end_time):
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
        cmap="RdYlGn_r",
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
