import matplotlib.pyplot as plt
import pandas as pd
from datetime import time, datetime
import matplotlib.patches as mpatches
import folium
import geopandas as gpd
from typing import Optional, List, Union
from matplotlib import colors, colormaps as mpl_colormaps
import matplotlib.colors as colors
from folium.plugins import BeautifyIcon
import numpy as np 


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


def general_map(
    m: Optional[folium.Map] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    pois: Optional[Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]] = [],
    gdfs: Optional[Union[gpd.GeoDataFrame, pd.DataFrame, List[Union[gpd.GeoDataFrame, pd.DataFrame]]]] = [],
    poi_column: Optional[str] = None,
    poi_color: Optional[str] = None,
    poi_cmap: Optional[str] = None,
    poi_vmin: Optional[float] = None,
    poi_vmax: Optional[float] = None,
    poi_opacity: float = 1.0,
    column: Optional[str] = None,
    color: str = "black",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    opacity: float = 0.4,
    size_column: Optional[str] = None,
) -> folium.Map:
    """
    General-purpose interactive Folium map builder.

    - Supports polygons, lines, points
    - AOI clipping
    - Multiple GeoDataFrames
    - Optional thematic coloring
    - Optional point-size scaling
    """

    # ------------------------------------------------------------------
    # CRS normalization
    # ------------------------------------------------------------------
    if aoi is not None:
        aoi = aoi.to_crs(4326)

    def _normalize_gdfs(objs):
        out = []
        for g in objs:
            if isinstance(g, gpd.GeoDataFrame):
                g = g.to_crs(4326)
            else:
                raise ValueError("Unsupported GeoDataFrame input")
            
            g = g[g.geometry.is_valid]
            out.append(g)
        return out

    if not isinstance(gdfs, list):
        gdfs = [gdfs]
    gdfs = _normalize_gdfs(gdfs)

    if not isinstance(pois, list):
        pois = [pois]
    pois = _normalize_gdfs(pois)

    if len(gdfs) == 0 and len(pois) == 0:
        raise ValueError("Nothing to map")
    
    # ------------------------------------------------------------------
    # Map centering
    # ------------------------------------------------------------------
    if aoi is not None:
        centroid = aoi.union_all().centroid
        gdfs = [g[g.intersects(aoi.union_all())] for g in gdfs]
        pois = [p[p.intersects(aoi.union_all())] for p in pois]
    elif pois:
        centroid = pd.concat([p.geometry for p in pois]).union_all().centroid
    else:
        centroid = pd.concat([g.geometry for g in gdfs]).union_all().centroid

    if m is None:
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=11,
            tiles="CartoDB positron",
        )


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def split_geoms(gdf):
        return (
            gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])],
            gdf[gdf.geometry.type.isin(["LineString", "MultiLineString", "LinearRing"])],
            gdf[gdf.geometry.type.isin(["Point", "MultiPoint"])],
        )

    def is_thematic(gdf, column, cmap):
        return column is not None and cmap is not None and column in gdf.columns

    def compute_radius(series: pd.Series, max_radius: float = 12.0):
        """
        Scale values from 0 → p90 into 0 → max_radius
        """
        p90 = series.quantile(0.9)
        clipped = series.clip(lower=0, upper=p90)
        return max_radius * clipped / p90 if p90 > 0 else max_radius

    # ------------------------------------------------------------------
    # vmin / vmax for gdfs
    # ------------------------------------------------------------------
    if poi_cmap is None:
        poi_cmap = cmap 

    if poi_color is None:
        poi_color = color 

    if poi_vmin is None:
        poi_vmin = vmin 
    
    if poi_vmax is None:
        poi_vmax = vmax 

    if column:
        values = [g[column].dropna() for g in gdfs if column in g.columns]
        if values:
            if vmin is None:
                vmin = min(v.min() for v in values)
            if vmax is None:
                vmax = max(v.max() for v in values)

    if poi_column is None:
        poi_column = column 

    if poi_column:
        values = [p[poi_column].dropna() for p in pois if poi_column in p.columns]
        if values:
            if poi_vmin is None:
                poi_vmin = min(v.min() for v in values)
            if poi_vmax is None:
                poi_vmax = max(v.max() for v in values)

    # ------------------------------------------------------------------
    # Draw gdfs
    # ------------------------------------------------------------------
    legend = True
    for g in gdfs:
        polys, lines, points = split_geoms(g)

        # Polygons
        if not polys.empty:
            if is_thematic(polys, column, cmap):
                m = polys.explore(
                    m=m,
                    column=column,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    legend=legend,
                    style_kwds={"color": None, "weight": 0, "fillOpacity": opacity},
                )
                legend = False
            else:
                m = polys.explore(
                    m=m,
                    color=color,
                    style_kwds={"fillColor": color, "fillOpacity": opacity, "weight": 0},
                )

        # Lines
        if not lines.empty:
            if is_thematic(lines, column, cmap):
                m = lines.explore(
                    m=m,
                    column=column,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    legend=legend,
                    style_kwds={"weight": 2},
                )
                legend = False
            else:
                m = lines.explore(m=m, color=color, style_kwds={"weight": 2})


        # Points with size scaling
        if not points.empty:
            if size_column is not None and size_column in points.columns:
                # Compute radii
                radii = compute_radius(points[size_column])
                points = points.assign(__radius=radii)
                
                # --- Prepare size legend ---
                # Choose 5 representative values from the size column
                size_values = np.linspace(points[size_column].min(), points[size_column].max(), 5)
                radius_values = compute_radius(pd.Series(size_values))
                
                # Add legend as a separate HTML overlay
                legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background:white; padding:10px; border-radius:5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">'
                legend_html += f'<b>{size_column}</b><br>'
                for val, r in zip(size_values, radius_values):
                    # Small circle with text
                    legend_html += f'<i style="background: black; border-radius:50%; width:{2*r}px; height:{2*r}px; display:inline-block; margin-right:5px;"></i>{val:.1f}<br>'
                legend_html += '</div>'
                m.get_root().html.add_child(folium.Element(legend_html))
                
            else:
                points = points.assign(__radius=4)  # default radius

            # Style function for dynamic radius and no border
            style_fn = lambda feature: {
                "radius": feature["properties"]["__radius"],
                "color": None,        # no border
                "weight": 0,          # border thickness (0 = none)
                "fillOpacity": 1.0,   # full fill
                "opacity": 1.0,       # stroke opacity (irrelevant here)
            }

            if is_thematic(points, column, cmap):
                # Thematic coloring with variable size
                m = points.explore(
                    m=m,
                    column=column,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    legend=legend,
                    marker_type="circle_marker",
                    style_kwds={"style_function": style_fn},  # dynamic radius
                )
                legend = False
            else:
                # Fixed color with variable size
                points = points.assign(__color=color)
                style_fn_fixed = lambda feature: {
                    "radius": feature["properties"]["__radius"],
                    "fillColor": feature["properties"]["__color"],
                    "color": None,
                    "weight": 0,
                    "fillOpacity": 1.0,
                    "opacity": 1.0,
                }
                m = points.explore(
                    m=m,
                    marker_type="circle_marker",
                    style_kwds={"style_function": style_fn_fixed},
                )



    # ------------------------------------------------------------------
    # POIs
    # ------------------------------------------------------------------
    if cmap is not None and poi_cmap != cmap:
        legend = True

    for p in pois:
        polys, lines, points = split_geoms(p)

        if not polys.empty:
            if is_thematic(polys, poi_column, poi_cmap):
                m = polys.explore(
                    m=m,
                    column=poi_column,
                    cmap=poi_cmap,
                    vmin=poi_vmin,
                    vmax=poi_vmax,
                    legend=legend,
                    style_kwds={"color": "black","fillOpacity": poi_opacity,"weight": 1,},
                )
                legend = False
            else:
                m = polys.explore(
                    m=m,
                    style_kwds={
                        "color": "black",
                        "fillColor": poi_color,
                        "fillOpacity": poi_opacity,
                        "weight": 1,
                    },
                )
        if not lines.empty:
            if is_thematic(lines, poi_column, poi_cmap):
                m = lines.explore(
                    m=m,
                    column=poi_column,
                    cmap=poi_cmap,
                    vmin=poi_vmin,
                    vmax=poi_vmax,
                    legend=legend,
                    style_kwds={"weight": 2},
                )
            else:
                m = lines.explore(
                    m=m,
                    color=poi_color,
                    style_kwds={"weight": 2},
                )

        if not points.empty:
            # Determine if theming is active
            thematic = is_thematic(points, poi_column, poi_cmap)

            # Handle categorical color mapping
            color_map = {}
            if thematic and points[poi_column].dtype == "object":
                categories = points[poi_column].unique()
                cmap = mpl_colormaps[poi_cmap]
                color_map = {
                    cat: colors.to_hex(cmap(i / len(categories))) for i, cat in enumerate(categories)
                }

            # Prepare colormap for numeric data if needed
            elif thematic:
                cmap = mpl_colormaps[poi_cmap]
                norm = colors.Normalize(vmin=poi_vmin, vmax=poi_vmax)
            # Compute colors safely
            def compute_color(row):
                if thematic:
                    if points[poi_column].dtype == "object":
                        color_val = color_map.get(row[poi_column], "#000000")
                    else:
                        color_val = colors.to_hex(cmap(norm(row[poi_column])))
                else:
                    color_val = poi_color
                return str(color_val) if color_val else "#392F2F"

            # Assign colors to _color column
            points["_color"] = points.apply(compute_color, axis=1).astype(str)

            # Convert to EPSG:4326 for Folium
            points_geojson = points.to_crs(4326).copy()

            # Keep only JSON-serializable columns + _color + geometry
            serializable_cols = [
                c for c in points_geojson.columns
                if c != points_geojson.geometry.name
                and points_geojson[c].apply(lambda x: isinstance(x, (str, int, float, type(None)))).all()
            ]
            for col in list(set(points_geojson.columns) - set(serializable_cols)):
                if col == points_geojson.geometry.name:
                    continue 
                
                try:
                    points_geojson[col] = points_geojson[col].astype(str)
                    serializable_cols.append(col)
                except:
                    None

            points_geojson = points_geojson[serializable_cols + [points_geojson.geometry.name]]

            # Add proper Markers using BeautifyIcon
            for _, row in points_geojson.iterrows():
                tooltip_text = "<br>".join(f"{c}: {row[c]}" for c in serializable_cols if c != "_color")
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    icon=BeautifyIcon(
                        icon="circle",
                        icon_shape="marker",
                        background_color=row["_color"],
                        border_color="black",
                        text_color="white",
                    ),
                    tooltip=tooltip_text,
                    legend=legend
                ).add_to(m)
                legend=False

    # ------------------------------------------------------------------
    # AOI outline & clipping
    # ------------------------------------------------------------------
    if aoi is not None:
        m = aoi.explore(
            m=m,
            color="blue",
            fill=False,
            style_kwds={"weight": 4, "dashArray": "5,5", "opacity": 1.0},
        )

    return m