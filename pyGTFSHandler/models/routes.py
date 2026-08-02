# -*- coding: utf-8 -*-
"""GTFS routes.txt handling: loading, `route_id`/`route_type` filtering, and
`route_type` normalization.

Why this module exists and how it's organized:
-----------------------------------------------
`route_type` is read as a raw string (not cast straight to int at CSV-scan
time -- see `io.read_csv_lazy`'s `"route_type"` schema-marker handling)
because GTFS permits it to be either a standard numeric code (0-7), one of
several GTFS-extended/Transmodel numeric codes (e.g. 401 for "metro"), *or*
a small set of named strings ("bus", "tram", "cable car", ...). `_read_routes`
resolves whichever of these was actually given (via
`gtfs_checker.normalize_route_type` for the named-string case, then
`gtfs_checker.extended_to_standard_route_type` for the numeric-code case)
down to a standard 0-7 `route_type`, and raises if a *present* value can't
be resolved to either (as opposed to filling it in as "-1 unknown," which
would silently hide a real data error).
"""

from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from ..utils import geo_polars
from ..utils import gtfs_checker
from ..utils import io
from ..utils.colors import route_id_to_color, contrasting_text_color
import os
import warnings

"""
TODO: LLM prompt like this for route type 3
Look in the internet if the city has a Bus with High Level of Service and respond to this question: Does the city council or the public transit administrator consider the following row route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,route_url,route_color,route_text_color,route_sort_order,continuous_pickup,continuous_drop_off
1,c-6392dd86,1,Circular,1 - Circular,3,,ffdd00,000000,21,, of a gtfs routes.txt file called transportes_urbanos_de_vitoria_tuvisa from the city Vitoria-Gasteiz in Euskadi Spain a Bus with High Level of Service or a Bus Rapid Transit? True or False.
"""


class Routes:
    """
    A class to load and optionally filter GTFS routes data from one or more `routes.txt` files using Polars LazyFrame.

    Attributes:
        paths (List[Path]): List of directory paths containing `routes.txt` files.
        lf (pl.LazyFrame): A Polars LazyFrame containing the (optionally filtered) routes data.
    """
    def __init__(self,lf=None,route_ids=None) -> None:
        self.lf = lf 
        self.route_ids = route_ids

    def load(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        route_ids: Optional[List[str] | pl.LazyFrame | pl.DataFrame] = None,
        route_types: Optional[List[int]] = None,
        check_files:bool=False,
        min_file_id=0
    ):
        """
        Initializes the Routes class by reading and filtering the routes data.

        Args:
            path (str | Path | list[str | Path]): One or more paths to directories containing `routes.txt` files.
            route_ids (list[str], optional): List of route IDs to filter by.
        """
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        self.lf = self._read_routes(paths, route_ids, route_types, check_files=check_files, min_file_id=min_file_id)
        if self.lf is not None:
            if (route_ids is not None) or (route_types is not None):
                self.route_ids = (
                    self.lf.select("route_id").unique().collect()["route_id"].to_list()
                )
                if (len(self.route_ids) > 0) and (self.route_ids[0] is None):
                    self.route_ids = []
            else:
                self.route_ids = None
        else:
            self.route_ids = None
            
    def _read_routes(
        self, paths, route_ids: Optional[List[str]], route_types: Optional[List[int]], check_files=False, min_file_id=0
    ) -> pl.LazyFrame:
        """
        Reads the routes data from one or more `routes.txt` files and applies optional filters.

        Args:
            route_ids (list[str], optional): List of route IDs to filter by.

        Returns:
            pl.LazyFrame: Filtered routes data as a LazyFrame.
        """
        route_paths: List[Path] = []
        file = "routes.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                route_paths.append(None)
                warnings.warn(f"File {file} does not exist in {p}", UserWarning)
            else:
                route_paths.append(new_p)


        schema_dict, _ = gtfs_checker.get_df_schema_dict("routes.txt")
        routes = io.read_csv_list(route_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id)
        if (routes is None) or (routes.select(pl.len()).collect().item() == 0):
            return None

        # Identify values that cannot be converted to int directly (e.g. the
        # named route_type strings GTFS also accepts, like "bus"/"tram").
        non_convertible = routes.filter(
            pl.col("route_type").cast(pl.Int64, strict=False).is_null()
            & pl.col("route_type").is_not_null()
        ).select("route_type").collect()["route_type"].unique().to_list()

        if non_convertible:
            routes = routes.with_columns(pl.col("route_type").alias("route_type_orig"))
            # `gtfs_checker.normalize_route_type` understands the named
            # strings the GTFS spec also permits for `route_type`
            # ("bus", "tram", "cable car", ...); resolve those to their
            # numeric code here, before the extended-code mapping below.
            resolved_names = {
                value: gtfs_checker._try_normalize_route_type_name(value)
                for value in non_convertible
            }
            still_unresolved = [v for v, resolved in resolved_names.items() if resolved is None]
            if still_unresolved:
                warnings.warn(
                    f"These route_type values could not be converted to int or matched to a "
                    f"known route_type name. Original values kept in route_type_orig column. "
                    f"Unrecognized values: {still_unresolved}"
                )
            routes = routes.with_columns(
                pl.col("route_type").replace(resolved_names).alias("route_type")
            )

        # Cast column, replacing non-integer values with None
        routes = (
            routes
            .with_columns(
                pl.col("route_type")
                    .cast(pl.Int64, strict=False)
                    .alias("extended_route_type"),
            )
            .with_columns(
                pl.col("extended_route_type")
                    .map_elements(gtfs_checker.extended_to_standard_route_type,pl.Int64)
                    .alias("route_type"),
            )
        )

        # Any row whose route_type was present but resolves to neither a
        # standard (0-7) nor a known GTFS-extended code is a genuine data
        # error, not a merely-missing value -- flag it loudly rather than
        # silently filing it under "-1 unknown".
        unmappable = (
            routes.filter(
                pl.col("route_type").is_null()
                & pl.col("extended_route_type").is_not_null()
            )
            .select("extended_route_type")
            .unique()
            .collect()["extended_route_type"]
            .to_list()
        )
        if unmappable:
            raise Exception(
                f"routes.txt has route_type value(s) that are neither a standard GTFS "
                f"route_type (0-7) nor a recognized GTFS-extended route_type code: {unmappable}"
            )

        routes = (
            routes
            .with_columns(
                pl.col("route_type").fill_null(-1),
                pl.col("extended_route_type").fill_null(-1)
            )
            .with_columns(
                pl.col("route_type")
                    .map_elements(gtfs_checker.route_type_to_str,return_dtype=pl.String)
                    .alias("route_type_text")
            )
        )
        routes = self._fill_route_colors(routes)

        routes = geo_polars.filter_by_id_column(routes, "route_id", route_ids)

        if route_types is not None:
            route_types_df = pl.LazyFrame({"route_type": route_types})
            routes = routes.join(route_types_df, on="route_type", how="semi")

        return routes

    def _fill_route_colors(self, routes: pl.LazyFrame) -> pl.LazyFrame:
        """Ensures every route ends up with a non-null `route_color`/
        `route_text_color` (hex strings, no leading `#`).

        Routes that already give a non-empty `route_color`/`route_text_color`
        in `routes.txt` keep it as-is. Routes missing `route_color` get a
        deterministic-but-varied color hashed from their `route_id` (see
        `utils.colors.route_id_to_color`) rather than one hardcoded fallback
        color for every missing route. `route_text_color`, when missing, is
        derived from the *actual* `route_color` (given or generated) via
        WCAG-style relative luminance so text stays readable against it.
        """
        schema_names = routes.collect_schema().names()
        if "route_color" not in schema_names:
            routes = routes.with_columns(pl.lit(None, dtype=pl.String).alias("route_color"))
        if "route_text_color" not in schema_names:
            routes = routes.with_columns(pl.lit(None, dtype=pl.String).alias("route_text_color"))

        routes = routes.with_columns(
            pl.col("route_color").str.strip_chars().str.strip_chars("#"),
            pl.col("route_text_color").str.strip_chars().str.strip_chars("#"),
        ).with_columns(
            pl.when(pl.col("route_color").eq("")).then(None).otherwise(pl.col("route_color")).alias("route_color"),
            pl.when(pl.col("route_text_color").eq(""))
            .then(None)
            .otherwise(pl.col("route_text_color"))
            .alias("route_text_color"),
        )

        routes = routes.with_columns(
            pl.when(pl.col("route_color").is_null())
            .then(pl.col("route_id").map_elements(route_id_to_color, return_dtype=pl.String))
            .otherwise(pl.col("route_color"))
            .alias("route_color")
        )

        routes = routes.with_columns(
            pl.when(pl.col("route_text_color").is_null())
            .then(pl.col("route_color").map_elements(contrasting_text_color, return_dtype=pl.String))
            .otherwise(pl.col("route_text_color"))
            .alias("route_text_color")
        )

        return routes
