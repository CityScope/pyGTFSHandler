from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from .. import utils
import os

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

    def __init__(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        route_ids: Optional[List[str] | pl.LazyFrame | pl.DataFrame] = None,
        route_types: Optional[List[int]] = None,
        check_files:bool=False
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

        self.lf = self.__read_routes(paths, route_ids, route_types, check_files=check_files)
        if (route_ids is not None) or (route_types is not None):
            self.route_ids = (
                self.lf.select("route_id").unique().collect()["route_id"].to_list()
            )
            if (len(self.route_ids) > 0) and (self.route_ids[0] is None):
                self.route_ids = []
        else:
            self.route_ids = None

    def __read_routes(
        self, paths, route_ids: Optional[List[str]], route_types: Optional[List[int]], check_files=False
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
            new_p = utils.search_file(p, file=file)
            if new_p is None:
                raise FileNotFoundError(f"File {file} does not exist in {p}")
            else:
                route_paths.append(new_p)


        schema_dict = utils.get_df_schema_dict(route_paths[0])
        routes = utils.read_csv_list(route_paths, schema_overrides=schema_dict, check_files=check_files)
        # Identify values that cannot be converted to int
        non_convertible = routes.filter(
            pl.col("route_type").cast(pl.Int64, strict=False).is_null()
        ).select("route_type").collect()["route_type"].unique().to_list()

        if non_convertible:
            routes = routes.with_columns(pl.col("route_type").alias("route_type_orig"))
            print(f"Warning: These route_type values could not be converted to int. Orig values in route_type_orig column. Non convertible values: {non_convertible}")

        # Cast column, replacing non-integer values with None
        routes = routes.with_columns(
            pl.col("route_type").cast(pl.Int64, strict=False).alias("route_type")
        )

        routes = utils.filter_by_id_column(routes, "route_id", route_ids)

        if route_types is not None:
            route_types_df = pl.LazyFrame({"route_type": route_types})
            routes = routes.join(route_types_df, on="route_type", how="semi")

        return routes
