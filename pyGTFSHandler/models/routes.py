from pathlib import Path
import polars as pl
from typing import Optional, List, Union
from utils import read_csv_list, get_df_schema_dict

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
        route_ids: Optional[List[str]] = None,
        route_types: Optional[List[int]] = None,
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

        self.lf = self.__read_routes(paths, route_ids, route_types)
        if route_ids or route_types:
            self.route_ids = (
                self.lf.select("route_id").unique().collect()["route_id"].to_list()
            )
        else:
            self.route_ids = None

    def __read_routes(
        self, paths, route_ids: Optional[List[str]], route_types: Optional[List[int]]
    ) -> pl.LazyFrame:
        """
        Reads the routes data from one or more `routes.txt` files and applies optional filters.

        Args:
            route_ids (list[str], optional): List of route IDs to filter by.

        Returns:
            pl.LazyFrame: Filtered routes data as a LazyFrame.
        """
        route_paths = [p / "routes.txt" for p in paths if (p / "routes.txt").exists()]
        if not route_paths:
            raise FileNotFoundError(
                "No routes.txt files found in the provided path(s)."
            )

        schema_dict = get_df_schema_dict(route_paths[0])
        routes = read_csv_list(route_paths, schema_overrides=schema_dict)

        if route_ids:
            route_ids_df = pl.DataFrame({"route_id": route_ids})
            routes = routes.join(route_ids_df.lazy(), on="route_id", how="semi")

        if route_types:
            route_types_df = pl.DataFrame({"route_type": route_types})
            routes = routes.join(route_types_df.lazy(), on="route_type", how="semi")

        return routes
