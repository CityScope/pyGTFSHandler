from pathlib import Path
import polars as pl
from typing import Optional, List, Union
import utils


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
    ):
        """
        Initializes the Routes class by reading and filtering the routes data.

        Args:
            path (str | Path | list[str | Path]): One or more paths to directories containing `routes.txt` files.
            route_ids (list[str], optional): List of route IDs to filter by.
        """
        if isinstance(path, (str, Path)):
            self.paths = [Path(path)]
        else:
            self.paths = [Path(p) for p in path]

        self.lf = self.__read_routes(route_ids)
        if route_ids:
            self.route_ids = (
                self.lf.select("route_id").unique().collect()["route_id"].to_list()
            )
        else:
            self.route_ids = None

    def __read_routes(
        self,
        route_ids: Optional[List[str]],
    ) -> pl.LazyFrame:
        """
        Reads the routes data from one or more `routes.txt` files and applies optional filters.

        Args:
            route_ids (list[str], optional): List of route IDs to filter by.

        Returns:
            pl.LazyFrame: Filtered routes data as a LazyFrame.
        """
        route_paths = [
            p / "routes.txt" for p in self.paths if (p / "routes.txt").exists()
        ]
        if not route_paths:
            raise FileNotFoundError(
                "No routes.txt files found in the provided path(s)."
            )

        schema_dict = utils.get_df_schema_dict(route_paths[0])
        routes = utils.read_csv_list(route_paths, schema_overrides=schema_dict)

        if route_ids:
            route_ids_df = pl.DataFrame({"route_id": route_ids})
            routes = routes.join(route_ids_df.lazy(), on="route_id", how="inner")

        return routes
