import polars as pl
import os
from pathlib import Path
from . import utils
import shutil


def stop_ids_in_aoi(df, aoi=None):
    aoi = aoi.to_crs(4326)
    minx, miny, maxx, maxy = aoi.total_bounds
    stop_ids = (
        df.filter(
            (pl.col("stop_lon") > minx)
            & (pl.col("stop_lon") < maxx)
            & (pl.col("stop_lat") > miny)
            & (pl.col("stop_lat") < maxy)
        )
        .select("stop_id")
        .unique()
        .collect()
    )

    return stop_ids["stop_id"].to_list()


def load_stack(filename, paths, stop_ids=[], rename_service_ids=None):
    id_cols = ["trip_id", "service_id", "route_id", "stop_id", "shape_id"]
    schema_dict = utils.get_df_schema_dict(filename)
    df = utils.read_csv_list(
        [p / filename for p in paths], schema_overrides=schema_dict
    )
    df = df.with_columns(pl.lit(None).alias("file_date"))
    for i in range(len(paths)):
        file_date_str = paths[i].name.split("_start_date_")[-1]
        df = df.with_columns(
            pl.when(pl.col("file_id") == i)
            .then(pl.lit(file_date_str))
            .otherwise(pl.col("file_date"))
            .alias("file_date")
        )

    columns = df.collect_schema().names()

    cols_in_id = []
    for col in id_cols:
        if col in columns:
            cols_in_id.append(col)
            df = df.rename({col: f"orig_{col}"})
            df = df.with_columns(
                pl.concat_str(
                    [pl.col(f"orig_{col}"), pl.lit("_date_"), pl.col("file_date")]
                ).alias(col)
            )

    if "service_id" in columns:
        if rename_service_ids is not None:
            df = df.join(rename_service_ids.unique(), on="service_id", how="left")
            df = df.drop("service_id").rename({"new_service_id": "service_id"})

    if (len(stop_ids) > 0) and ("stop_id" in columns):
        ids_df = pl.LazyFrame({"stop_id": stop_ids, "isin_aoi": True})
        df = df.join(ids_df, on="stop_id", how="left")
        df = df.with_columns(pl.col("isin_aoi").fill_null(False))
        df = df.with_columns(
            pl.col("isin_aoi").any().over(list(set(cols_in_id) - {"stop_id"}))
        )
        df = df.filter(pl.col("isin_aoi")).drop("isin_aoi")

    return df


def rank_column(df, column, sequence_column):
    new_column = f"new_{column}"
    rank_columns = list(
        set(df.collect_schema().names()) - {column, "file_id", "gtfs_name", "file_date"}
    )

    df = (
        df.group_by([column, "file_id", "gtfs_name", "file_date"])
        .agg(pl.all().sort_by(sequence_column))
        .with_columns((pl.struct(rank_columns).rank().cast(pl.Int32)).alias(new_column))
        .explode(pl.exclude([column, new_column, "file_id", "gtfs_name", "file_date"]))
        .with_columns(pl.min(column).over(new_column).alias(new_column))
    )

    return df


def reorganize_service_ids(trips):
    df = (
        trips.group_by("trip_id")
        .agg(pl.col("service_id").unique().sort())  # trips -> service_id list
        .group_by("service_id")
        .agg(pl.col("trip_id").unique())  # group back trips per service
    )

    # Get number of services
    n_services = df.select(pl.count()).collect().item()

    df = df.with_columns(
        (pl.lit("service_") + pl.arange(1, n_services + 1).cast(pl.Utf8)).alias(
            "new_service_id"
        )
    )
    rename_service_ids = df.select(["service_id", "new_service_id"]).explode(
        ["service_id"]
    )
    return rename_service_ids.collect().lazy()


def historic_stack(paths, new_path, aoi=None):
    paths = [Path(p) for p in paths]
    new_path = Path(new_path)
    if new_path.exists():
        shutil.rmtree(new_path)

    new_path.mkdir(parents=True, exist_ok=True)

    stops = load_stack("stops.txt", paths)
    if aoi is None:
        stop_ids = []
    else:
        stop_ids = stop_ids_in_aoi(stops, aoi)

    trips = load_stack("trips.txt", paths, stop_ids)
    rename_service_ids = reorganize_service_ids(trips)
    trips = trips.join(rename_service_ids.unique(), on="service_id", how="left")
    trips = trips.drop("service_id").rename({"new_service_id": "service_id"})

    stop_times = load_stack("stop_times.txt", paths, stop_ids, rename_service_ids)

    trip_data = trips.drop(["service_id", "gtfs_name", "file_date"])
    exclude = ["file_id", "trip_id"]
    trip_data = trip_data.rename(
        {
            col: col + "_trip_data"
            for col in trip_data.collect_schema().names()
            if col not in exclude
        }
    )
    trip_cols = list(set(trip_data.collect_schema().names()) - set(exclude))
    stop_times = stop_times.join(
        trip_data.unique(), on=["trip_id", "file_id"], how="left"
    )

    frequencies = None
    if os.path.isfile(paths[0] / "frequencies.txt"):
        frequencies = load_stack("frequencies.txt", paths, stop_ids, rename_service_ids)

    frequency_cols = []
    if frequencies is not None:
        frequency_data = frequencies.select(
            ["trip_id", "start_time", "end_time", "headway_secs", "file_id"]
        )
        frequency_data = (
            frequency_data.with_columns(
                pl.concat_str(
                    [pl.col("start_time"), pl.col("end_time"), pl.col("headway_secs")],
                    separator="|",
                ).alias("frequency_column")
            )
            .group_by(["trip_id", "file_id"])
            .agg(pl.col("frequency_column").str.join(","))
        )
        frequency_cols.append("frequency_column")
        stop_times = stop_times.join(
            frequency_data.lazy().unique(), on=["trip_id", "file_id"], how="left"
        )

    stop_times = rank_column(stop_times, "trip_id", "stop_sequence")
    stop_times = stop_times.drop(trip_cols)
    if frequencies is not None:
        stop_times = stop_times.drop(frequency_cols)

    shapes = None
    if os.path.isfile(paths[0] / "shapes.txt"):
        shapes = load_stack("shapes.txt", paths, stop_ids, rename_service_ids)
        shapes = rank_column(shapes, "shape_id", "shape_pt_sequence")

    to_drop = ["file_id", "gtfs_name", "file_date"]
    stops.drop(to_drop).collect().write_csv(new_path / "stops.txt")
    trips.drop(to_drop).collect().write_csv(new_path / "trips.txt")
    stop_times.drop(to_drop).collect().write_csv(new_path / "stop_times.txt")
    if frequencies is not None:
        frequencies.drop(to_drop).collect().write_csv(new_path / "frequencies.txt")

    if shapes is not None:
        shapes.drop(to_drop).collect().write_csv(new_path / "shapes.txt")

    already_loaded = [
        "stops.txt",
        "stop_times.txt",
        "trips.txt",
        "frequencies.txt",
        "shapes.txt",
    ]
    for filename in os.listdir(paths[0]):
        if filename not in already_loaded:
            try:
                file = load_stack(filename, paths, stop_ids, rename_service_ids)
                file.drop(to_drop).collect().write_csv(new_path / filename)
            except Exception as e:
                print(f"file {filename} error: {e}")


"""
def check_for_duplicated_id(df,col):
    with_file_id = df.unique(pl.exclude(["file_id","gtfs_name"])).select(col).filter(pl.col(col).is_duplicated()).unique()[col]
    if len(with_file_id) > 0:
        df = df.with_columns(
            pl.when(
                pl.col(col).is_in(with_file_id)
            ).then(
                pl.concat_str([pl.col(col),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col(col)
            ).alias(col)
        )
    else:
        with_file_id = []

    return df, with_file_id


def stops(paths,new_path,aoi=None):
    schema_dict = utils.get_df_schema_dict("stops.txt")
    df = utils.read_csv_list([p / "stops.txt" for p in paths],schema_overrides=schema_dict)
    df = df.collect()
    df, stop_ids_with_file_id = check_for_duplicated_id(df,"stop_id")
    df = df.drop(["file_id","gtfs_name"]).unique()
    df.write_csv(new_path / "stops.txt")
    if aoi is not None:
        aoi = aoi.to_crs(4326) 
        minx, miny, maxx, maxy = aoi.total_bounds
        stop_ids = df.filter(
            (pl.col("stop_lon") > minx)
            & (pl.col("stop_lon") < maxx)
            & (pl.col("stop_lat") > miny)
            & (pl.col("stop_lat") < maxy)
        ).select("stop_id").unique()

    return stop_ids_with_file_id, stop_ids


def routes(paths,new_path):
    schema_dict = utils.get_df_schema_dict("routes.txt")
    df = utils.read_csv_list([p / "routes.txt" for p in paths],schema_overrides=schema_dict)
    df = df.collect()
    df, route_ids_with_file_id = check_for_duplicated_id(df,"route_id")
    df = df.drop(["file_id","gtfs_name"]).unique()
    df.write_csv(new_path / "routes.txt")
    return route_ids_with_file_id


def trips_1(
        paths,
        shape_ids_with_file_id=[],
        rename_shapes=None,
        route_ids_with_file_id=[]
    ):
    schema_dict = utils.get_df_schema_dict("trips.txt")
    df = utils.read_csv_list([p / "trips.txt" for p in paths],schema_overrides=schema_dict)
    df = df.collect()
    df = df.with_columns(
        pl.concat_str([pl.col("service_id"),pl.lit("_"),pl.col("file_id")]).alias("service_id")
    )

    if rename_shapes is not None:
        rename_shapes = rename_shapes.filter(pl.col("shape_id").is_in(df['shape_id']))
        df = df.join(rename_shapes,on="shape_id",how="left")
        df = df.with_columns(
            pl.when(
                pl.col("new_shape_id").is_null()
            ).then(
                pl.col("shape_id")
            ).otherwise(
                pl.col("new_shape_id")
            ).alias("shape_id")
        ).drop("new_shape_id")

    if len(shape_ids_with_file_id) > 0: 
        df = df.with_columns(
            pl.when(
                pl.col("shape_id").is_in(shape_ids_with_file_id)
            ).then(
                pl.concat_str([pl.col("shape_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("shape_id")
            ).alias("shape_id")
        )

    if len(route_ids_with_file_id) > 0: 
        df = df.with_columns(
            pl.when(
                pl.col("route_id").is_in(route_ids_with_file_id)
            ).then(
                pl.concat_str([pl.col("route_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("route_id")
            ).alias("route_id")
        )

    df, trip_ids_with_file_id = check_for_duplicated_id(df,"trip_id")

    df = df.unique(pl.exclude(["file_id","gtfs_name"]))

    return df, trip_ids_with_file_id


def stop_times(paths,new_path,trip_data=None,trip_ids_with_file_id=[],stop_ids_with_file_id=[],stop_ids=[]):
    schema_dict = utils.get_df_schema_dict("stop_times.txt")
    df = utils.read_csv_list([p / "stop_times.txt" for p in paths],schema_overrides=schema_dict)
    if len(stop_ids) > 0:
        ids_df = pl.LazyFrame({"stop_id": stop_ids, "isin_aoi":True})
        df = df.join(ids_df, on="stop_id", how="left")
        df = df.with_columns(pl.col("isin_aoi").fill_null(False))
        df = df.with_columns(pl.col("isin_aoi").any().over("trip_id"))
        df = df.filter(pl.col("isin_aoi")).drop("isin_aoi")

    if len(stop_ids_with_file_id) > 0: 
        df = df.join(pl.LazyFrame({"stop_id":stop_ids_with_file_id,"with_file_id":True}).unique(),on="stop_id",how="left")
        df = df.with_columns(
            pl.when(
                pl.col("with_file_id")
            ).then(
                pl.concat_str([pl.col("stop_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("stop_id")
            ).alias("stop_id")
        ).drop("with_file_id")

    if len(trip_ids_with_file_id) > 0: 
        df = df.join(pl.LazyFrame({"trip_id":trip_ids_with_file_id,"with_file_id":True}).unique(),on="trip_id",how="left")
        df = df.with_columns(
            pl.when(
                pl.col("with_file_id")
            ).then(
                pl.concat_str([pl.col("trip_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("trip_id")
            ).alias("trip_id")
        ).drop("with_file_id")

    trip_cols = []
    if trip_data is not None:
        trip_data = trip_data.drop(["service_id","gtfs_name"])
        exclude = ["file_id", "trip_id"]
        trip_data = trip_data.rename({col: col + "_trip_data" for col in trip_data.columns if col not in exclude})
        trip_cols = list(set(trip_data.columns) - set(exclude))
        df = df.join(trip_data.lazy().unique(),on=["trip_id","file_id"],how="left")

    columns = list(set(df.collect_schema().names()) - {"trip_id","file_id","gtfs_name"})

    df = (
        df
        .group_by(["trip_id","file_id","gtfs_name"])
        .agg(pl.all().sort_by("stop_sequence"))
        .with_columns(
            pl.concat_str([
                pl.lit("trip_"),
                pl.struct(columns).rank().cast(pl.Int32)
            ]).alias("new_trip_id")
        )
        .with_columns(
            pl.when(
                pl.col("new_trip_id").is_duplicated() | pl.col("trip_id").is_duplicated()
            ).then(
                pl.col("new_trip_id")
            ).otherwise(
                pl.col("trip_id")
            ).alias("new_trip_id")
        )
    )

    if len(trip_cols) > 0: 
        df = df.select(pl.exclude(trip_cols))

    trip_ids = df.select("trip_id").unique().collect()["trip_id"]
    rename_trips = df.select(["trip_id","new_trip_id"]).unique().collect().filter(pl.col("trip_id")!=pl.col("new_trip_id"))

    if len(rename_trips) == 0:
        rename_trips = None 
    else:
        df = df.with_columns(
                pl.col("new_trip_id").alias("trip_id")
            )

    df = (
        df
        .drop("new_trip_id")
        .unique(pl.exclude("file_id","gtfs_name"))
    )

    df = df.collect()
    trip_ids_with_file_id = df.select("trip_id").filter(pl.col("trip_id").is_duplicated()).unique()['trip_id']
    if len(trip_ids_with_file_id) > 0:
        df = df.with_columns(
            pl.when(
                pl.col("trip_id").is_in(trip_ids_with_file_id)
            ).then(
                pl.concat_str([pl.col("trip_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("trip_id")
            ).alias("trip_id")
        )
    df = df.drop(["file_id","gtfs_name"]).unique()
    df = df.explode(pl.exclude(["trip_id","file_id","gtfs_name"])).sort("trip_id","stop_sequence")
    df.write_csv(new_path / "stop_times.txt")
    return trip_ids_with_file_id, rename_trips, trip_ids



def shapes(paths,new_path):
    schema_dict = utils.get_df_schema_dict("shapes.txt")
    df = utils.read_csv_list([p / "shapes.txt" for p in paths],schema_overrides=schema_dict)

    columns = list(set(df.collect_schema().names()) - {"shape_id","file_id","gtfs_name"})
    df = (
        df
        .group_by(["shape_id","file_id","gtfs_name"])
        .agg(pl.all().sort_by("shape_pt_sequence"))
        .with_columns(
            pl.concat_str([
                pl.lit("shape_"),
                pl.struct(columns).rank().cast(pl.Int32)
            ]).alias("new_shape_id")
        )
        .with_columns(
            pl.when(
                pl.col("new_shape_id").is_duplicated()
            ).then(
                pl.col("new_shape_id")
            ).otherwise(
                pl.col("shape_id")
            ).alias("new_shape_id")
        )
        .collect().lazy()
    )

    rename_shapes = df.select(["shape_id","new_shape_id"]).unique().collect().filter(pl.col("shape_id")!=pl.col("new_shape_id"))
    if len(rename_shapes) == 0:
        rename_shapes = None 
    else:
        df = df.with_columns(
                pl.col("new_shape_id").alias("shape_id")
            )

    df = (
        df
        .drop("new_shape_id")
        .unique(pl.exclude("file_id","gtfs_name"))
    )
    df = df.collect()
    shape_ids_with_file_id = df.select("shape_id").filter(pl.col("shape_id").is_duplicated()).unique()['shape_id']
    if len(shape_ids_with_file_id):
        df = df.with_columns(
            pl.when(
                pl.col("shape_id").is_in(shape_ids_with_file_id)
            ).then(
                pl.concat_str([pl.col("shape_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("shape_id")
            ).alias("shape_id")
        )

    df = df.drop(["file_id","gtfs_name"]).unique()
    df = df.explode(pl.exclude("shape_id","file_id","gtfs_name")).sort("shape_id","shape_pt_sequence")
    df.write_csv(new_path / "shapes.txt")
    return shape_ids_with_file_id, rename_shapes


def frequencies(paths,new_path,trip_ids_with_file_id=[],trip_ids=[],rename_trips=None):
    schema_dict = utils.get_df_schema_dict("frequencies.txt")
    df = utils.read_csv_list([p / "frequencies.txt" for p in paths],schema_overrides=schema_dict)
    df = df.collect()
    if len(trip_ids) > 0:
        df = df.filter(pl.col("trip_id").is_in(trip_ids))

    if rename_trips is not None:
        df = df.join(rename_trips,on="trip_id",how="left")
        df = df.with_columns(
            pl.when(
                pl.col("new_trip_id").is_null()
            ).then(
                pl.col("trip_id")
            ).otherwise(
                pl.col("new_trip_id")
            ).alias("trip_id")
        ).drop("new_trip_id")

    if len(trip_ids_with_file_id) > 0: 
        df = df.with_columns(
            pl.when(
                pl.col("trip_id").is_in(trip_ids_with_file_id)
            ).then(
                pl.concat_str([pl.col("trip_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("trip_id")
            ).alias("trip_id")
        )

    df = df.drop(["file_id","gtfs_name"]).unique()
    df.write_csv(new_path / "frequencies.txt")
    return None 


def rename_services_by_trips(trips):
    df = (
        trips
        .group_by("trip_id")
        .agg(pl.col("service_id").unique().sort())   # trips -> service_id list
        .group_by("service_id")
        .agg(pl.col("trip_id").unique())             # group back trips per service
    )

    # Get number of services
    n_services = df.height  

    df = df.with_columns(
        (
            pl.lit("service_") + pl.arange(1, n_services+1).cast(pl.Utf8)
        ).alias("new_service_id")
    )
    rename_service_ids = df.select(["service_id","new_service_id"]).explode(["service_id"])
    rename_service_ids_trips = df.select(["trip_id","new_service_id"]).explode(["trip_id"])
    return rename_service_ids, rename_service_ids_trips

def trips_2(
        df,
        new_path,
        trip_ids_with_file_id=[],
        trip_ids=[],
        rename_trips=None,
    ):
    if len(trip_ids) > 0:
        df = df.filter(pl.col("trip_id").is_in(trip_ids))

    if rename_trips is not None:
        df = df.join(rename_trips,on="trip_id",how="left")
        df = df.with_columns(
            pl.when(
                pl.col("new_trip_id").is_null()
            ).then(
                pl.col("trip_id")
            ).otherwise(
                pl.col("new_trip_id")
            ).alias("trip_id")
        ).drop("new_trip_id")

    if len(trip_ids_with_file_id) > 0: 
        df = df.with_columns(
            pl.when(
                pl.col("trip_id").is_in(trip_ids_with_file_id)
            ).then(
                pl.concat_str([pl.col("trip_id"),pl.lit("_"),pl.col("file_id")])
            ).otherwise(
                pl.col("trip_id")
            ).alias("trip_id")
        )

    df = df.drop(["file_id","gtfs_name"]).unique()
    if df.select("trip_id").is_duplicated().any():
        rename_services, rename_service_ids_trips = rename_services_by_trips(df) 
    else:
        rename_services = None
    
    if rename_services is not None:
        df = df.join(rename_service_ids_trips,on="trip_id",how="inner")
        df = df.with_columns(
            pl.col("new_service_id").alias("service_id")
        ).drop("new_service_id")
        df = df.unique()

    df.write_csv(new_path / "trips.txt")
    return rename_services

def calendar(paths,new_path,rename_services=None):
    schema_dict = utils.get_df_schema_dict("calendar.txt")
    df = utils.read_csv_list([p / "calendar.txt" for p in paths],schema_overrides=schema_dict)
    df = df.collect()
    df = df.with_columns(
        pl.concat_str([pl.col("service_id"),pl.lit("_"),pl.col("file_id")]).alias("service_id")
    ).drop(["file_id","gtfs_name"])
    if rename_services is not None:
        df = df.join(rename_services,on="service_id",how="inner")
        df = df.with_columns(
            pl.col("new_service_id").alias("service_id")
        ).drop("new_service_id")

    df = df.unique()
    df.write_csv(new_path / "calendar.txt")
    return None

def calendar_dates(paths,new_path,rename_services=None):
    schema_dict = utils.get_df_schema_dict("calendar_dates.txt")
    df = utils.read_csv_list([p / "calendar_dates.txt" for p in paths],schema_overrides=schema_dict)
    df = df.collect()
    df = df.with_columns(
        pl.concat_str([pl.col("service_id"),pl.lit("_"),pl.col("file_id")]).alias("service_id")
    ).drop(["file_id","gtfs_name"])
    if rename_services is not None:
        df = df.join(rename_services,on="service_id",how="inner")
        df = df.with_columns(
            pl.col("new_service_id").alias("service_id")
        ).drop("new_service_id")

    df = df.unique()
    df.write_csv(new_path / "calendar_dates.txt")
    return None


def historic_stack(paths,new_path,aoi=None):
    path = paths[0]
    filenames = [f for f in os.listdir(path)]
    paths = [Path(p) for p in paths]
    new_path = Path(new_path)
    new_path.mkdir(parents=True, exist_ok=True)

    stop_ids_with_file_id = []
    stop_ids = [] 
    if "stops.txt" in filenames:
        stop_ids_with_file_id, stop_ids = stops(paths,new_path,aoi)
        filenames.remove("stops.txt")

    route_ids_with_file_id = []
    if "routes.txt" in filenames:
        route_ids_with_file_id = routes(paths,new_path)
        filenames.remove("routes.txt")

    shape_ids_with_file_id = []
    rename_shapes = None
    if "shapes.txt" in filenames:
        shape_ids_with_file_id, rename_shapes = shapes(paths,new_path)
        filenames.remove("shapes.txt")

    trip_data = None
    trip_ids_with_file_id = []
    if "trips.txt" in filenames:
        trip_data, trip_ids_with_file_id = trips_1(
            paths,
            shape_ids_with_file_id=shape_ids_with_file_id,
            rename_shapes=rename_shapes,
            route_ids_with_file_id=route_ids_with_file_id
        )

    rename_trips = None
    trip_ids = []
    if "stop_times.txt" in filenames:
        trip_ids_with_file_id, rename_trips, trip_ids = stop_times(
            paths,
            new_path,
            trip_data,
            trip_ids_with_file_id,
            stop_ids_with_file_id,
            stop_ids
        )
        filenames.remove("stop_times.txt")

    if "frequencies.txt" in filenames:
        frequencies(paths,new_path,trip_ids_with_file_id,trip_ids,rename_trips)
        filenames.remove("frequencies.txt")

    rename_services = None
    if "trips.txt" in filenames:
        rename_services = trips_2(
            trip_data,
            new_path,
            trip_ids_with_file_id=trip_ids_with_file_id,
            trip_ids=trip_ids,
            rename_trips=rename_trips,
        )
        filenames.remove("trips.txt")

    if "calendar_dates.txt" in filenames: 
        calendar_dates(paths,new_path,rename_services)
        filenames.remove("calendar_dates.txt")
    
    if "calendar.txt" in filenames: 
        calendar(paths,new_path,rename_services)
        filenames.remove("calendar.txt")

    for filename in filenames:
        df = utils.read_csv_list([p / filename for p in paths]) 
        df = df.drop(["file_id","gtfs_name"]).unique()
        df.collect().write_csv(new_path / filename)
"""
