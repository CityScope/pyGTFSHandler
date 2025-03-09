import os
import polars as pl
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import shapely
from datetime import datetime
import unicodedata
import re 
gpd.options.io_engine = "pyogrio"
"TODO check if route_group_id are in the right order"
"TODO get_frequencies what if two stop ids are in the same stop group and have the same line?"
"TODO If shapes.txt does not exist take it from stops_gdf"
"TODO Do not process stops outside bounds (do not group etc)"
"TODO Do not load stop_gdf before grouping stops to create less geometries. This would mean implementing conversion to utm directly from x and y columns."
"TODO search in calendar_dates the day with most or least services"
"TODO stop times should include trips from the day before and after as negative and >24 numbers for time intervals that pass 24:00:00"
"TODO get calendar n trips by a date range. Show differences in the week or months. This would make selecting max or min days better."
"TODO rethink route filter"
"TODO the number of trips_ids is not a good value to meassure the amount of services as one line cut in two would be double. Maybe do number of rows in stop_times or trips * shape distance."
"TODO first and last stops in squedule pattern to detect loop lines. Is it a good idea? Example 655 y 653 Majadahonda? Solution: Only first and last stop should be common but not the rest"
"TODO shapes in trips"
"TODO set time bounds before counting service ids (It is slower) or at least do it if not service_id counts are needed"
"TODO stops with parent_station and group stops merging geometries"
"TODO stop_times deletes repeating stops if the same rep_trip id stops two times in the stop group"
"TODO if one trip shares the exact stop time for more than x stops with another put them in the same trip group"
"TODO group_stops might not be doing the mean in the location" 
"""
If a bus starts it's journey at 23:00 and finishes at 2:00 the following day it could appear as 26:00. Calculating frequencies etc this is counted as a bus from 
00:00:00 to 02:00:00 and a bus from 23:59:59 to 23:59:59 both the same day which is wrong. 
Possible corrections. Take busses from the following day and add 24 hours to the times and from the previous day the ones 
with times > 24 and subtract 24. Check that departure times secs and stop sequence are always the same. Detect cases
of timetables like this 23:40 -> 00:10 and change it 23:40 -> 24:10. 
"""
"""
Some frequencies files have the headway in minutes instead of seconds. When reading if the min headway is less than 30 and the max less than
480 it is considered to be in minutes
"""

def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Remove tildes (accent marks) by decomposing and stripping accents
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    return s

def clean_string(string):
    # Replace special characters and multiple spaces/underscores with a single underscore
    string = normalize_string(string)
    string = re.sub(r'[^a-zA-Z0-9]', '_', string)  # Replace non-alphanumeric characters with underscores
    string = re.sub(r'_+', '_', string)    # Replace multiple underscores with a single one
    string = string.strip('_')             # Remove leading/trailing underscores
    
    return string

def latlon_to_utm(lat_col, lon_col,epsg):
    # Create a Transformer for EPSG:4326 to UTM projection
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)

    # Perform the transformation
    utm_x, utm_y = transformer.transform(lon_col, lat_col)
    return utm_x, utm_y

class GTFS(object):
    """
    Main class to store GTFS data and provde helper
    functions.
    """

    def __init__(
        self,
        gtfs_dir:str|list[str],
        service_date:str, 
        start_time:str='00:00:00',
        end_time:str='00:00:00', 
        bounds:gpd.GeoSeries|gpd.GeoDataFrame=None,
        strict_bounds:bool=True,
        stop_group_distance=0,
        trip_group_distance=0,
        trip_group_overlap=0.75,
        correct_stop_sequence=True,crs=4326,
        route_filter:pd.DataFrame|pl.DataFrame=None,
        all_stops:bool=True
    ):
        """
        Instantiate class with directory of GTFS Files and a service_date for
        which to get service.
        """
        if type(gtfs_dir) == str:
            gtfs_dir = [gtfs_dir]

        if type(route_filter) == pd.DataFrame:
            route_filter = pl.from_pandas(route_filter)

        self.gtfs_dir = gtfs_dir
        if service_date == 'today':
            service_date = datetime.now().strftime("%Y%m%d")

        if '-' in service_date:
            try:
                # Convert the string to a datetime object
                service_date = datetime.strptime(service_date, "%d-%m-%Y")

                # Convert the datetime object to the desired string format
                service_date = service_date.strftime("%Y%m%d")
            except:
                None            

        self.service_date = service_date
        #self.int_service_date = int(service_date)
        self.bounds = bounds
        self.strict_bounds = strict_bounds
        self.crs = crs
        self.start_time = start_time
        self.end_time = end_time
        self.stop_group_distance = stop_group_distance 
        self.trip_group_distance = trip_group_distance
        self.trip_group_overlap = trip_group_overlap
        self.correct_stop_sequence = correct_stop_sequence
        self.route_filter = route_filter
        self.all_stops = all_stops

        self.__load_gtfs(state=0)

    def __load_gtfs(self,state=0):
        if type(self.bounds) != type(None):
            self.stops_pl, self.stops_gdf, self.trips, self.stop_times, self.stop_ids_in_bounds = self.__set_bounds(
                    bounds=self.bounds,strict_bounds=self.strict_bounds,service_ids=[],route_filter=self.route_filter
            )
            if len(self.stop_ids_in_bounds) == 0:
                raise Exception("No stops inside bounds")
        else:
            self.stop_ids_in_bounds = []
        
        self.trips = self.__read_trips(route_filter=self.route_filter)
        self.stop_times = self.__read_stop_times(trips=self.trips)

        self.stops_pl = self.__read_stops_pl()
        self.stops_gdf = self.__read_stops_gdf(stops_pl=self.stops_pl,stop_id_list=[])
        self.first_stops_gdf = self.stops_gdf.copy()

        #  !!!!!!!!!!!!!!!!
        self.stop_id_list = self.stop_times['stop_id'].unique().to_list()
        
        self.stops_gdf = self.stops_gdf.loc[self.stops_gdf['stop_id'].isin(self.stop_id_list)].reset_index(drop=True)
        self.stops_pl = self.stops_pl.filter(self.stops_pl['stop_id'].is_in(self.stop_id_list))

        self.stop_groups_gdf, self.stops_gdf, self.stop_times = self.__group_stops(self.stops_gdf,self.stop_times,self.stop_group_distance)

        #self.stop_times = self.stop_times.group_by(['stop_group_id','departure_time']).agg(
        #    pl.all(),
        #    temp=pl.col('file_number').min()
        #).explode(pl.exclude(['stop_group_id','departure_time','temp'])).filter(
        #    pl.col('temp')==pl.col('file_number')
        #).drop('temp')

        self.trips = self.trips.filter(pl.col('trip_id').is_in(self.stop_times['trip_id']))
        
        self.frequencies = self.__read_frequencies(trips=self.trips)
        self.service_id_counts = self.__get_service_id_counts(stop_times=self.stop_times,trips=self.trips,frequencies=self.frequencies,stop_ids_in_bounds=self.stop_ids_in_bounds)
        
        self.calendar = self.__read_calendar(service_id_counts=self.service_id_counts)
        self.calendar_dates = self.__read_calendar_dates(service_id_counts=self.service_id_counts)

        self.counts_by_date = self.__get_counts_by_date(self.calendar,self.calendar_dates)            
        self.service_ids,self.service_date = self.__get_service_ids(calendar=self.calendar,calendar_dates=self.calendar_dates,
                                                                    service_date=self.service_date,counts_by_date=self.counts_by_date)
        
        self.trips = self.trips.filter(self.trips['service_id'].is_in(self.service_ids))  
        self.trips = self.trips.unique('trip_id').sort('trip_id')
        self.stop_times = self.stop_times.filter(pl.col('trip_id').is_in(self.trips['trip_id']))

        if self.correct_stop_sequence:
            self.stop_times, self.trips = self.__correct_stop_sequence(stop_times=self.stop_times,trips=self.trips)
            self.stop_times = self.__sort_sequence_col(stop_times=self.stop_times) 

        self.stop_times, self.trips = self.__set_time_bounds(stop_times=self.stop_times,trips=self.trips,frequencies=self.frequencies,
                                                            start_time=self.start_time,end_time=self.end_time) 

        self.stop_id_list = list(self.stop_times['stop_id'].unique())

        #self.stops_pl = self.__read_stops_pl()
        #self.stops_pl = self.stops_pl.filter(pl.col('stop_id').is_in(self.stop_id_list))

        #self.stops_gdf = self.__read_stops_gdf(stops_pl=self.stops_pl,stop_id_list=self.stop_id_list)
        #self.stops_gdf = self.stops_gdf.loc[self.stops_gdf['stop_id'].isin(self.stop_id_list)].reset_index(drop=True)
        if self.all_stops == False:
            self.stops_pl = self.stops_pl.filter(pl.col('stop_id').is_in(self.stop_id_list))
            self.stops_gdf = self.stops_gdf.loc[self.stops_gdf['stop_id'].isin(self.stop_id_list)].reset_index(drop=True)
            
        self.routes = self.__read_routes(trips=self.trips)

        if 'stop_group_id' not in self.stop_times.collect_schema().names():
            self.stop_groups_gdf, self.stops_gdf, self.stop_times = self.__group_stops(self.stops_gdf,self.stop_times,self.stop_group_distance)

            self.stop_times = self.stop_times.group_by(['stop_group_id','departure_time']).agg(
                pl.all(),
                temp=pl.col('file_number').min()
            ).explode(pl.exclude(['stop_group_id','departure_time','temp'])).filter(
                pl.col('temp')==pl.col('file_number')
            ).drop('temp')

        self.stop_times, self.trips, self.routes = self.__get_schedule_pattern(self.stop_times,self.trips,self.routes) ## REP TRIP ID

        self.stop_times = self.stop_times.sort('trip_id','departure_time').group_by('stop_group_id','rep_trip_id','stop_sequence').agg(
            pl.all()
        ).group_by('stop_group_id','rep_trip_id').agg(
            pl.all().last()
        ).explode(pl.exclude('stop_group_id','rep_trip_id','stop_sequence'))

        if 'shape_id' in self.trips.collect_schema().names():
            self.trips = self.trips.with_columns(
                shape_id = pl.when(pl.col('shape_id').is_null()).then(pl.col('rep_trip_id')).otherwise(pl.col('shape_id'))
            )
        else:
            self.trips = self.trips.with_columns(
                shape_id = pl.col('rep_trip_id')
            )

        self.shapes = self.__read_shapes(trips=self.trips,stop_times=self.stop_times,stops_pl=self.stops_pl)
 
        self.stop_times, self.trips, self.routes = self.__group_trips(self.stops_gdf,self.stop_times,self.trips,self.routes,
                                                                      self.trip_group_distance,self.trip_group_overlap)
        self.trip_group_id_list = list(self.trips['trip_group_id'].unique())

        return None
    
    def __read_csv(self,path,schema_overrides:dict=None,columns:list=None):
        df = pl.read_csv(
            path,
            columns=columns,
            infer_schema=None
        )
        rename_dict = {name:clean_string(normalize_string(name)) for name in df.columns}
        df = df.rename(rename_dict)
        df = df.with_columns(
            pl.all().str.strip_chars()
            .str.replace_all(r"[áàãâä]", "a")  # Replaces accented 'a' characters
            .str.replace_all(r"[éèêë]", "e")    # Replaces accented 'e' characters
            .str.replace_all(r"[íìîï]", "i")    # Replaces accented 'i' characters
            .str.replace_all(r"[óòõôö]", "o")   # Replaces accented 'o' characters
            .str.replace_all(r"[úùûü]", "u")    # Replaces accented 'u' characters
            .str.replace_all(r"[ñ]", "n")       # Replaces ñ with n
            .str.replace_all(r"['`’]", "")      # Removes quotes and apostrophes
        )
        if type(schema_overrides) == dict:
            keys = []
            for k in schema_overrides.keys():
                if k in df.columns:
                    keys.append(k)
            
            if len(k) > 0:
                df = df.with_columns(
                    pl.col(k).cast(schema_overrides[k]) for k in keys
                )

        return df
    
    def __read_stops_pl(self,stop_id_list=[],reload_data:bool=False):
        if reload_data == True:
            stops = None
        else:
            try:
                stops = self.stops_pl 
            except:
                stops = None 
        
        if type(stops) == type(None):
            stops = pl.concat([
                            self.__read_csv(
                                os.path.join(self.gtfs_dir[j], 'stops.txt'),
                                schema_overrides={'stop_id':str,'stop_lon':float,'stop_lat':float},
                                #columns=['stop_id','stop_lon','stop_lat']
                            ).with_columns(file_number=j) for j in range(len(self.gtfs_dir))
                        ], how='diagonal_relaxed')
            stops = stops.filter(~pl.col('stop_id').is_null())
            stops = stops.with_columns(
                orig_stop_id = pl.col('stop_id'),
                stop_id = pl.col('stop_id').cast(str) + "_" + pl.col('file_number').cast(str)
            )
            stops = stops.unique('stop_id')
            if len(stop_id_list) > 0:
                stops = stops.filter(stops['stop_id'].is_in(stop_id_list))

        return stops.unique('stop_id')
    
    def __read_stops_gdf(self,stops_pl,stop_id_list=[],reload_data:bool=False):
        if reload_data == True:
            stops_gdf = None
        else:
            try:
                stops_gdf = self.stops_gdf 
            except:
                stops_gdf = None 
        
        if type(stops_gdf) == type(None):
            stops_gdf = gpd.GeoDataFrame(stops_pl.to_pandas(), geometry=gpd.points_from_xy( ######### geopolars
                stops_pl['stop_lon'], stops_pl['stop_lat'], crs=self.crs))
            stops_gdf = stops_gdf.set_crs(epsg=self.crs) 
            """
            if 'parent_station' in stops_gdf.columns: 
                grouped = stops_gdf.groupby('parent_station')['geometry'].apply(lambda x: shapely.unary_union(x))
                # Merge the grouped geometries back into the original GeoDataFrame by 'parent_station'
                stops_gdf = stops_gdf.merge(grouped, on='parent_station',how='left')#.explore()
                stops_gdf['geometry'] = stops_gdf['geometry_x']
                stops_gdf.loc[stops_gdf['geometry_y'].isna()==False,'geometry'] = stops_gdf.loc[stops_gdf['geometry_y'].isna()==False,'geometry_y']
                stops_gdf = stops_gdf.drop(columns=['geometry_x','geometry_y']).set_geometry('geometry') 
            """                
            if len(stop_id_list) > 0:
                stops_gdf = stops_gdf.loc[stops_gdf['stop_id'].isin(stop_id_list)].reset_index(drop=True)

        return stops_gdf
    
    def __read_frequencies(self,trips=None,reload_data:bool=False):
        if reload_data == True:
            frequencies = None
        else:
            try:
                frequencies = self.frequencies 
            except:
                frequencies = None 

        if type(frequencies) == type(None):
            files = []
            for j in range(len(self.gtfs_dir)):
                if os.path.exists(os.path.join(self.gtfs_dir[j], 'frequencies.txt')):
                    with open(os.path.join(self.gtfs_dir[j], 'frequencies.txt'), 'r') as file:
                        lines = file.readlines()
            
                    if len(lines) > 1:
                        files.append(j)

            if len(files) > 0:
                frequencies = pl.concat([self.__read_csv(os.path.join(self.gtfs_dir[j], 'frequencies.txt'), 
                                        schema_overrides={'trip_id': str,'headway_secs':float}
                                        ).with_columns(file_number=j) for j in files
                ], how='diagonal_relaxed')
                frequencies = frequencies.filter(~pl.all_horizontal(pl.exclude('file_number').is_null()))
                frequencies = frequencies.with_columns(
                    orig_trip_id = pl.col('trip_id'),
                    trip_id = pl.col('trip_id').cast(str) + "_" + pl.col('file_number').cast(str)
                )
                if type(frequencies) != type(None):
                    frequencies = frequencies.filter(pl.col('trip_id').is_in(trips['trip_id']))

                    max_value = frequencies["headway_secs"].max()
                    min_value = frequencies["headway_secs"].min()
                    if max_value < 480 and min_value < 30:
                        frequencies = frequencies.with_columns(headway_secs = (frequencies["headway_secs"] * 60))

            else:
                frequencies = pl.DataFrame(schema=['trip_id','orig_trip_id','start_time','end_time','headway_secs','file_number'])
                            
        return frequencies.unique(['trip_id','start_time','end_time'])
    
    def __read_trips(self,service_ids=[],route_filter=None,reload_data:bool=False):
        """
        route filter is a dataframe with columns 'column' 'function' 'value'
        """
        if reload_data == True:
            trips = None
        else:
            try:
                trips = self.trips 
            except:
                trips = None

        if type(trips) == type(None): 
            trips = pl.concat([
                self.__read_csv(os.path.join(self.gtfs_dir[j],'trips.txt'),
                    schema_overrides={'route_id': str,'service_id': str,'trip_id': str,'shape_id': str,'direction_id':int}
                ).with_columns(file_number = j) for j in range(len(self.gtfs_dir))
            ], how='diagonal_relaxed')
            trips = trips.filter(~pl.col('trip_id').is_null())

            trips = trips.with_columns(
                orig_trip_id = pl.col('trip_id'),
                trip_id = pl.col('trip_id').cast(str) + "_" + pl.col('file_number').cast(str),
                orig_route_id = pl.col('route_id'),
                route_id = pl.col('route_id').cast(str) + "_" + pl.col('file_number').cast(str),
                orig_service_id = pl.col('service_id'),
                service_id = pl.col('service_id').cast(str) + "_" + pl.col('file_number').cast(str),
            )
            if 'shape_id' in trips.collect_schema().names():
                trips = trips.with_columns(
                    orig_shape_id = pl.col('shape_id'), 
                    shape_id = pl.col('shape_id').cast(str) + "_" + pl.col('file_number').cast(str),
                )

            if type(route_filter) != type(None):
                routes = self.__read_routes(trips=None)
                _filter_in = None
                _filter_out = None
                route_filter = route_filter.with_columns(
                    function = pl.col('function').cast(str)
                )
                route_filter = route_filter.with_columns(
                    function = pl.when((pl.col('function') == pl.lit('in')) | (pl.col('function') == pl.lit('contains'))).then(1
                                ).when((pl.col('function') == pl.lit('not in')) | (pl.col('function') == pl.lit('!contains')
                                                                                ) | (
                                                                        pl.col('function') == pl.lit('! contains')) | (pl.col('function') == pl.lit('not contains'))).then(3
                                ).when((pl.col('function') == pl.lit('isin')) | (pl.col('function') == pl.lit('is_in'))).then(2
                                ).when((pl.col('function') == pl.lit('not isin')) | (pl.col('function') == pl.lit('! is_in')
                                                                                ) | (
                                                                        pl.col('function') == pl.lit('not contains')) | (pl.col('function') == pl.lit('not is_in'))).then(4)
                ).sort('function')
                _filter_in = (routes['route_id'] != routes['route_id']).cast(int)
                _filter_out = (routes['route_id'] == routes['route_id']).cast(int)
                for f in route_filter.rows(named=True):
                    if f['column'] in routes.collect_schema().names():
                        if f['function'] == 1:
                            _filter_in += routes[f['column']].cast(str).str.contains(f['value']).cast(int).fill_null(0)
                        elif f['function'] == 2:
                            _filter_in += routes[f['column']].is_in(f['value']).cast(int).fill_null(0)
                        elif f['function'] == 3:
                            _filter_out *= (routes[f['column']].cast(str).str.contains(f['value']) == False).cast(int).fill_null(1)
                        elif f['function'] == 4:
                            _filter_out *= (routes[f['column']].is_in(f['value']) == False).cast(int).fill_null(1)
                        else:
                            raise Exception(f"Function {f['function']} in route_filter not implemented.")
                    
                if (route_filter['function'] < 3).any(): 
                    routes = routes.filter((_filter_in > 0) & (pl.DataFrame({'in':_filter_in,'out':_filter_out}).max_horizontal() > 0))
                else:
                    routes = routes.filter(_filter_out > 0)

                trips = trips.filter(pl.col('route_id').is_in(routes['route_id']))

            if len(service_ids) > 0:
                trips = trips.filter(trips['service_id'].is_in(service_ids))
            
            trips = trips.unique('trip_id').sort('trip_id')

        return trips    
    
    def __read_calendar(self,service_id_counts=None,reload_data:bool=False):
        if reload_data == True:
            calendar = None
        else:
            try:
                calendar = self.calendar 
            except:
                calendar = None

        if type(calendar) == type(None): 
            files = []
            for j in range(len(self.gtfs_dir)):
                if os.path.isfile(os.path.join(self.gtfs_dir[j],'calendar.txt')):
                    files.append(j)

            if len(files) > 0:
                calendar = pl.concat([self.__read_csv(os.path.join(self.gtfs_dir[j], 'calendar.txt'), 
                                        schema_overrides={'service_id': str,
                                        'monday':int,'tuesday':int,'wednesday':int,'thursday':int,'friday':int,'saturday':int,'sunday':int,
                                        'start_date':str,'end_date':str}
                                        ).with_columns(file_number=j) for j in files
                ], how='diagonal_relaxed')
                calendar = calendar.with_columns(
                    (pl.col('start_date').str.strip_chars().str.replace_all(r"^[^0-9]+|[^0-9]+$", "").str.slice(0, 8)).str.strptime(pl.Date, "%Y%m%d"),
                    (pl.col('end_date').str.strip_chars().str.replace_all(r"^[^0-9]+|[^0-9]+$", "").str.slice(0, 8)).str.strptime(pl.Date, "%Y%m%d"),
                )
                calendar = calendar.with_columns(
                    orig_service_id = pl.col('service_id'),
                    service_id = pl.col('service_id').cast(str) + "_" + pl.col('file_number').cast(str),
                )
                
                if type(service_id_counts) != type(None):
                    # Merge the trip counts with calendar_dates
                    calendar = calendar.join(
                        service_id_counts,
                        on="service_id",
                        how="left"  # Use 'left' to keep all entries in calendar_dates_df
                    )
                    calendar = calendar.filter((pl.col('n_stops').is_not_null()) & (pl.col('n_stops') > 0))
                else:
                    calendar = calendar.with_columns(n_trips=0,n_stops=0)
            else:
                calendar = pl.DataFrame(schema=['service_id','orig_service_id','monday','tuesday','wednesday','thursday','friday','saturday','sunday',
                                                'start_date','end_date','file_number','n_trips','n_stops'])
                
        return calendar.unique(['service_id','start_date','end_date'])
    
    def __read_calendar_dates(self,service_id_counts=None,reload_data:bool=False):
        if reload_data == True:
            calendar_dates = None
        else:
            try:
                calendar_dates = self.calendar_dates 
            except:
                calendar_dates = None

        if type(calendar_dates) == type(None): 
            files = []
            for j in range(len(self.gtfs_dir)):
                if os.path.isfile(os.path.join(self.gtfs_dir[j],'calendar_dates.txt')):
                    files.append(j)

            if len(files) > 0:
                calendar_dates = pl.concat([self.__read_csv(os.path.join(self.gtfs_dir[j], 'calendar_dates.txt'), 
                                        schema_overrides={'service_id': str,'exception_type':int},
                                        ).with_columns(file_number=j) for j in files
                ], how='diagonal_relaxed')
                calendar_dates = calendar_dates.with_columns(
                    (pl.col('date').cast(str).str.strip_chars().str.replace_all(r"^[^0-9]+|[^0-9]+$", "").str.slice(0, 8)).str.strptime(pl.Date, "%Y%m%d"),
                    pl.col('exception_type').cast(str).str.strip_chars().str.replace_all(r"^[^0-9]+|[^0-9]+$", "").str.slice(0, 1).cast(int)
                )
                calendar_dates = calendar_dates.filter(~pl.all_horizontal(pl.exclude('file_number').is_null()))
                calendar_dates = calendar_dates.with_columns(
                    orig_service_id = pl.col('service_id'),
                    service_id = pl.col('service_id').cast(str) + "_" + pl.col('file_number').cast(str),
                )

                if type(service_id_counts) != type(None):
                    # Merge the trip counts with calendar_dates
                    calendar_dates = calendar_dates.join(
                        service_id_counts,
                        on="service_id",
                        how="left"  # Use 'left' to keep all entries in calendar_dates_df
                    )
                    calendar_dates = calendar_dates.filter((pl.col('n_stops').is_not_null()) & (pl.col('n_stops') > 0))
                else:
                    calendar_dates = calendar_dates.with_columns(n_trips=0,n_stops=0)

            else:
                calendar_dates = pl.DataFrame(schema=['service_id', 'orig_service_id', 'date', 'exception_type','file_number','n_trips','n_stops'])

        return calendar_dates.unique(['service_id','date','exception_type'])
    
    def __read_routes(self,trips=None,reload_data:bool=False):
        if reload_data == True:
            routes = None
        else:
            try:
                routes = self.routes 
            except:
                routes = None

        if type(routes) == type(None): 
            routes = pl.concat([self.__read_csv(os.path.join(self.gtfs_dir[j], 'routes.txt'), 
                                        schema_overrides={'route_id':str,'route_type':int}
                                ).with_columns(file_number = j) for j in range(len(self.gtfs_dir))], how='diagonal_relaxed')
            routes = routes.filter(~pl.all_horizontal(pl.exclude('file_number').is_null()))
            routes = routes.with_columns(
                orig_route_id = pl.col('route_id'),
                route_id = pl.col('route_id').cast(str) + "_" + pl.col('file_number').cast(str),
            )

            if type(trips) != type(None):
                routes = routes.filter(routes['route_id'].is_in(trips['route_id']))

        return routes.unique(['route_id'])
    
    def __read_stop_times(self,trips=None,reload_data:bool=False):
        if reload_data == True:
            stop_times = None
        else:
            try:
                stop_times = self.stop_times 
            except:
                stop_times = None

        if type(stop_times) == type(None):
            stop_times = pl.concat([self.__read_csv(os.path.join(self.gtfs_dir[j], 'stop_times.txt'), schema_overrides={'trip_id':str,
                'arrival_time':str,'departure_time':str,'stop_id':str,'shape_dist_traveled':str,'stop_sequence':int}).with_columns(file_number = j) for j in range(len(self.gtfs_dir))], how='diagonal_relaxed')
            
            stop_times = stop_times.filter(~pl.col('stop_id').is_null())
            stop_times = stop_times.filter(~pl.col('trip_id').is_null())
            stop_times = stop_times.with_columns(
                orig_trip_id = pl.col('trip_id'),
                trip_id = pl.col('trip_id').cast(str) + "_" + pl.col('file_number').cast(str),
                orig_stop_id = pl.col('stop_id'),
                stop_id = pl.col('stop_id').cast(str) + "_" + pl.col('file_number').cast(str),
            )

            if type(trips) != type(None):
                stop_times = stop_times.filter(pl.col('trip_id').is_in(trips['trip_id']))

            stop_times = stop_times.with_columns(
                arrival_time = ("0" + pl.col('arrival_time').cast(str)).str.slice(-8,8),
                departure_time = ("0" + pl.col('departure_time').cast(str)).str.slice(-8,8),
                orig_stop_sequence=pl.col('stop_sequence')
            )

            stop_times = stop_times.with_columns(
                departure_time = pl.when(
                    pl.col('departure_time').is_null()).then(
                    pl.col('arrival_time')).otherwise(
                    pl.col('departure_time')
                )
            )

            stop_times = stop_times.with_columns(
                arrival_time = pl.when(
                    pl.col('arrival_time').is_null()).then(
                    pl.col('departure_time')).otherwise(
                    pl.col('arrival_time')
                )
            )
            
            stop_times = stop_times.with_columns(
                departure_time_secs = pl.col('departure_time').str.slice(0, 2).cast(int) * 3600 + 
                    pl.col('departure_time').str.slice(3, 2).cast(int) * 60 + 
                    pl.col('departure_time').str.slice(6, 2).cast(int),
                arrival_time_secs = pl.col('arrival_time').str.slice(0, 2).cast(int) * 3600 + 
                    pl.col('arrival_time').str.slice(3, 2).cast(int) * 60 + 
                    pl.col('arrival_time').str.slice(6, 2).cast(int),
            ).sort('trip_id','stop_sequence')

            stop_times = stop_times.with_columns(
                pl.col('departure_time_secs').interpolate(),
                pl.col('arrival_time_secs').interpolate(),
            ).filter(pl.col('departure_time_secs').is_null() == False)

            stop_times = stop_times.with_columns(
                departure_time = self.to_hhmmss(stop_times,'departure_time_secs'),
                arrival_time = self.to_hhmmss(stop_times,'arrival_time_secs'),
            )

            stop_times =  stop_times.with_columns(
                departure_time_secs_24 = pl.when(pl.col("departure_time_secs")>=24*3600).then(
                    pl.col("departure_time_secs")-(24*3600)).otherwise(
                        pl.col("departure_time_secs")
                    )
            )

        return stop_times.unique(['trip_id','stop_id','departure_time_secs']).sort('trip_id','departure_time','stop_sequence')
    
    def __read_shapes(self,trips,stop_times,stops_pl,reload_data:bool=False):
        if reload_data == True:
            shapes = None
        else:
            try:
                shapes = self.shapes 
            except:
                shapes = None

        if type(shapes) == type(None):
            files = []
            for j in range(len(self.gtfs_dir)):
                if os.path.isfile(os.path.join(self.gtfs_dir[j],'shapes.txt')):
                    files.append(j)

            if len(files) > 0:
                shapes = pl.concat([self.__read_csv(os.path.join(self.gtfs_dir[j], 'shapes.txt'), 
                                schema_overrides={'shape_id': str,'shape_pt_lat':float,'shape_pt_lon':float,'shape_pt_sequence':pl.UInt32}
                                ).with_columns(file_number = j) for j in files], how='diagonal_relaxed')
                shapes = shapes.filter(~pl.all_horizontal(pl.exclude('file_number').is_null()))
                shapes = shapes.with_columns(
                    orig_shape_id = pl.col('shape_id'),
                    shape_id = pl.col('shape_id').cast(str) + "_" + pl.col('file_number').cast(str),
                )

                if type(trips) != type(None):
                    shapes = shapes.filter(shapes['shape_id'].is_in(trips['shape_id']))

                shapes = shapes.select(['shape_id','shape_pt_lat','shape_pt_lon','shape_pt_sequence'])
        
                missing_shapes = trips.select(['shape_id','rep_trip_id']).filter(pl.col('shape_id').is_in(shapes['shape_id']) == False)
            else:
                missing_shapes = trips.select(['shape_id','rep_trip_id'])

            if len(missing_shapes) > 0:
                missing_shapes = missing_shapes.unique('shape_id')
                missing_shapes_stops = stop_times.select(['trip_id','rep_trip_id','stop_id','stop_sequence']).filter(pl.col('rep_trip_id')==pl.col('trip_id'))
                missing_shapes_stops = missing_shapes_stops.filter(pl.col('rep_trip_id').is_in(missing_shapes['rep_trip_id']))
                missing_shapes_stops = missing_shapes_stops.join(stops_pl.select(['stop_id','stop_lat','stop_lon']),on='stop_id',how='inner')
                missing_shapes_stops = missing_shapes_stops.rename({
                    'rep_trip_id':'shape_id',
                    'stop_lat':'shape_pt_lat',
                    'stop_lon':'shape_pt_lon',
                    'stop_sequence':'shape_pt_sequence'
                }).select(['shape_id','shape_pt_lat','shape_pt_lon','shape_pt_sequence'])
                #missing_shapes_stops = missing_shapes_stops.with_columns(shape_pt_sequence=pl.col('shape_pt_sequence').cast(int))
                #missing_shapes_stops = missing_shapes_stops.group_by('rep_trip_id').agg(
                #    shape_id = pl.col('rep_trip_id').first(),
                #    shape_pt_lat=pl.col('stop_lat'),
                #    shape_pt_lon=pl.col('stop_lon'),
                #    shape_pt_sequence=pl.col('stop_sequence')
                #).explode(pl.exclude('rep_trip_id','shape_id')).select(pl.exclude('rep_trip_id'))

                if len(files) > 0:
                    shapes = pl.concat([shapes,missing_shapes_stops])
                else:
                    shapes = missing_shapes_stops
                
            self.shapes_pl = shapes
            shapes = shapes.group_by('shape_id').agg(
                    (pl.col('shape_pt_lon').sort_by('shape_pt_sequence').cast(str) + " " + pl.col('shape_pt_lat').sort_by('shape_pt_sequence').cast(str)).alias('wkt')
                ).with_columns(
                    wkt= pl.format(
                        "LINESTRING({})", 
                        pl.col("wkt").cast(pl.List(pl.Utf8)).list.join(", ")
                    )
                )
            #gdf = gpd.GeoDataFrame(gdf.to_pandas(), geometry=gpd.points_from_xy(gdf['shape_pt_lon'], gdf['shape_pt_lat'],crs=self.crs))
            #gdf = gpd.GeoDataFrame(gdf.groupby('shape_id')['geometry'].apply(lambda x: LineString(x.to_list()))) # geopolars
            shapes = gpd.GeoDataFrame(shapes['shape_id'].to_pandas(),geometry=gpd.GeoSeries.from_wkt(shapes['wkt'].to_pandas(),on_invalid='warn'),crs=self.crs)
            shapes = shapes.reset_index(drop=True)
            shapes = shapes.set_crs(epsg=self.crs)
            shapes =  shapes.drop_duplicates('shape_id')

        return shapes
    
    def __set_bounds(self,bounds,strict_bounds:bool=False,stop_id_list=[],service_ids=[],route_filter=None):
        minx,miny,maxx,maxy = self.bounds.geometry.to_crs(self.crs).total_bounds
        stops_pl = self.__read_stops_pl(stop_id_list=stop_id_list)
        stop_ids_in_bounds = stops_pl.filter((pl.col('stop_lon') > minx) & (pl.col('stop_lon') < maxx
                                                                            ) & (
                                            pl.col('stop_lat') > miny) & (pl.col('stop_lat') < maxy))
        stop_ids_in_bounds = list(stop_ids_in_bounds['stop_id'])
        if len(stop_ids_in_bounds) == 0:
            assert "No stops found inside bounds"

        all_trips = self.__read_trips(service_ids=service_ids,route_filter=route_filter)
        stop_times = self.__read_stop_times(trips=all_trips)

        stop_times_in_bounds = stop_times.filter(pl.col('stop_id').is_in(stop_ids_in_bounds))
        trip_ids_in_bounds = stop_times_in_bounds['trip_id'].unique()

        all_trips = all_trips.filter(pl.col('trip_id').is_in(trip_ids_in_bounds))
        if strict_bounds:
            stop_times = stop_times_in_bounds
            stops_pl = stops_pl.filter(pl.col('stop_id').is_in(stop_ids_in_bounds))
        else:
            stop_times = stop_times.filter(pl.col('trip_id').is_in(all_trips['trip_id']))
            stops_pl = stops_pl.filter(pl.col('stop_id').is_in(stop_times['stop_id'].unique()))

        stops_gdf = self.__read_stops_gdf(stops_pl=stops_pl,stop_id_list=stop_id_list)
        stops_gdf = stops_gdf.loc[stops_gdf['stop_id'].isin(stop_times['stop_id'].unique().to_list())].reset_index(drop=True)
        geoms = list(stops_gdf.geometry)
        shapely.prepare(geoms)
        stop_ids_in_bounds = list(stops_gdf.loc[shapely.intersects(geoms,bounds.geometry.to_crs(self.crs).union_all()),'stop_id'])
        if len(stop_ids_in_bounds) == 0:
            assert "No stops found inside bounds"

        stop_times_in_bounds = stop_times.filter(pl.col('stop_id').is_in(stop_ids_in_bounds))
        trip_ids_in_bounds = stop_times_in_bounds['trip_id'].unique()
        all_trips = all_trips.filter(pl.col('trip_id').is_in(trip_ids_in_bounds))
        if strict_bounds:
            stop_times = stop_times_in_bounds
            stops_pl = stops_pl.filter(pl.col('stop_id').is_in(stop_ids_in_bounds))
        else:
            stop_times = stop_times.filter(pl.col('trip_id').is_in(trip_ids_in_bounds))
            stops_pl = stops_pl.filter(pl.col('stop_id').is_in(stop_times['stop_id'].unique()))

        return stops_pl, stops_gdf, all_trips, stop_times, stop_ids_in_bounds
    
    def __frequencies_to_trips(self,stop_times,trips,frequencies,start_time,end_time):
        """
        For each trip_id in frequencies.txt, calculates the number
        of trips and creates records for each trip in trips.txt and
        stop_times.txt. Deletes the original represetative trip_id
        in both of these files. 
        """

        frequencies = frequencies.filter(~pl.all_horizontal(pl.all().is_null()))
        frequencies = frequencies.filter(pl.col('trip_id').is_in(trips['trip_id']))#.sort('trip_id')

        if len(frequencies) == 0:
            return stop_times, trips
        
        start_time_secs = int(start_time[0:2]) * 3600 + int(start_time[3:5]) * 60 + int(start_time[6:8])
        end_time_secs = int(end_time[0:2]) * 3600 + int(end_time[3:5]) * 60 + int(end_time[6:8])

        # some feeds will use the same trip_id for multiple rows
        # need to create a unique id for each row
        frequencies = frequencies.with_columns(
            start_time_secs = self.convert_to_seconds(frequencies,"start_time"),
            end_time_secs = self.convert_to_seconds(frequencies,"end_time")
        )

        frequencies =  frequencies.with_columns(
            frequency_id = np.arange(0,len(frequencies)),
            start_time_secs_24 = pl.when(pl.col("start_time_secs")>=24*3600).then(
                pl.col("start_time_secs")-(24*3600)).otherwise(
                    pl.col("start_time_secs")
                ),
            end_time_secs_24 = pl.when(pl.col("end_time_secs")>=24*3600).then(
                pl.col("end_time_secs")-(24*3600)).otherwise(
                    pl.col("end_time_secs")
                )
        )
        frequencies = frequencies.filter(
            (pl.col('start_time_secs_24') <= end_time_secs) | (pl.col('end_time_secs_24') >= start_time_secs)
        )

        # following is coded so the total number of trips
        # does not include a final one that leaves the first
        # stop at end_time in frequencies. I think this is the
        # correct interpredtation of the field description:
        # 'Time at which service changes to a different headway
        # (or ceases) at the first stop in the trip.'

        # Rounding total trips to make sure all trips are counted
        # when end time is in the following format: 14:59:59,
        # instead of 15:00:00.

        frequencies = frequencies.with_columns(
            total_trips = pl.Series((frequencies['end_time_secs']-frequencies['start_time_secs']) / frequencies['headway_secs']).ceil().cast(int) -1 #################### !!! polars rounds 22.5 to 23 but numpy eounds to 22
        )

        trips_update = trips.join(frequencies, on='trip_id').filter(pl.col('trip_id').is_in(frequencies['trip_id']))

        trips_update = trips_update.select(
            pl.all().repeat_by('total_trips').explode()
        )

        trips_update = trips_update.with_columns(
            counter = pl.col('trip_id').cum_count().over('trip_id'),
            main_trip_id = pl.col('trip_id')
        )

        trips_update = trips_update.with_columns(
            trips_update['trip_id'].cast(str) + '_' + trips_update['counter'].cast(str)
        )

        stop_times_update = stop_times.join(frequencies, on='trip_id').filter(pl.col('trip_id').is_in(frequencies['trip_id']))

        elapsed_time = stop_times_update.group_by(['trip_id', 'start_time_secs']).agg(
            pl.col('arrival_time_secs'),
            pl.col('arrival_time_secs').min().alias('elapsed_time')).explode("arrival_time_secs")

        stop_times_update = stop_times_update.join(elapsed_time, on=['trip_id','start_time_secs','arrival_time_secs'], how='right')
        stop_times_update = stop_times_update.with_columns(elapsed_time = stop_times_update['arrival_time_secs'] - stop_times_update['elapsed_time'])

        stop_times_update = stop_times_update.with_columns(
            arrival_time_secs = (stop_times_update['start_time_secs'] + stop_times_update['elapsed_time']),
            departure_time_secs = (stop_times_update['start_time_secs'] + stop_times_update['elapsed_time'])
        )

        stop_times_update = stop_times_update.select(
            pl.all().repeat_by('total_trips').explode()
        )

        stop_times_update = stop_times_update.with_columns(
            counter = pl.col(['stop_sequence']).cum_count().over(['frequency_id', 'stop_sequence']) - 1
        )

        stop_times_update = stop_times_update.with_columns(
            departure_time_secs = (stop_times_update['departure_time_secs'] + (stop_times_update[
                'counter'] * stop_times_update['headway_secs'])),
            arrival_time_secs = (stop_times_update['arrival_time_secs'] + (stop_times_update[
                'counter'] * stop_times_update['headway_secs']))
        )

        stop_times_update =  stop_times_update.with_columns(
            departure_time_secs_24 = pl.when(pl.col("departure_time_secs")>=24*3600).then(
                pl.col("departure_time_secs")-(24*3600)).otherwise(
                    pl.col("departure_time_secs")
                )
        )

        stop_times_update = stop_times_update.with_columns(
            departure_time = self.to_hhmmss(stop_times_update,"departure_time_secs"),
            arrival_time = self.to_hhmmss(stop_times_update,"arrival_time_secs"),
            trip_id = (stop_times_update['trip_id'].cast(str) + '_' + pl.col(['stop_sequence']).cum_count().over(['trip_id', 'stop_sequence']).cast(str))
        )

        # remove trip_ids that are in frequencies
        stop_times = stop_times.filter(~stop_times['trip_id'].is_in(frequencies['trip_id']))

        trips = trips.filter(~trips['trip_id'].is_in(frequencies['trip_id']))

        # get rid of some columns
        stop_times_update = stop_times_update[stop_times.collect_schema().names()]

        trips_update = trips_update[trips.collect_schema().names()]

        # add new trips/stop times
        trips = pl.concat([trips, trips_update])
        stop_times = pl.concat([stop_times, stop_times_update])

        return stop_times.sort('trip_id','stop_sequence'), trips.sort('trip_id')
    
    def __set_time_bounds(self,stop_times, trips, frequencies=None, start_time=None,end_time=None):
        """
        Creates a merged dataframe consisting of trips & stop_ids for the
        start time, end time and service_id (from GTFS Calender.txt). This
        can include partial itineraries as only stops within the start and
        end time are included.
        """
        if type(frequencies) == type(None):
            frequencies = self.__read_frequencies()

        stop_times, trips = self.__frequencies_to_trips(stop_times=stop_times,trips=trips,frequencies=frequencies,start_time=start_time,end_time=end_time)
        
        start_time_secs = int(start_time[0:2]) * 3600 + int(start_time[3:5]) * 60 + int(start_time[6:8])
        end_time_secs = int(end_time[0:2]) * 3600 + int(end_time[3:5]) * 60 + int(end_time[6:8])
        if end_time_secs > start_time_secs: 
            stop_times = stop_times.group_by('trip_id').agg(
                    pl.all(),
                    pl.col('departure_time_secs_24').first().alias('time_first'),
                    pl.col('departure_time_secs_24').last().alias('time_last')
                ).filter(
                    (pl.col('time_last') > start_time_secs) & (pl.col('time_first') < end_time_secs)
                ).drop(['time_first','time_last']).explode(pl.exclude('trip_id'))
        elif end_time_secs < start_time_secs:
            stop_times = stop_times.group_by('trip_id').agg(
                    pl.all(),
                    pl.col('departure_time_secs_24').first().alias('time_first'),
                    pl.col('departure_time_secs_24').last().alias('time_last')
                ).filter(
                    (pl.col('time_last') > start_time_secs) | (pl.col('time_first') < end_time_secs)
                ).drop(['time_first','time_last']).explode(pl.exclude('trip_id'))
        stop_times = stop_times.with_columns(
            departure_time_hrs = (stop_times['departure_time_secs']/3600).cast(int)   
        )

        stop_times = stop_times.join(trips.drop('orig_trip_id'), how='left', on='trip_id')

        trips = trips.filter(pl.col('trip_id').is_in(stop_times['trip_id']))
        return stop_times.sort('trip_id','stop_sequence'), trips.sort('trip_id')
    
    def __get_service_id_counts(self,stop_times,trips,frequencies=[],stop_ids_in_bounds=[]):
        stop_counts = stop_times.select('trip_id','stop_id').with_columns(n_stops=1)
        if len(stop_ids_in_bounds) > 0:
            stop_counts = stop_counts.filter(pl.col('stop_id').is_in(stop_ids_in_bounds))

        if len(frequencies) > 0:
            frequencies = frequencies.with_columns(
                start_time_secs = self.convert_to_seconds(frequencies,"start_time"),
                end_time_secs = self.convert_to_seconds(frequencies,"end_time")
            )
            frequencies = frequencies.with_columns(
                n_freq = pl.Series((frequencies['end_time_secs']-frequencies['start_time_secs']) / frequencies['headway_secs']).ceil().cast(int) -1 #################### !!! polars rounds 22.5 to 23 but numpy eounds to 22
            ).group_by('trip_id').agg(pl.col('n_freq').sum())
            stop_counts = stop_counts.join(frequencies.select('n_freq','trip_id'),on='trip_id',how='left').fill_null(0) 
            stop_counts = stop_counts.with_columns(
                n_stops = pl.col('n_stops') + pl.col('n_freq')
            ).drop('n_freq')

        stop_counts = stop_counts.group_by('trip_id').agg(
            pl.col('n_stops').sum().cast(int)
        )
        stop_counts = stop_counts.join(trips.select('trip_id','service_id'),on='trip_id').fill_null(0)

        service_id_counts = stop_counts.group_by("service_id").agg(
            pl.col('n_stops').sum().cast(int),
        )
   
        return service_id_counts

    def to_weekday(self, my_date,column='date'):
        """
        Gets the day of week from user parameter service date.
        """
        if (type(my_date) == datetime) or (type(my_date) == str) or (type(my_date) == int):
            if type(my_date) != datetime:
                my_date = str(my_date)
                my_date = datetime(int(my_date[0:4]), int(
                        my_date[4:6]), int(my_date[6:8]))
                
            week_days = ['monday', 'tuesday', 'wednesday',
                        'thursday', 'friday', 'saturday', 'sunday']
            return week_days[my_date.weekday()]
        else:
            if my_date[column].dtype == pl.Date:
                my_date = my_date.with_columns(
                    weekday = pl.col(column)
                )
            else:
                my_date = my_date.with_columns(
                    weekday = pl.col(column).cast(str).str.strptime(pl.Date, "%Y%m%d")
                )
            
            my_date = my_date.with_columns(weekday = pl.col('weekday').dt.weekday())
            my_date = my_date.with_columns(
                weekday = pl.when(pl.col("weekday") == 1).then(pl.lit("monday"))
                .when(pl.col("weekday") == 2).then(pl.lit("tuesday"))
                .when(pl.col("weekday") == 3).then(pl.lit("wednesday"))
                .when(pl.col("weekday") == 4).then(pl.lit("thursday"))
                .when(pl.col("weekday") == 5).then(pl.lit("friday"))
                .when(pl.col("weekday") == 6).then(pl.lit("saturday"))
                .when(pl.col("weekday") == 7).then(pl.lit("sunday"))
            )
            return my_date

    def __get_counts_by_date(self,calendar,calendar_dates):
        if len(calendar) > 0:
            min_date = calendar['start_date'].min() 
            max_date = calendar['end_date'].max() 
            if len(calendar_dates) > 0:
                min_date = min(calendar['start_date'].min(),calendar_dates['date'].min())
                max_date = max(calendar['end_date'].max(),calendar_dates['date'].max())
        else:
            min_date = calendar_dates['date'].min()
            max_date = calendar_dates['date'].max()

        counts_by_date = pl.DataFrame({'date':pl.date_range(min_date,max_date,eager=True)})
        counts_by_date = self.to_weekday(counts_by_date,'date')

        weekdays = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
        if len(calendar) > 0:
            #calendar = calendar.with_columns(start_date = pl.col('start_date').cast(str).str.strptime(pl.Date, "%Y%m%d"),
            #                end_date = pl.col('end_date').cast(str).str.strptime(pl.Date, "%Y%m%d"),)
            calendar = calendar.select([
                            'service_id',
                            'start_date',
                            'end_date',
                            *[(pl.col(day) * pl.col('n_stops')).alias(day)
                            for day in weekdays]
                        ]).fill_null(0)
            calendar = calendar.with_columns(date = pl.date_ranges("start_date", "end_date")).explode('date')
            calendar = calendar.group_by('date').agg(pl.exclude('service_id','start_date','end_date').sum())
            calendar = calendar.with_columns(n_stops_regular=calendar.select(pl.exclude('date')).to_numpy()[
                np.arange(len(calendar)),
                calendar['date'].dt.weekday().to_numpy()-1
            ]).drop(weekdays)

            counts_by_date = counts_by_date.join(calendar,on='date',how='left').fill_null(0)
        else:
            counts_by_date = counts_by_date.with_columns(n_stops_regular=0)

        if len(calendar_dates) > 0:
            #calendar_dates = calendar_dates.with_columns(date = pl.col('date').cast(str).str.strptime(pl.Date, "%Y%m%d"))
            calendar_dates = calendar_dates.select('date','exception_type','n_stops').with_columns(
                n_stops = pl.when(pl.col('exception_type').cast(int) == 1).then(pl.col('n_stops').cast(int)).otherwise(-1 * pl.col('n_stops').cast(int))
            ).group_by('date').agg(pl.col('n_stops').cast(int).sum())

            counts_by_date = counts_by_date.join(calendar_dates.select('date','n_stops'),on='date',how='left').fill_null(0)
            counts_by_date = counts_by_date.with_columns(n_stops = pl.col('n_stops_regular') + pl.col('n_stops'))
        else:
            counts_by_date = counts_by_date.with_columns(n_stops = pl.col('n_stops_regular'))

        return counts_by_date

    def __get_service_ids(self,calendar,calendar_dates,service_date,counts_by_date=None):
        """
        Returns a list of valid service_id(s) from each feed using the user
        specified service_date.
        """
        add_exceptions = True
        if service_date in ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','max','min']:
            if type(counts_by_date) == type(None):
                raise Exception("counts_by_date missing")

            if service_date == 'max':
                service_date = counts_by_date.filter(pl.col('n_stops').max() == pl.col('n_stops'))
            elif service_date == 'min':
                service_date = counts_by_date.filter(pl.col('n_stops').min() == pl.col('n_stops'))
            elif service_date in ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']:
                counts_by_date = counts_by_date.filter(pl.col('weekday') == pl.lit(service_date))
                service_date = counts_by_date.filter(pl.col('n_stops_regular').max() == pl.col('n_stops_regular'))
                service_date = service_date.with_columns(n_stops = pl.col('n_stops_regular'))
                add_exceptions = False
            else:
                raise Exception(f"service_date {service_date} not valid.")

            print(f"Service date found {service_date['date'][0]} {service_date['weekday'][0]} with {service_date['n_stops'][0]} trips x stops.")
            service_date = service_date['date'][0]
            service_date = datetime.combine(service_date, datetime.min.time())

        if type(service_date) != datetime:
            service_date = str(service_date)
            service_date = datetime(year=int(service_date[0:4]), month=int(
                            service_date[4:6]), day=int(service_date[6:8]))
                
        day_of_week = self.to_weekday(service_date)

        regular_service_dates = calendar.filter((
                pl.col('start_date') <= service_date) & (
                pl.col('end_date') >= service_date) & (
                pl.col(day_of_week) == 1))['service_id'].to_list()

        if add_exceptions:
            exceptions_df = calendar_dates.filter(
                pl.col('date') == service_date)

            add_service = exceptions_df.filter(
                pl.col('exception_type') == 1)['service_id'].to_list()

            remove_service = exceptions_df.filter(
                pl.col('exception_type') == 2)['service_id'].to_list()

            service_id_list = [x for x in (
                add_service + regular_service_dates) if x not in remove_service]
        else:
            service_id_list = list(regular_service_dates)

        assert service_id_list, "No service found in feed."

        return service_id_list, service_date

    def convert_to_seconds(self, df, field):
        """
        Convert HH:MM:SS to seconds since midnight
        for comparison purposes.
        """
        df = df.with_columns(("0" + pl.col(field).cast(str)).str.slice(-8,8).alias(field))
        hours = df[field].str.slice(0, 2).cast(int)
        minutes = df[field].str.slice(3, 2).cast(int)
        seconds = df[field].str.slice(6, 2).cast(int)

        time = hours * 3600 + minutes * 60 + seconds

        error = None
        error_time = None
        if hours.min() < 0:
            error = df.filter(hours == hours.min())
            error_time = time.filter(hours == hours.min())
        elif minutes.min() < 0:
            error = df.filter(minutes == minutes.min())
            error_time = time.filter(minutes == minutes.min())
        elif seconds.min() < 0:
            error = df.filter(seconds == seconds.min())
            error_time = time.filter(seconds == seconds.min())
        
        if type(error) != type(None):
            raise Exception(f"Error. Negative time {error_time} in dataframe row {error}")

        return time


    def to_hhmmss(self, df, field):

        hours = ("0" + (df[field] // 3600).cast(int).cast(str)).str.slice(-2,2)
        minutes = ("0" + ((df[field] % 3600) // 60).cast(int).cast(str)).str.slice(-2,2)
        seconds = ("0" + (df[field] % 60).cast(int).cast(str)).str.slice(-2,2)
        return hours + ":" + minutes + ":" + seconds


    def __correct_stop_sequence(self,stop_times, trips):
        stop_times = stop_times.group_by(['trip_id', 'stop_sequence']).agg(
            pl.all().sort_by('departure_time'),
            temp = pl.col('trip_id').alias('temp') + 
                    "_" + 
                    (pl.arange(pl.col('trip_id').len())+1).cast(str)
        ).explode(pl.exclude('trip_id','stop_sequence'))

        stop_times = stop_times.with_columns(
            temp = pl.when(
                pl.col('temp').str.slice(-2,2)=='_1'
            ).then(
                pl.col('temp').str.slice(0,pl.col('temp').str.len_chars()-2)
            ).otherwise(pl.col('temp'))
        )

        trips = trips.join(stop_times[['trip_id','temp']].unique('temp'),on='trip_id',how='left')

        trips = trips.drop('trip_id')
        trips = trips.rename({'temp': 'trip_id'})

        stop_times = stop_times.drop('trip_id')
        stop_times = stop_times.rename({'temp': 'trip_id'})

        return stop_times.sort('trip_id','stop_sequence'), trips.sort('trip_id')
    
    def __sort_sequence_col(self,stop_times):
        stop_times = stop_times.sort('trip_id', 'departure_time', 'stop_sequence').group_by('trip_id').agg(
            pl.all(),
            pl.col('stop_sequence').cum_count().alias('temp')
        ).explode(pl.exclude('trip_id'))

        stop_times = stop_times.drop('stop_sequence')
        stop_times = stop_times.rename({'temp': 'stop_sequence'})

        return stop_times#.unique(['rep_trip_id','stop_id','departure_time_secs'])
    
    def __group_stops(self, stops, stop_times, distance, int_groups:bool=False, _incall:bool=False): ##### stops pandas gdf stop_times polars df !!!!!!!! geopolars
        from sklearn.cluster import AgglomerativeClustering

        if _incall: 
            int_groups = True 

        if 'stop_group_id' in stop_times.collect_schema().names():  
            stop_times = stop_times.drop('stop_group_id')

        if not _incall and 'stop_group_id' in stops.columns:
            stops = stops.drop('stop_group_id')
        
        if (distance > 0) and (len(stops) > 1):
            stop_times = stop_times.sort('trip_id','departure_time','stop_sequence','stop_id')
            geo = stops.geometry.copy()
            if geo.crs.is_projected == False:
                geo = gpd.GeoDataFrame(geo.to_crs(stops.geometry.estimate_utm_crs()))

            geo['x'] = geo.geometry.x
            geo['y'] = geo.geometry.y
            cluster_func = AgglomerativeClustering(n_clusters=None,distance_threshold=distance, metric='euclidean',linkage='complete')
            clusters = cluster_func.fit(geo[['x','y']]).labels_#list(zip(x,y)))

            if int_groups: 
               stop_group_id = pl.DataFrame({'stop_id' : pl.from_pandas(stops['stop_id']),'stop_group_id' : clusters})
            else:
                stop_group_id = pl.DataFrame({'stop_id' : pl.from_pandas(stops['stop_id']),'stop_group_int' : clusters})
                stop_group_id = stop_group_id.group_by('stop_group_int').agg(
                        pl.col('stop_id'),
                        pl.col('stop_id').min().alias('stop_group_id')
                    ).explode('stop_id').drop('stop_group_int')
                
            if not _incall:
                stops = stops.merge(stop_group_id.to_pandas(),on='stop_id') ######### geopolars

            stop_times = stop_times.join(stop_group_id,on='stop_id')
        else:
            if int_groups: 
                stop_group_id = pl.DataFrame({'stop_id' : pl.from_pandas(stops['stop_id']),'stop_group_id' : np.arange(len(stops['stop_id']))})
                if not _incall:
                    stops = stops.merge(stop_group_id.to_pandas(),on='stop_id')

                stop_times = stop_times.join(stop_group_id,on='stop_id')
            else:
                stops['stop_group_id'] = stops['stop_id']
                stop_times = stop_times.with_columns(stop_group_id = pl.col('stop_id'))

        if not _incall:
            stop_groups = pl.from_pandas(stops.drop(columns='geometry')).sort('stop_id')
            stop_groups = stop_groups.group_by('stop_group_id').agg(
                pl.exclude('stop_lat','stop_lon').unique(),
                pl.col('stop_lat').cast(float).mean(),
                pl.col('stop_lon').cast(float).mean()
            ).sort('stop_group_id')
            stop_groups = gpd.GeoDataFrame(stop_groups.to_pandas(), geometry=gpd.points_from_xy(stop_groups['stop_lon'], stop_groups['stop_lat'], crs=self.crs)) ################# geopolars
            stop_groups = stop_groups.set_crs(epsg=self.crs)

        if _incall: 
            return stop_times.sort('trip_id','stop_sequence')
        else:
            return stop_groups, stops.sort_values(['stop_id']), stop_times.sort('trip_id','stop_sequence')

    def __get_schedule_pattern(self,stop_times,trips,routes):
        """
        Returns a DataFrame with a field for each trip_id used
        to represent a unique stop_pattern (rep_trip_id) and a
        column with the other trip_ids that share the same stop
        pattern
        """
        df = stop_times.group_by('trip_id').agg(
                pl.col('stop_group_id').unique().sort().alias('stop_group_id_no_squence'),
                pl.col(['stop_group_id','stop_sequence']).sort_by('stop_sequence')
        ).group_by('stop_group_id').agg(
            pl.col('trip_id'),
            pl.col('stop_group_id_no_squence'),
            pl.col('trip_id').min().alias("rep_trip_id")
        ).explode(['trip_id','stop_group_id_no_squence']).group_by('stop_group_id_no_squence').agg(
            pl.col('trip_id'),
            pl.col('rep_trip_id'),
            pl.col('rep_trip_id').min().alias('rep_route_id') 
        ).explode(['trip_id','rep_trip_id'])
        
        """
        df = df.group_by('trip_id').agg(
            (pl.col('stop_group_id').first().list.concat(
            pl.col('stop_group_id').last())).sort().alias('first_and_last_stops')
        )
        df = df.group_by(['stop_group_id','first_and_last_stops']).agg(
            pl.col('trip_id')
        ) check that lines only have first and last stops in common.
        circular routes with same route_id
        df = df.group_by('first_and_last_stops').agg(
                pl.col('trip_id'),
                pl.col('rep_trip_id'),
                pl.col('rep_trip_id').min().alias("rep_route_id")
            ).explode(['trip_id','rep_trip_id'])   
        

        df = df.group_by('stop_group_id_no_squence').agg(
            pl.col('trip_id'),
            pl.col('rep_trip_id'),
            pl.col('rep_route_id').min() 
        ).explode(['trip_id','rep_trip_id'])
        """
        
        stop_times = stop_times.join(df.select(['trip_id', 'rep_trip_id','rep_route_id']), how='left', on='trip_id')
        trips = trips.join(df.select(['trip_id', 'rep_trip_id','rep_route_id']), how='left', on='trip_id')
        routes = routes.join(trips.select(['route_id','rep_route_id']).unique('route_id'),on='route_id',how='left')

        return stop_times.sort('trip_id','stop_sequence'), trips.sort('trip_id'), routes.sort('route_id')
    
    def __group_trips(self,stops,stop_times,trips,routes,distance,overlap,int_groups:bool=False):
        if 'trip_group_id' in stop_times.collect_schema().names():  
            stop_times = stop_times.drop(['trip_group_id','route_group_id'])

        if 'trip_group_id' in trips.collect_schema().names():  
            trips = trips.drop(['trip_group_id','route_group_id'])

        if 'trip_group_id' in routes.collect_schema().names():  
            routes = routes.drop(['route_group_id','trip_group_id'])

        if (distance > 0) and (len(stops) > 1):
            from sklearn.cluster import AgglomerativeClustering
            stop_times = stop_times.sort('trip_id','departure_time','stop_sequence','stop_id')
            trips = trips.sort('trip_id')
            routes = routes.sort('route_id')
            stop_times_group = self.__group_stops(stops,stop_times,distance,int_groups=True,_incall=True)
            stop_times_group = stop_times_group.sort('trip_id','departure_time','stop_sequence')
            stop_times_group = stop_times_group.filter(pl.col('trip_id')==pl.col('rep_trip_id'))['stop_group_id','rep_trip_id','trip_id','stop_sequence']
            stop_times_group = stop_times_group.group_by('rep_trip_id').agg(
                    pl.col('stop_group_id').sort_by(pl.col('stop_sequence')).unique(maintain_order=True)
                )
            
            stops_list = stop_times_group['stop_group_id'].to_list()
            stop_times_group = stop_times_group.with_columns(
                    pl.from_dict({stop_times_group['rep_trip_id'][i]:stop_times_group['stop_group_id'].list.set_intersection(stops_list[i]) for i in range(len(stop_times_group))})
                )
            
            first_stop_id = stop_times_group.select(pl.exclude('rep_trip_id','stop_group_id','stop_sequence').list.first()).to_numpy()
            last_stop_id = stop_times_group.select(pl.exclude('rep_trip_id','stop_group_id','stop_sequence').list.last()).to_numpy()
            trips_overlap = stop_times_group.select(
                    pl.exclude('rep_trip_id','stop_group_id','stop_sequence').list.len() * (pl.col('stop_group_id').list.len() > 1) / pl.col('stop_group_id').list.len()
                ).to_numpy()

            routes_overlap = trips_overlap * (trips_overlap >= trips_overlap.transpose()) + trips_overlap.transpose() * (trips_overlap < trips_overlap.transpose())
            trips_overlap = trips_overlap * (first_stop_id == first_stop_id.transpose()) * (last_stop_id == last_stop_id.transpose())
            trips_overlap = trips_overlap * (trips_overlap >= trips_overlap.transpose()) + trips_overlap.transpose() * (trips_overlap < trips_overlap.transpose())
            cluster_func = AgglomerativeClustering(n_clusters=None,distance_threshold=1-overlap, metric='precomputed',linkage='complete')
            trips_clustering = cluster_func.fit(1-trips_overlap).labels_
            routes_clustering = cluster_func.fit(1-routes_overlap).labels_

            if int_groups: 
                trip_group_id = pl.DataFrame({'rep_trip_id':stop_times_group['rep_trip_id'],
                                            'trip_group_id':trips_clustering,
                                            'route_group_id':routes_clustering})
            else:
                trip_group_id = pl.DataFrame({'rep_trip_id':stop_times_group['rep_trip_id'],
                                            'trip_group_int':trips_clustering,
                                            'route_group_int':routes_clustering})
               
                trip_group_id = trip_group_id.group_by('trip_group_int').agg(
                        pl.col('rep_trip_id'),
                        pl.col('route_group_int'),
                        pl.col('rep_trip_id').min().alias('trip_group_id')
                    ).explode(['rep_trip_id','route_group_int']).drop('trip_group_int')
                trip_group_id = trip_group_id.group_by('route_group_int').agg(
                        pl.col('rep_trip_id'),
                        pl.col('trip_group_id'),
                        pl.col('rep_trip_id').min().alias('route_group_id')
                    ).explode(['trip_group_id','rep_trip_id']).drop('route_group_int')

            stop_times = stop_times.join(trip_group_id,on='rep_trip_id',how='left')
            trips = trips.join(trip_group_id,on='rep_trip_id',how='left')

        else: 
            if int_groups: 
                trip_group_id = pl.DataFrame({'rep_trip_id':stop_times_group['rep_trip_id'].unique(),
                                              'trip_group_id' : np.arange(len(trips['rep_trip_id'].unique())),
                                              'route_group_id' : np.arange(len(trips['rep_route_id'].unique()))})
                stop_times = stop_times.join(trip_group_id,on='rep_trip_id',how='left')
                trips = trips.join(trip_group_id,on='rep_trip_id',how='left')
            else:
                stop_times = stop_times.with_columns(trip_group_id=pl.col('rep_trip_id'),
                                                     route_group_id=pl.col('rep_route_id'))
                trips = trips.with_columns(trip_group_id=pl.col('rep_trip_id'),
                                           route_group_id=pl.col('rep_route_id'))
        
        routes = routes.join(trips.select(['route_id','route_group_id']).unique('route_id'),on='route_id',how='left')

        return stop_times.sort('trip_id','stop_sequence'), trips.sort('trip_id'), routes.sort('route_id')
    
    def change_data(self, gtfs_dir:str|list[str]=None, service_date:str|list[str]=None, 
                 start_time:str=None,end_time:str=None, 
                 bounds:gpd.GeoSeries|gpd.GeoDataFrame=None, strict_bounds:bool=None,
                 stop_group_distance=None,trip_group_distance=None,trip_group_overlap=None,
                 correct_stop_sequence=None,crs=None):
        """
        Instantiate class with directory of GTFS Files and a service_date for
        which to get service.
        """
        state = 999
        if type(gtfs_dir) == str:
            gtfs_dir = [gtfs_dir]

        if type(service_date) == str:
            service_date = [service_date]

        if gtfs_dir != self.gtfs_dir:
            self.gtfs_dir = gtfs_dir
            state = 0

        if service_date != self.service_date:
            self.service_date = service_date
            state = 0

        if type(bounds) != type(None) and bounds != self.bounds:
            if len(bounds) == 0:
                bounds = None
            self.bounds = bounds
            state = 0
        
        if strict_bounds != self.strict_bounds:
            self.strict_bounds = strict_bounds
            state = 0

        if self.crs != crs:
            self.crs = crs
            state = 0

        if self.start_time != start_time:
            self.start_time = start_time
            state = min(state,1)

        if self.end_time != end_time:
            self.end_time = end_time
            state = min(state,1)

        if self.stop_group_distance != stop_group_distance:
            self.stop_group_distance = stop_group_distance 
            state = min(state,3)

        if self.trip_group_distance != trip_group_distance:
            self.trip_group_distance = trip_group_distance
            state = min(state,4)

        if self.trip_group_overlap != trip_group_overlap:
            self.trip_group_overlap = trip_group_overlap
            state = min(state,4)

        if self.correct_stop_sequence != correct_stop_sequence:
            self.correct_stop_sequence = correct_stop_sequence
            state = 0
            
        if state != 999:
            self.__load_gtfs(state=state)

        return None

    def get_tph_by_line(self,trip_groups:bool=True):
        """
        Returns a DataFrame with records for each rep_trip_id and
        columns with the number of trips for each hour after midnight
        with service. For example 2:00-3:00 AM is called hour_2 and
        3:00-4:00 PM is called hour_15.
        """
        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'

        # get the first stop for every trip
        first_departure = self.stop_times.sort('stop_sequence').group_by('trip_id').first()
        # this may not be necessary
        first_departure = first_departure.filter((
            first_departure['stop_sequence'] == 1))
        first_departure = first_departure.group_by([trip_group_id, 'departure_time_hrs']).agg(
            pl.col('departure_time_hrs').len().alias('frequency')).sort([trip_group_id,'departure_time_hrs']
        )
        t = first_departure.pivot(values='frequency', index=[trip_group_id],
            on=['departure_time_hrs'])

        col_hours = []
        for col in t.collect_schema().names():
            if not col == trip_group_id:
                col_hours.append(int(col))

        col_hours = list(np.sort(col_hours).astype(str))        

        t = t.join(self.trips[['trip_group_id', 'route_id', 'rep_trip_id', 'direction_id']].group_by(trip_group_id).agg(pl.all().unique()),
                    how='left', on=trip_group_id)
        
        t = t[['trip_group_id','route_id','rep_trip_id','direction_id'] + col_hours]

        for col in col_hours:
            t = t.rename({col: 'hour_' + str(col)})

        t = t.fill_null(0)
        
        return t

    def get_tph_at_stops(self,stop_groups:bool=True):
        """
        Returns a DataFrame with records for each stop_id and
        columns with the number of trips for each hour after midnight
        with service. For example 2:00-3:00 AM is called hour_2 and
        3:00-4:00 PM is called hour_15.
        """
        if stop_groups:
            stop_group_id = 'stop_group_id'
        else:
            stop_group_id = 'stop_id'

        df = self.stop_times.with_columns(
            pl.when(pl.col("departure_time_hrs")>23).then(
                pl.col("departure_time_hrs")-24).otherwise(
                    pl.col("departure_time_hrs")
            )).group_by(
                [stop_group_id, 'departure_time_hrs']
            ).agg(
                pl.col('departure_time_hrs').count().alias('frequency')
            ).sort(stop_group_id,"departure_time_hrs")
        

        t = df.pivot(values='frequency', index=[stop_group_id], on=[
            'departure_time_hrs'])
        t = t.fill_null(0)

        col_hours = []
        for col in t.collect_schema().names():
            if not col == stop_group_id:
                col_hours.append(int(col))

        col_hours = list(np.sort(col_hours).astype(str))  

        t = t[[stop_group_id] + col_hours]

        for col in col_hours:
                t = t.rename({col: 'hour_' + str(col)})

        return t

    
    def get_lines_gdf(self):
        """
        Returns a GeoDataFrame with records for each rep_trip_id and
        line geomery for the shape_id used by the trip_id. Useful GTFS
        columns inlcude route_id, direction_id, route_type, route_short_name,
        route_long_name, and route_desc.
        """

        rep_trips = self.trips.filter(pl.col('rep_trip_id') == pl.col('trip_id'))
        
        rep_trips = rep_trips.join(self.routes, how='left', on='route_id')
        rep_trips = self.shapes.merge(rep_trips.to_pandas(), how='right', on='shape_id') ### geopolars
        #assert rep_trips.geometry.hasnans==False
        return rep_trips

    def get_line_stops_gdf(self,trip_groups:bool=True):
        """
        Returns a GeoDataFrame with records for each stop for each
        rep_trip_id.
        """
        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'
  
        route_stops = self.stop_times.filter(pl.col('trip_id') == pl.col(trip_group_id)).drop('stop_id')
        route_stops = route_stops.to_pandas().merge(self.stop_groups_gdf, how='left', on='stop_id') ################# geopolars
        route_stops = gpd.GeoDataFrame(route_stops, geometry=route_stops['geometry'],crs=self.crs)      ###################### geopolars

        return route_stops

    def get_line_time(self,trip_groups:bool=True):
        """
        Returns a DataFrame with records for each rep_trip_id
        and their total service time.
        """
        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'
        
        print("Not implemented")
        first = self.stop_times.group_by([trip_group_id]).agg(pl.col('departure_time_secs').min().alias('first'))['first']

        last = self.stop_times.group_by([trip_group_id]).agg(pl.col('departure_time_secs').max().alias('last'))['last']

        route_id = self.stop_times.group_by([trip_group_id]).agg(pl.col(['rep_trip_id', 'route_id']).unique())

        df = route_id.with_columns(first = first, last = last)
        df = df.with_columns(last=pl.when(pl.col('last') < pl.col('first')).then(pl.col('last') + 24*3600).otherwise(pl.col('last')))
        df = df.with_columns(total_line_time = (pl.col('last')-pl.col('first'))/60)
        return df

    def get_service_hours_by_line(self):
        """
        Returns a DataFrame with records for each rep_trip_id and columns with
        the number of service hours for each hour after midnight with service.
        For example 2:00-3:00 AM is called hour_2 and 3:00-4:00 PM is called
        hour_15.
        """
        print("Not implemented")
        df = self.get_line_time()
        df = df.group_by('trip_group_id').agg(pl.col(['total_line_time']).sum())
        df = df.join(self.trips[['route_id', 'rep_trip_id', 'trip_group_id', 'direction_id']], how='left', on='trip_group_id')

        return df[['trip_group_id','rep_trip_id', 'route_id', 'direction_id','total_line_time']]

    def get_routes_by_stops(self,stop_groups:bool=True):
        """
        Returns a DataFrame with records for each rep_trip_id and a column
        holding a list of stops for each line.
        """
        if stop_groups:
            stop_group_id = 'stop_group_id'
        else:
            stop_group_id = 'stop_id'
        
        df = self.stop_times.filter(self.stop_times['trip_id'] == self.stop_times['rep_trip_id'])
        df = df.group_by(stop_group_id).agg(pl.col(['route_id']).unique())
        return df.sort(stop_group_id)

    def get_total_trips_by_line(self,trip_groups:bool=True):
        """
        Returns a DataFrame with records for each rep_trip_id and a column
        holding the total number of trips for each line.
        """

        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'

        df = self.trips.group_by(trip_group_id).agg(pl.all().unique(), pl.col(['trip_id']).count().alias('total_trips'))
        return df[['trip_group_id','rep_trip_id', 'route_id', 'direction_id', 'total_trips']].sort(trip_group_id)
    
    def get_squedule_symetry(self,trip_groups:bool=True):

        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'
    
        df = self.stop_times.group_by('stop_group_id','trip_group_id').agg(
            pl.all(),
            squedule_symetry = pl.when(pl.col('departure_time_secs').n_unique() > 1).then(
                np.floor((pl.col('departure_time_secs').unique() % 3600) / 60).unique_counts().sort().reverse().slice(
                0,
                np.ceil(pl.col('departure_time_secs').n_unique() / (
                    pl.col('departure_time_secs').max() - pl.col('departure_time_secs').min()
                    )).replace({np.inf : 1}).cast(int)
                ).sum() / pl.col('departure_time_secs').n_unique()
            ).otherwise(
                -1
            )
        ).explode(pl.exclude('squedule_symetry','stop_group_id',trip_group_id))
        return df

    def get_frequency(self,start_time:str=None,end_time:str=None,trip_groups:bool=True,stop_groups:bool=True,exclude_first_stop:bool=True):

        if trip_groups:
            trip_group_id = 'trip_group_id'
            route_group_id = 'route_group_id'
        else:
            trip_group_id = 'rep_trip_id'
            route_group_id = 'rep_route_id'

        if stop_groups: 
            stop_group_id = 'stop_group_id'
        else:
            stop_group_id = 'stop_id'


        if not start_time:
            start_time = self.start_time 

        if not end_time:
            end_time = self.end_time

        start = int(start_time[0:2]) * 3600 + int(start_time[3:5]) * 60 + int(start_time[6:8])
        end = int(end_time[0:2]) * 3600 + int(end_time[3:5]) * 60 + int(end_time[6:8])
        if end == 0:
            end = 3600 * 24

        df = self.stop_times.with_columns(
            start_time = pl.when(pl.col('departure_time_secs') <= start).then(pl.col('departure_time_secs') - start).otherwise(-72*3600),
            end_time = pl.when(pl.col('departure_time_secs') >= end).then(end - pl.col('departure_time_secs')).otherwise(-72*3600)
        )
        if exclude_first_stop:
            if (type(self.bounds) == type(None)) or (self.strict_bounds == False):
                df = df.group_by('trip_id').agg(
                    pl.all().sort_by("stop_sequence").slice(pl.when(pl.col('stop_sequence').min() == 1).then(1).otherwise(0),None)
                ).explode(pl.exclude('trip_id'))
            else:
                df = df.group_by('trip_id').agg(
                    pl.all().sort_by("orig_stop_sequence").slice(pl.when(pl.col('orig_stop_sequence').min() == 1).then(1).otherwise(0),None)
                ).explode(pl.exclude('trip_id'))

        df = df.group_by(stop_group_id,trip_group_id).agg(
            pl.col(route_group_id).first(),
            extended_departure_time = pl.col('departure_time_secs').filter((
                pl.col('departure_time_secs') >= pl.col('start_time').max() + start
                    ) & (
                pl.col('departure_time_secs') <= end - pl.col('end_time').max()
            )).unique(),
            departure_time = pl.col('departure_time_secs').filter(
                (pl.col('departure_time_secs') >= start) & (pl.col('departure_time_secs') <= end)
            ).unique(),
        )

        
        df = df.with_columns(
            intervals = pl.col('extended_departure_time').list.diff(null_behavior='drop'),
            start_time = pl.col('extended_departure_time').list.min(),
            end_time = pl.col('extended_departure_time').list.max()
        )

        df = df.with_columns(
            start_time = pl.when(pl.col('start_time') < start).then(pl.col('start_time')).otherwise(start),
            end_time = pl.when(pl.col('end_time') > end).then(pl.col('end_time')).otherwise(end),
        )


        df = df.with_columns(
            frequency = (((
                pl.col('intervals').list.eval(pl.element().pow(2)).list.sum() + (
                pl.col('extended_departure_time').list.min() - pl.col('start_time') + pl.col('end_time') - pl.col('extended_departure_time').list.max()
                )**2) / (
                (pl.col('end_time') - pl.col('start_time')) * 60)
            ) * (
                pl.col('departure_time').list.len() > 0
            ))
        ).drop('extended_departure_time','start_time','end_time')

        return df
    
    def get_capacity(self,start_time:str=None,end_time:str=None,trip_groups:bool=True,stop_groups:bool=True,exclude_first_stop:bool=True):
        if not start_time:
            start_time = self.start_time 

        if not end_time:
            end_time = self.end_time

        df = self.get_frequency(start_time=start_time,end_time=end_time,trip_groups=trip_groups,stop_groups=stop_groups,exclude_first_stop=exclude_first_stop)

        df = df.with_columns(
            capacity = pl.when(pl.col('frequency') > 0).then((60 / pl.col('frequency')).round(2)).otherwise(0)
        )

        return df
    
    def get_capacity_by_line(self,start_time:str=None,end_time:str=None,trip_groups:bool=True,stop_groups:bool=True,agg:str='max',exclude_first_stop:bool=False):
        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'

        if not start_time:
            start_time = self.start_time 

        if not end_time:
            end_time = self.end_time
    
        capacity = self.get_capacity(start_time=start_time,end_time=end_time,trip_groups=trip_groups,stop_groups=stop_groups,exclude_first_stop=exclude_first_stop)

        if agg == 'mean':
            capacity = capacity.group_by(trip_group_id).agg(
                    capacity = pl.col('capacity').mean(),
                )
        elif agg == 'max':
            capacity = capacity.group_by(trip_group_id).agg(
                    capacity = pl.col('capacity').max()
                )
        elif agg == 'min':   
            capacity = capacity.group_by(trip_group_id).agg(
                    capacity = pl.col('capacity').min()
                )
        else:
            raise Exception(f"agg {agg} not implemented")
        
        return capacity
    
    def get_capacity_at_stops(self,start_time:str=None,end_time:str=None,trip_groups:bool=True,stop_groups:bool=True,agg:str='max_sum',agg_factor:float=0.5,exclude_first_stop:bool=True):
        if stop_groups:
            stop_group_id = 'stop_group_id'
        else:
            stop_group_id = 'stop_id'

        if trip_groups:
            route_group_id = 'route_group_id'
        else:
            route_group_id = 'rep_route_id'

        if not start_time:
            start_time = self.start_time 

        if not end_time:
            end_time = self.end_time
    
        capacity_1 = self.get_capacity(start_time=start_time,end_time=end_time,trip_groups=trip_groups,stop_groups=stop_groups,exclude_first_stop=False)
        if agg == 'mean':
            capacity_1 = capacity_1.group_by(stop_group_id).agg(
                    capacity = pl.col('capacity').mean()
                )
        elif agg == 'max':
            capacity_1 = capacity_1.group_by(stop_group_id).agg(
                    capacity = pl.col('capacity').max()
                )
        elif agg == 'min':   
            capacity_1 = capacity_1.group_by(stop_group_id).agg(
                    capacity = pl.col('capacity').min()
                )
        elif agg == 'sum':
            #capacity = capacity.group_by([stop_group_id,route_group_id]).agg(
            #    capacity = pl.col('capacity').max()
            #)
            capacity_1 = capacity_1.group_by(stop_group_id).agg(
                capacity_sum_a = pl.col('capacity').sort(descending=True).slice(0,2),
                capacity_sum_b = pl.col('capacity').sort(descending=True).slice(2,None)
            ).with_columns(
                capacity = pl.col('capacity_sum_a').list.sum() * 0.5 + pl.col('capacity_sum_b').list.sum() * agg_factor * 0.5
            ).drop(['capacity_sum_a','capacity_sum_b']).sort('stop_group_id')
        elif agg == 'max_sum':
            capacity_1 = capacity_1.group_by(stop_group_id).agg(
                capacity_max = pl.col('capacity').max()
            ).sort('stop_group_id')
            capacity_2 = self.get_capacity(start_time=start_time,end_time=end_time,trip_groups=trip_groups,stop_groups=stop_groups,exclude_first_stop=True)
            capacity_2 = capacity_2.group_by(stop_group_id).agg(
                capacity_sum_a = pl.col('capacity').sort(descending=True).slice(0,2),
                capacity_sum_b = pl.col('capacity').sort(descending=True).slice(2,None)
            ).with_columns(
                capacity_sum = pl.col('capacity_sum_a').list.sum() * 0.5 + pl.col('capacity_sum_b').list.sum() * agg_factor * 0.5
            ).drop(['capacity_sum_a','capacity_sum_b']).sort('stop_group_id')
            capacity_1 = capacity_1.join(capacity_2,on='stop_group_id',how='left').fill_null(0)
            capacity_1 = capacity_1.with_columns(
                capacity = pl.when(pl.col('capacity_max') > pl.col('capacity_sum')).then(pl.col('capacity_max')).otherwise(pl.col('capacity_sum'))
            )
        else:
            raise Exception(f"agg {agg} not implemented")
        
        return capacity_1.sort('stop_group_id')

    def get_cph_by_line(self,trip_groups:bool=True,stop_groups:bool=True,agg:str='max',exclude_first_stop:bool=False):
        """
        Returns a DataFrame with records for each rep_trip_id and
        columns with the number of trips for each hour after midnight
        with service. For example 2:00-3:00 AM is called hour_2 and
        3:00-4:00 PM is called hour_15.
        """
        if trip_groups:
            trip_group_id = 'trip_group_id'
        else:
            trip_group_id = 'rep_trip_id'

        df = self.trips[['trip_group_id', 'route_id', 'rep_trip_id', 'direction_id']].group_by(trip_group_id).agg(pl.all().unique())
        
        for i in range(23):
            start_time = (f"0{i}:00:00")[-8:]
            end_time = (f"0{i+1}:00:00")[-8:]
            capacity = self.get_capacity_by_line(start_time=start_time,end_time=end_time,trip_groups=trip_groups,stop_groups=stop_groups,agg=agg,exclude_first_stop=exclude_first_stop)
            capacity = capacity.with_columns(
                capacity = pl.when((pl.col('capacity') > 0) & (pl.col('capacity') < 1)).then(1).otherwise(pl.col('capacity'))
            )
            df = df.with_columns(capacity['capacity'].alias(f'hour_{i}'))

        return df
    
    def get_cph_at_stops(self,trip_groups:bool=True,stop_groups:bool=True,agg:str='max_sum',agg_factor:float=1,exclude_first_stop:bool=True):
        """
        Returns a DataFrame with records for each rep_trip_id and
        columns with the number of trips for each hour after midnight
        with service. For example 2:00-3:00 AM is called hour_2 and
        3:00-4:00 PM is called hour_15.
        """
        if stop_groups:
            stop_group_id = 'stop_group_id'
        else:
            stop_group_id = 'stop_id'

        df = self.stop_times[['stop_group_id','stop_id','trip_group_id', 'route_id', 'rep_trip_id', 'direction_id']].group_by(stop_group_id).agg(pl.all().unique())
        
        for i in range(23):
            start_time = (f"0{i}:00:00")[-8:]
            end_time = (f"0{i+1}:00:00")[-8:]
            capacity = self.get_capacity_at_stops(start_time=start_time,end_time=end_time,trip_groups=trip_groups,stop_groups=stop_groups,agg=agg,agg_factor=agg_factor,exclude_first_stop=exclude_first_stop)
            capacity = capacity.with_columns(
                capacity = pl.when((pl.col('capacity') > 0) & (pl.col('capacity') < 1)).then(1).otherwise(pl.col('capacity'))
            )
            df = df.with_columns(capacity['capacity'].alias(f'hour_{i}'))

        return df
    
    def stop_service_quality(self,frequencies:list,start_time=None,end_time=None,agg='max',agg_factor:float=1,exclude_first_stop:bool=True):
        if type(start_time) == type(None):
            start_time = self.start_time
            
        if type(end_time) == type(None):
            end_time = self.end_time

        capacity = self.get_capacity_at_stops(start_time,end_time,agg=agg,agg_factor=agg_factor,exclude_first_stop=exclude_first_stop)
        capacity = capacity.with_columns(
            frequency = 60/pl.col('capacity'),
            service_quality = len(frequencies)+1
        )
        service_quality = len(frequencies)
        for c in np.sort(frequencies)[::-1]: 
            capacity = capacity.with_columns(
                service_quality = pl.when(pl.col('frequency') < c).then(service_quality).otherwise(pl.col('service_quality')).cast(int)
            )
            service_quality -= 1 

        stops_gdf = self.stops_gdf.copy()
        stops_gdf = stops_gdf.merge(capacity.to_pandas(),on='stop_group_id',how='left')
        return stops_gdf 

"""
def get_service_class_gdf(gtfs_path,city,city_bounds,nap,route_exceptions=None,date='today',
                          service_date='max',start_time='06:00:00',end_time='22:00:00',
                            stop_group_distance=[100,200,300],trip_group_distance=[500,1000,2000], trip_group_overlap = 0.5,
                            class_frequencies=[6.5,11.5,21.5,41.5,66.5,96.5,131.5,261.5],agg='max_sum',agg_factor=0.5):

    if type(date) != list:
        date = [date,date,date]

    service_class_gdf = gpd.GeoDataFrame(columns=['stop_group_id','service_class','transport_type','capacity_max','capacity_sum','capacity','frequency'],geometry=[],crs=4326)
    for i in range(3):
        if i == 0:
            transport_type = ['bus']
            search_transport_type = ['bus']
            transport_type_int = 2
        elif i == 1:
            transport_type = ['brt','tram']
            search_transport_type = ['bus','train']
            transport_type_int = 1
        else:
            transport_type = ['train']
            search_transport_type = ['train']
            transport_type_int = 0


        date_i = date[i]
        if date_i == 'today':
            date_i = datetime.now().strftime("%d-%m-%Y")

        print(transport_type)

        files = nap.find_files(region=city,region_type="municipio",transport_type=search_transport_type,file_type='gtfs',start_date=date_i,end_date=date_i)
        file_names = nap.download_file(file_ids=files,output_path=gtfs_path)

        if type(route_exceptions) != type(None):
            route_filter = route_exceptions.filter(pl.col('search_transport_type').is_in(search_transport_type))
            route_filter = route_filter.with_columns(
                    function = pl.when(pl.col('transport_type').is_in(transport_type)).then(pl.lit('in')).otherwise(pl.lit('not in'))
            )
        else: 
            route_filter = None

        try:
            gtfs = GTFS(gtfs_dir=file_names,service_date=service_date,start_time=start_time,end_time=end_time,bounds=city_bounds,
                    strict_bounds=False,stop_group_distance=stop_group_distance[i],trip_group_distance=trip_group_distance[i],trip_group_overlap=trip_group_overlap,
                    correct_stop_sequence=True,crs=4326,route_filter=route_filter)
            
        except Exception as e:
            print(f"An exeception occured. Skipping: {e}")
            continue

        stop_quality = gtfs.stop_quality_class(class_frequencies=class_frequencies,agg=agg,agg_factor=agg_factor).drop_duplicates(['stop_group_id']).to_crs(4326)
        transport_type_str = ""
        transport_type_str= ""
        for j in transport_type:
            transport_type_str += f"{j} "

        transport_type_str = transport_type_str[0:-1]

        stop_quality.loc[:,'transport_type'] = transport_type_str
        stop_quality = stop_quality.loc[:,['stop_group_id','service_class','transport_type','capacity_max','capacity_sum','capacity','frequency','geometry']]
        stop_quality.loc[:,'service_class'] = stop_quality['service_class'] + transport_type_int
        stop_quality = stop_quality.loc[stop_quality['service_class'].isna()==False]
        stop_quality['service_class'] = stop_quality['service_class'].astype(int)
        if len(service_class_gdf) == 0:
            service_class_gdf = stop_quality
        else:
            service_class_gdf = pd.concat([service_class_gdf,stop_quality])

    return service_class_gdf
"""
        
