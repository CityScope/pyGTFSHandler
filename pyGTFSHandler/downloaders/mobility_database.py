import os
import json
import random
import zipfile
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import itertools 

import requests
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm import tqdm

from .. import utils
# Configure logging for better output control
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_city_geometry(city_name: str) -> gpd.GeoDataFrame:
    """
    Download city boundary geometry from OpenStreetMap.

    Parameters
    ----------
    city_name : str
        Name of the city (e.g., "Berlin, Germany").

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the city boundary polygon in EPSG:4326.
    """
    # Query OSM for place boundary
    gdf = ox.geocode_to_gdf(city_name)

    # Ensure CRS is WGS84
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def get_geographic_suggestions_from_string(
    query: str,
    user_agent: str = "MobilityDatabaseClient",
    max_results: int = 25
) -> Dict[str, List[str]]:
    """
    Suggests all possible country codes, subdivisions, and municipalities
    for a given string using OpenStreetMap's Nominatim service.
    
    This version collects all relevant fields without skipping any.
    Counties are always included in municipalities.
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=10)

    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    try:
        locations = geolocator.geocode(
            query,
            addressdetails=True,
            language='en',
            exactly_one=False,
            limit=max_results
        )
        if locations:
            for location in locations:
                address = location.raw.get('address', {})

                # Country code
                country_code = address.get('country_code')
                if country_code:
                    suggested_country_codes.add(country_code.upper())

                # Collect all possible subdivisions
                for key in ['state', 'province', 'region', 'county']:
                    value = address.get(key)
                    if value:
                        suggested_subdivision_names.add(value)

                # Collect all possible municipalities
                for key in ['city', 'town', 'village', 'county']:
                    value = address.get(key)
                    if value:
                        suggested_municipalities.add(value)

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return {
        'country_codes': sorted(suggested_country_codes),
        'subdivision_names': sorted(suggested_subdivision_names),
        'municipalities': sorted(suggested_municipalities),
    }

def get_geographic_suggestions_from_aoi(
    aoi: Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries],
    num_points: int = 1, # Number of points to sample for geocoding
    user_agent: str = "MobilityDatabaseClient" # Required by Nominatim
) -> Dict[str, List[str]]:
    """
    Suggests country codes, subdivisions, and municipalities for a given
    Area of Interest (AOI) by performing reverse geocoding.
    This simplified version generates sample points within the AOI's *bounding box*.

    This function uses OpenStreetMap's Nominatim service via geopy.
    Please be respectful of Nominatim's usage policy (one request per second).
    It's recommended to set a specific `user_agent` for your application.

    Args:
        aoi: An Area of Interest, which can be a shapely.geometry.Polygon,
             shapely.geometry.MultiPolygon, geopandas.GeoDataFrame, or
             geopandas.GeoSeries.
        num_points: The number of random points to sample within the AOI's
                    bounding box for reverse geocoding. More points can provide broader
                    coverage for large AOIs, but increases the number of Nominatim requests.
                    Defaults to 1 (representative point).
        user_agent: A unique user agent string for Nominatim requests. This is required.

    Returns:
        A dictionary containing lists of suggested 'country_codes',
        'subdivision_names', and 'municipalities'. Example:
        {
            'country_codes': ['US', 'CA'],
            'subdivision_names': ['California', 'Québec'],
            'municipalities': ['Los Angeles', 'Montreal']
        }
        Returns lists that might be empty if geocoding fails or no relevant info is found.

    Raises:
        ImportError: If geopandas or geopy are not installed.
        TypeError: If `aoi` is not a supported geospatial object.
        ValueError: If the AOI is empty or invalid.
    """
    if gpd is None or Nominatim is None or random is None:
        raise ImportError("geopandas, geopy, and random must be available to use get_geographic_suggestions_from_aoi. Please run 'pip install geopandas geopy'.")

    target_geometry: Union[Polygon, MultiPolygon]
    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if aoi.empty:
            raise ValueError("Provided GeoDataFrame/GeoSeries is empty.")
        target_geometry = aoi.to_crs(4326).union_all() # Combine all geometries
    elif isinstance(aoi, (Polygon, MultiPolygon)):
        target_geometry = aoi
    else:
        raise TypeError("aoi must be a shapely.geometry.Polygon, MultiPolygon, geopandas.GeoDataFrame, or geopandas.GeoSeries object.")

    if target_geometry.is_empty:
        raise ValueError("AOI geometry is empty.")

    geolocator = Nominatim(user_agent=user_agent, timeout=10)

    suggested_country_codes = set()
    suggested_subdivision_names = set()
    suggested_municipalities = set()

    points_to_geocode: List[Point] = []
    if num_points <= 0:
        num_points = 1 # Ensure at least one point is sampled

    # Calculate bounding box once
    min_lon, min_lat, max_lon, max_lat = target_geometry.bounds

    if num_points == 1:
        # For a single point, representative_point is often more meaningful than bbox center
        points_to_geocode.append(target_geometry.representative_point())
    else:
        # Generate random points within the bounding box
        for _ in range(num_points):
            rand_lon = random.uniform(min_lon, max_lon)
            rand_lat = random.uniform(min_lat, max_lat)
            points_to_geocode.append(Point(rand_lon, rand_lat))

    for i, point in enumerate(points_to_geocode):
        lat, lon = point.y, point.x
        logger.debug(f"Geocoding point {i+1}/{len(points_to_geocode)}: ({lat}, {lon}) (sampled from bbox)")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location and location.raw:
                address = location.raw.get('address', {})
                
                # Country code (ISO 3166-1 alpha-2)
                country_code_long = address.get('country_code')
                if country_code_long:
                    suggested_country_codes.add(country_code_long.upper())

                # Subdivision name (state, province, region, etc.)
                # Ordered by common usage/specificity
                subdivision = address.get('state') or address.get('province') or address.get('region') or address.get('county')
                if subdivision:
                    suggested_subdivision_names.add(subdivision)

                # Municipality (city, town, village)
                municipality = address.get('city') or address.get('town') or address.get('village')
                if municipality:
                    suggested_municipalities.add(municipality)
            else:
                logger.warning(f"No location data found for point ({lat}, {lon}).")
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding failed for point ({lat}, {lon}): {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during geocoding for point ({lat}, {lon}): {e}")

    return {
        'country_codes': sorted(list(suggested_country_codes)), # Sort for consistent output
        'subdivision_names': sorted(list(suggested_subdivision_names)),
        'municipalities': sorted(list(suggested_municipalities))
    }

class MobilityDatabaseClient:
    """
    A client for interacting with the Mobility Database API.

    Handles OAuth2 authentication with refresh tokens and provides methods for
    searching and downloading GTFS feeds.
    """
    BASE_URL = "https://api.mobilitydatabase.org/v1"
    TOKEN_ENDPOINT = f"{BASE_URL}/tokens"
    GTFS_FEEDS_ENDPOINT = f"{BASE_URL}/gtfs_feeds"

    def __init__(self, refresh_token: str):
        """
        Initializes the MobilityDatabaseClient and obtains an initial access token.

        Args:
            refresh_token: Your long-lived refresh token for the Mobility Database API.
                           (You'll need to obtain this from the Mobility Database website,
                           as indicated by "Your refresh token is hidden" in the documentation.)
        """
        if not refresh_token:
            raise ValueError("Refresh token cannot be empty.")
        self.refresh_token = refresh_token
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._get_access_token() # Acquire initial token

    def _get_access_token(self) -> None:
        """
        Obtains or refreshes an access token.
        Tokens are valid for one hour, so this method refreshes it if it's
        expired or close to expiration (within a 5-minute buffer).
        """
        # Refresh token if it's expired or will expire in less than 5 minutes
        if self._access_token and self._token_expires_at and \
           datetime.now() < self._token_expires_at - timedelta(minutes=5):
            logger.debug("Access token is still valid.")
            return

        logger.info("Obtaining/refreshing access token...")
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "refresh_token": self.refresh_token
        }
        try:
            response = requests.post(self.TOKEN_ENDPOINT, headers=headers, data=json.dumps(data))
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            token_data = response.json()
            self._access_token = token_data.get('access_token')
            if not self._access_token:
                raise ValueError("Access token not found in response.")

            # Access tokens are valid for one hour (3600 seconds) as per documentation.
            # 'expires_in' field might be present in the response for a specific duration.
            expires_in_seconds = token_data.get('expires_in', 3600)
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
            logger.info("Access token obtained/refreshed successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obtaining access token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}, content: {e.response.text}")
            raise ConnectionError(f"Failed to obtain access token: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid token response: {e}")
            raise

    def _authorized_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Makes an HTTP request with the Authorization header.
        Ensures the access token is fresh before making the request.

        Args:
            method: The HTTP method (e.g., "GET", "POST").
            url: The URL for the request.
            **kwargs: Additional keyword arguments to pass to requests.request.

        Returns:
            The requests.Response object.

        Raises:
            ConnectionError: If unable to get a valid access token.
            requests.exceptions.RequestException: For HTTP errors during the request.
        """
        self._get_access_token() # Ensure token is fresh before any request
        
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {self._access_token}'
        
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Request URL: {e.response.request.url}")
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            raise

    def _prepare_list_param_for_api(
        self,
        param_name: str,
        value: Optional[Union[str, List[Optional[str]]]]
    ) -> Tuple[Optional[List[str]], bool]:
        """
        Process list-based query parameters.
        Returns a list of strings and a flag if None was included.
        """
        if value is None:
            return None, False

        if isinstance(value, str):
            return [value], False

        if isinstance(value, list):
            has_none_in_list = False
            valid_string_items = []
            for item in value:
                if item is None:
                    has_none_in_list = True
                elif isinstance(item, str):
                    valid_string_items.append(item)
                else:
                    raise TypeError(f"All items in the list for '{param_name}' must be strings or None, got {type(item)}")
            return valid_string_items or None, has_none_in_list

        raise TypeError(f"Expected string, list of strings (potentially including None), or None for '{param_name}', got {type(value)}")

    def search_gtfs_feeds(
        self,
        aoi: Optional[Union[Polygon, MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        provider: Optional[Union[str, List[Optional[str]]]] = None,
        producer_url: Optional[Union[str, List[Optional[str]]]] = None,
        country_code: Optional[Union[str, List[Optional[str]]]] = None,
        subdivision_name: Optional[Union[str, List[Optional[str]]]] = None,
        municipality: Optional[Union[str, List[Optional[str]]]] = None,
        bounding_filter_method: str = "partially_enclosed",
        is_official: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for GTFS feeds. Supports lists of values per parameter
        by making separate API calls for each combination of values.
        Handles None values client-side.
        """
        # Prepare parameters
        param_requests = {
            'provider': self._prepare_list_param_for_api('provider', provider),
            'producer_url': self._prepare_list_param_for_api('producer_url', producer_url),
            'country_code': self._prepare_list_param_for_api('country_code', country_code),
            'subdivision_name': self._prepare_list_param_for_api('subdivision_name', subdivision_name),
            'municipality': self._prepare_list_param_for_api('municipality', municipality),
        }

        needs_none_filter_pass = any(include_none_flag for _, include_none_flag in param_requests.values())

        # Base parameters
        base_params: Dict[str, Any] = {}
        if limit is not None:
            if not (0 <= limit <= 2500):
                logger.warning(f"Limit {limit} is outside typical range [0,2500].")
            base_params['limit'] = limit
        if offset is not None:
            if offset < 0:
                raise ValueError("Offset cannot be negative.")
            base_params['offset'] = offset
        if is_official is not None:
            base_params['is_official'] = str(is_official).lower()

        # AOI bounding box
        if aoi:
            if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
                if aoi.empty:
                    raise ValueError("AOI is empty.")
                bounds_geom = aoi.geometry.to_crs(4326).union_all()
                if bounds_geom.is_empty:
                    raise ValueError("AOI geometry is empty after union.")
            elif isinstance(aoi, (Polygon, MultiPolygon)):
                bounds_geom = aoi
            else:
                raise TypeError("Invalid AOI type.")

            min_lon, min_lat, max_lon, max_lat = bounds_geom.bounds
            base_params['dataset_latitudes'] = f"{min_lat},{max_lat}"
            base_params['dataset_longitudes'] = f"{min_lon},{max_lon}"

            valid_methods = ["completely_enclosed", "partially_enclosed", "disjoint"]
            if bounding_filter_method not in valid_methods:
                raise ValueError(f"Invalid bounding_filter_method: {bounding_filter_method}")
            base_params['bounding_filter_method'] = bounding_filter_method

        all_results_set: Set[str] = set()
        final_results: List[Dict[str, Any]] = []

        # --- Generate all combinations of non-None values ---
        non_none_values = {k: v[0] for k, v in param_requests.items() if v[0]}  # dict of lists
        keys, lists = zip(*non_none_values.items()) if non_none_values else ([], [])

        for combination in itertools.product(*lists):
            query_params = {**base_params, **dict(zip(keys, combination))}
            logger.info(f"API Call for combination: {query_params}")
            try:
                results = self._authorized_request("GET", self.GTFS_FEEDS_ENDPOINT, params=query_params).json()
            except Exception as e:
                logger.warning(f"API call failed for {query_params}: {e}")
                continue

            for feed in results:
                feed_id = feed.get('id')
                if feed_id and feed_id not in all_results_set:
                    final_results.append(feed)
                    all_results_set.add(feed_id)

        # --- Second API Call for None fields if needed ---
        if needs_none_filter_pass:
            # Build query excluding parameters requesting None
            omitted_values = {k: v[0] for k, v in param_requests.items() if v[0] and not v[1]}
            keys_omit, lists_omit = zip(*omitted_values.items()) if omitted_values else ([], [])
            for combination in itertools.product(*lists_omit) if lists_omit else [()]:
                query_params = {**base_params, **dict(zip(keys_omit, combination))}
                logger.info(f"API Call for None fields: {query_params}")
                try:
                    results = self._authorized_request("GET", self.GTFS_FEEDS_ENDPOINT, params=query_params).json()
                except Exception as e:
                    logger.warning(f"API call failed for {query_params}: {e}")
                    continue

                for feed in results:
                    feed_id = feed.get('id')
                    if feed_id in all_results_set:
                        continue

                    matches_none_criteria = True
                    for param_name, (_, include_none_flag) in param_requests.items():
                        if include_none_flag:
                            field_value = None
                            if param_name in ['provider', 'producer_url']:
                                field_value = feed.get(param_name)
                            elif param_name in ['country_code', 'subdivision_name', 'municipality']:
                                locations = feed.get('locations')
                                if locations and isinstance(locations, list) and len(locations) > 0:
                                    field_value = locations[0].get(param_name)
                            if field_value is not None:
                                matches_none_criteria = False
                                break

                    if matches_none_criteria:
                        final_results.append(feed)
                        all_results_set.add(feed_id)

        logger.info(f"Total unique feeds found: {len(final_results)}")
        return final_results

    def download_feeds(self,feeds: List[Dict], download_folder: str, overwrite: bool = False) -> List[str]:
        os.makedirs(download_folder, exist_ok=True)
        extracted_paths = []

        for feed in tqdm(feeds, desc="Downloading feeds"):
            feed_id = feed.get('id', '')
            feed_name = feed.get('feed_name', '')
            feed_provider = feed.get('provider', '')
            # Ensure each component is at most x characters
            max_chars = 10
            feed_id_short = feed_id[:max_chars]
            feed_name_short = feed_name[:max_chars]
            feed_provider_short = feed_provider[:max_chars]

            # Combine and normalize
            feed_filename = utils.normalize_string(f"{feed_id_short}_{feed_name_short}_{feed_provider_short}")

            latest_dataset = feed.get('latest_dataset')
            if not latest_dataset:
                print(f"Feed '{feed_filename}' has no 'latest_dataset'. Skipping.")
                continue

            hosted_url = latest_dataset.get('hosted_url')
            if not hosted_url:
                print(f"Feed '{feed_filename}' has no 'hosted_url'. Skipping.")
                continue

            extraction_folder = os.path.join(download_folder, feed_filename)
            if os.path.exists(extraction_folder) and os.listdir(extraction_folder) and not overwrite:
                print(f"Feed '{feed_filename}' already exists. Skipping.")
                extracted_paths.append(os.path.abspath(extraction_folder))
                continue

            if os.path.exists(extraction_folder) and overwrite:
                import shutil
                shutil.rmtree(extraction_folder)

            zip_path = os.path.join(download_folder, f"{feed_filename}.zip")
            try:
                # Public download — no Authorization header
                with requests.get(hosted_url, stream=True) as r:
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                # Extract the ZIP
                os.makedirs(extraction_folder, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_folder)

                #print(f"Extracted '{feed_filename}' to '{extraction_folder}'.")
                extracted_paths.append(os.path.abspath(extraction_folder))

            except requests.exceptions.RequestException as e:
                print(f"Error downloading feed '{feed_filename}': {e}")
            except (OSError, zipfile.BadZipFile) as e:
                print(f"Error extracting feed '{feed_filename}': {e}")
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)

        return extracted_paths
