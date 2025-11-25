import os
import json
import itertools
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Set

import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm

from .. import gtfs_checker

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MobilityDatabaseClient:
    """
    A client for interacting with the Mobility Database API.

    Handles OAuth2 authentication via a refresh token and provides
    methods to search for and download GTFS feeds.
    """

    BASE_URL = "https://api.mobilitydatabase.org/v1"
    TOKEN_ENDPOINT = f"{BASE_URL}/tokens"
    GTFS_FEEDS_ENDPOINT = f"{BASE_URL}/gtfs_feeds"

    # -------------------------------------------------------------------
    # Initialization and Authentication
    # -------------------------------------------------------------------

    def __init__(self, refresh_token: str):
        """
        Initialize the Mobility Database client.

        Args:
            refresh_token: Long-lived refresh token obtained from
                the Mobility Database website.

        Raises:
            ValueError: If refresh_token is empty.
        """
        if not refresh_token:
            raise ValueError("Refresh token cannot be empty.")

        self.refresh_token = refresh_token
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # Get initial access token
        self._get_access_token()

    def _get_access_token(self) -> None:
        """
        Obtain or refresh the access token using the refresh token.

        Tokens typically expire after one hour. This method automatically
        refreshes them when necessary.
        """
        # Skip refresh if still valid for >5 minutes
        if self._access_token and self._token_expires_at and \
           datetime.now() < self._token_expires_at - timedelta(minutes=5):
            logger.debug("Access token is still valid; skipping refresh.")
            return

        logger.info("Obtaining or refreshing access token...")
        headers = {"Content-Type": "application/json"}
        data = {"refresh_token": self.refresh_token}

        try:
            response = requests.post(self.TOKEN_ENDPOINT, headers=headers, data=json.dumps(data))
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data.get("access_token")
            if not self._access_token:
                raise ValueError("Access token missing in response.")

            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            logger.info("Access token obtained successfully.")
        except requests.exceptions.RequestException as e:
            logger.exception("Failed to obtain access token.")
            raise ConnectionError(f"Failed to obtain access token: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid token response: {e}")
            raise

    # -------------------------------------------------------------------
    # HTTP Request Handling
    # -------------------------------------------------------------------

    def _authorized_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make an authorized HTTP request with a valid access token.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target endpoint URL.
            **kwargs: Additional arguments for requests.request().

        Returns:
            requests.Response: The HTTP response.

        Raises:
            requests.exceptions.RequestException: On network or HTTP errors.
        """
        self._get_access_token()  # Refresh token if needed

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            if e.response is not None:
                logger.error(
                    f"Response {e.response.status_code}: {e.response.text}"
                )
            raise

    # -------------------------------------------------------------------
    # Parameter Preparation
    # -------------------------------------------------------------------

    def _prepare_list_param_for_api(
        self,
        param_name: str,
        value: Optional[Union[str, List[Optional[str]]]]
    ) -> Tuple[Optional[List[str]], bool]:
        """
        Prepare list-based query parameters for API requests.

        Args:
            param_name: Parameter name (for error messages).
            value: Single string, list of strings, or list including None.

        Returns:
            Tuple[list_of_valid_strings or None, bool flag_if_None_included]
        """
        if value is None:
            return None, False

        if isinstance(value, str):
            return [value], False

        if isinstance(value, list):
            valid_items = []
            has_none = False
            for item in value:
                if item is None:
                    has_none = True
                elif isinstance(item, str):
                    valid_items.append(item)
                else:
                    raise TypeError(
                        f"All items in '{param_name}' must be strings or None; got {type(item)}"
                    )
            return valid_items or None, has_none

        raise TypeError(
            f"Expected string, list[str|None], or None for '{param_name}', got {type(value)}"
        )

    # -------------------------------------------------------------------
    # Search Feeds
    # -------------------------------------------------------------------

    def _search_gtfs_feeds(
        self,
        aoi=None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        provider=None,
        producer_url=None,
        country_code=None,
        subdivision_name=None,
        municipality=None,
        bounding_filter_method="partially_enclosed",
        is_official: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a single search request (no pagination),
        including param combination logic and None-handling.
        """

        # Prepare parameter combinations
        param_requests = {
            "provider": self._prepare_list_param_for_api("provider", provider),
            "producer_url": self._prepare_list_param_for_api("producer_url", producer_url),
            "country_code": self._prepare_list_param_for_api("country_code", country_code),
            "subdivision_name": self._prepare_list_param_for_api("subdivision_name", subdivision_name),
            "municipality": self._prepare_list_param_for_api("municipality", municipality),
        }

        needs_none_pass = any(flag for _, flag in param_requests.values())

        base_params: Dict[str, Any] = {}

        if limit is not None:
            base_params["limit"] = str(limit)

        if offset is not None:
            if offset < 0:
                raise ValueError("Offset cannot be negative.")
            base_params["offset"] = str(offset)

        if is_official is not None:
            base_params["is_official"] = str(is_official).lower()

        # AOI bounding setup
        if aoi is not None:
            if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
                if aoi.empty:
                    raise ValueError("AOI is empty.")
                bounds_geom = aoi.geometry.to_crs(4326).union_all()
            elif isinstance(aoi, (Polygon, MultiPolygon)):
                bounds_geom = aoi
            else:
                raise TypeError("AOI must be Polygon, MultiPolygon, GeoDataFrame, or GeoSeries.")

            min_lon, min_lat, max_lon, max_lat = bounds_geom.bounds
            base_params["dataset_latitudes"] = f"{min_lat},{max_lat}"
            base_params["dataset_longitudes"] = f"{min_lon},{max_lon}"

            valid_methods = ["completely_enclosed", "partially_enclosed", "disjoint"]
            if bounding_filter_method not in valid_methods:
                raise ValueError(f"Invalid bounding_filter_method: {bounding_filter_method}")

            base_params["bounding_filter_method"] = bounding_filter_method

        # Prepare combinations
        non_none_values = {k: v[0] for k, v in param_requests.items() if v[0]}
        keys, lists = zip(*non_none_values.items()) if non_none_values else ([], [])

        all_feed_ids = set()
        results_final: List[Dict[str, Any]] = []

        # Regular combinations
        for combo in itertools.product(*lists):
            params = {**base_params, **dict(zip(keys, combo))}
            try:
                resp = self._authorized_request("GET", self.GTFS_FEEDS_ENDPOINT, params=params)
                data = resp.json()
            except Exception as e:
                logger.warning(f"API request failed for params {params}: {e}")
                continue

            for feed in data:
                feed_id = feed.get("id")
                if feed_id and feed_id not in all_feed_ids:
                    results_final.append(feed)
                    all_feed_ids.add(feed_id)

        # Handle None fields
        if needs_none_pass:
            omitted_values = {k: v[0] for k, v in param_requests.items() if v[0] and not v[1]}
            keys_o, lists_o = zip(*omitted_values.items()) if omitted_values else ([], [])

            for combo in itertools.product(*lists_o) if lists_o else [()]:
                params = {**base_params, **dict(zip(keys_o, combo))}
                try:
                    resp = self._authorized_request("GET", self.GTFS_FEEDS_ENDPOINT, params=params)
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"API request failed for params {params}: {e}")
                    continue

                for feed in data:
                    feed_id = feed.get("id")
                    if not feed_id or feed_id in all_feed_ids:
                        continue

                    matches_none = True
                    for param_name, (_, include_none_flag) in param_requests.items():
                        if include_none_flag:
                            field_value = None

                            if param_name in ["provider", "producer_url"]:
                                field_value = feed.get(param_name)
                            else:
                                locs = feed.get("locations")
                                if locs:
                                    field_value = locs[0].get(param_name)

                            if field_value is not None:
                                matches_none = False
                                break

                    if matches_none:
                        results_final.append(feed)
                        all_feed_ids.add(feed_id)

        return results_final

    def search_gtfs_feeds(
        self,
        *args,
        limit: Optional[int] = 100_000,
        offset: Optional[int] = None,
        **kwargs
    ):
        """
        Search for GTFS feeds matching specified filters.

        Supports multiple values per parameter and handles client-side
        merging of results across combinations.

        If limit <= 200 → single call.
        If limit > 200 → automatic pagination using _search_gtfs_feeds().

        Args:
            aoi: Polygon or GeoDataFrame defining a geographic area.
            limit: Number of feeds to return (max 2500; values >200 trigger internal pagination).
            offset: Offset for pagination.
            provider, producer_url, country_code, subdivision_name, municipality:
                Filter fields (accept a list or single value).
            bounding_filter_method: Spatial inclusion rule. One of:
                'completely_enclosed', 'partially_enclosed', 'disjoint'.
            is_official: Whether to filter by official feeds. None, True, False

        Returns:
            A list of feed metadata dictionaries.
        """

        MAX_PAGE = 200

        if limit is None:
            limit = MAX_PAGE

        if limit <= MAX_PAGE:
            return self._search_gtfs_feeds(
                *args, limit=limit, offset=offset, **kwargs
            )

        # Pagination mode
        final_results = []
        used_ids = set()

        remaining = limit
        current_offset = offset if offset is not None else 0

        while remaining > 0:
            page_limit = min(MAX_PAGE, remaining)

            page_results = self._search_gtfs_feeds(
                *args,
                limit=page_limit,
                offset=current_offset,
                **kwargs
            )

            if not page_results:
                break

            for feed in page_results:
                fid = feed.get("id")
                if fid and fid not in used_ids:
                    final_results.append(feed)
                    used_ids.add(fid)

            remaining -= page_limit
            current_offset += page_limit

            if len(page_results) < page_limit:
                # No more results from API
                break

        return final_results

    # -------------------------------------------------------------------
    # Download Feeds
    # -------------------------------------------------------------------

    def download_feeds(
        self,
        feeds: List[Dict],
        download_folder: str,
        overwrite: bool = False
    ) -> List[str]:
        """
        Download GTFS feed ZIP files from the Mobility Database.

        Args:
            feeds: List of feed dictionaries (from search_gtfs_feeds()).
            download_folder: Directory to store the ZIP files.
            overwrite: If True, re-download and replace existing ZIPs.

        Returns:
            List of absolute paths to downloaded ZIP files.
        """
        os.makedirs(download_folder, exist_ok=True)
        zip_paths: List[str] = []

        overwrite_message = True 

        for feed in tqdm(feeds, desc="Downloading feeds"):
            feed_id = feed.get("id", "")
            feed_name = feed.get("feed_name", "")
            feed_provider = feed.get("provider", "")

            # Normalize and truncate filenames
            max_chars = 10
            feed_id_short = feed_id[:max_chars]
            feed_name_short = feed_name[:max_chars]
            feed_provider_short = feed_provider[:max_chars]
            feed_filename = gtfs_checker.normalize_string(
                f"{feed_id_short}_{feed_name_short}_{feed_provider_short}"
            )

            zip_path = os.path.join(download_folder, f"{feed_filename}.zip")

            latest_dataset = feed.get("latest_dataset")
            if not latest_dataset:
                logger.warning(f"Feed '{feed_filename}' has no 'latest_dataset'. Skipping.")
                continue

            hosted_url = latest_dataset.get("hosted_url")
            if not hosted_url:
                logger.warning(f"Feed '{feed_filename}' has no 'hosted_url'. Skipping.")
                continue

            # Skip or overwrite existing files
            if os.path.exists(zip_path):
                if overwrite:
                    if overwrite_message:
                        logger.info(f"Overwriting all existing feeds.")
                        overwrite_message = False
                else:
                    if overwrite_message:
                        logger.info(f"Skipping all already donwloaded feeds.")
                        overwrite_message = False
                        
                    zip_paths.append(os.path.abspath(zip_path))
                    continue

            try:
                #logger.info(f"Downloading feed '{feed_filename}' from {hosted_url}")
                with requests.get(hosted_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                #logger.info(f"Downloaded '{feed_filename}' to {zip_path}")
                zip_paths.append(os.path.abspath(zip_path))

            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading '{feed_filename}': {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                continue

        logger.info(f"Successfully downloaded {len(zip_paths)} feeds.")
        return zip_paths
