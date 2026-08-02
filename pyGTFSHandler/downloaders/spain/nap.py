# -*- coding: utf-8 -*-
"""Downloader client for Spain's National Access Point (NAP).

Spain's NAP (https://nap.transportes.gob.es) is the Spanish transport
ministry's catalog of public transport datasets, including GTFS feeds
published by regional and municipal operators. Its REST API
(https://nap.transportes.gob.es/api) is organized around a handful of
lookup/reference endpoints (regions, transport types, file types,
organizations) plus a filtering endpoint (`/Fichero/Filter`) that returns
datasets ("conjuntos de datos") matching those ids.

`NAPDownloader.search_feeds` mirrors that two-step flow: free-text filter
values (region, transport type, organization names) are first resolved to
NAP's internal numeric ids via fuzzy name matching (since the API itself
only filters by id), and the resulting ids are used to query
`/Fichero/Filter`. Each dataset can bundle several files (`ficherosDto`),
each downloadable individually; `search_feeds` returns one
`GTFSFeedMetadata` per file. Because NAP's download links require the
`ApiKey` header for authentication, `NAPDownloader.download_feeds`
overrides the base implementation to pass that header through to the
shared `downloaders.utils.download.download_feeds` helper.

Beyond the common `search_feeds`/`download_feeds` interface, this module
also exposes `download_historic_stack`, a NAP-specific workflow that
downloads a dataset's full publication history within a date range and
stitches the successive GTFS publications into one continuous feed via
`pyGTFSHandler.utils.stack_gtfs.historic_stack` (trimming each
publication's `calendar.txt`/`calendar_dates.txt` so consecutive
publications don't overlap). This has no equivalent in the other
downloaders, since NAP is, to our knowledge, the only supported source
that exposes a per-dataset publication history.
"""

import logging
import os
import shutil
import warnings
import zipfile
from copy import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
from rapidfuzz import fuzz, process

from ...utils import hashing
from ...utils.stack_gtfs import historic_stack
from ..base import BaseGTFSDownloader
from ..utils.download import download_feeds
from ..utils.models import GTFSFeedMetadata
from ..utils.naming import normalize_text, sanitize_filename
from .utils import input_date, process_calendar, process_calendar_dates

logger = logging.getLogger(__name__)

_REGION_TYPE_ALIASES = {
    "Provincia": {"0", "provincia", "province", "region"},
    "ComunidadAutonoma": {"1", "comunidadautonoma", "state"},
    "AreaUrbana": {"2", "ciudad", "areaurbana", "city", "urbanarea"},
    "Municipio": {"3", "municipio", "municipality"},
}

_TRANSPORT_TYPE_ALIASES = {
    "autobus": {"bus", "autobus"},
    "ferroviario": {"tren", "ferrocarril", "rail", "train"},
    "maritimo": {"barco", "boat", "ferry"},
    "aereo": {"avion", "plane", "air", "aereo"},
}


def _resolve_alias(value: Union[int, str], aliases: Dict[str, set]) -> str:
    """Resolve a user-supplied id/name/alias to a canonical NAP category name.

    Args:
        value: An integer index, canonical name, or known alias.
        aliases: Mapping of canonical name to the set of accepted aliases
            (lowercase, unaccented) for it. The set also contains the
            integer index of the category (as a string) when relevant.

    Returns:
        The canonical category name.

    Raises:
        ValueError: If `value` doesn't match any known category or alias.
    """
    normalized = normalize_text(value).replace(" ", "")
    for canonical, alias_set in aliases.items():
        if normalized in {a.replace(" ", "") for a in alias_set}:
            return canonical
    raise ValueError(f"'{value}' is not a recognized value among {list(aliases)}.")


class NAPDownloader(BaseGTFSDownloader):
    """Client for searching and downloading feeds from Spain's NAP.

    Requires a NAP API key, available after registering at
    https://nap.transportes.gob.es.
    """

    BASE_URL = "https://nap.transportes.gob.es/api"
    API_KEY_ENV_VAR = "NAP_API_KEY"
    SOURCE_NAME = "nap_es"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the client.

        Args:
            api_key: NAP API key. If not provided, it is read from the
                `NAP_API_KEY` environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        super().__init__(api_key=api_key)
        if not self.api_key:
            raise ValueError(
                "A NAP API key is required (pass api_key or set "
                f"{self.API_KEY_ENV_VAR})."
            )
        self.headers = {"ApiKey": self.api_key, "accept": "application/json"}

    def _get(self, path: str) -> requests.Response:
        """Issue an authenticated GET request against the NAP API.

        Args:
            path: Endpoint path relative to `BASE_URL`.

        Returns:
            The raw `requests.Response`.
        """
        return requests.get(f"{self.BASE_URL}{path}", headers=self.headers)

    # -------------------------------------------------------------------
    # Name -> id resolution (the NAP API only filters by numeric id)
    # -------------------------------------------------------------------

    def get_region_id(self, name: str, region_type: Union[int, str] = 3) -> Optional[int]:
        """Resolve a region name to its NAP region id via fuzzy matching.

        Args:
            name: Region name to look up (e.g. a municipality name).
            region_type: Which kind of region to search among: a NAP
                category name/alias (e.g. "municipio", "province") or its
                integer index (0=Provincia, 1=ComunidadAutonoma,
                2=AreaUrbana, 3=Municipio).

        Returns:
            The matched region's NAP id, or `None` if no region of that
            type was found or none matched closely enough.
        """
        region_type = _resolve_alias(region_type, _REGION_TYPE_ALIASES)
        name = normalize_text(name)

        response = self._get("/Region")
        if response.status_code != 200:
            logger.error(f"Error fetching regions: {response.status_code} - {response.text}")
            return None

        regions = [
            r for r in response.json() if normalize_text(r.get("tipoNombre")) == normalize_text(region_type)
        ]
        if not regions:
            logger.warning(f"No regions found of type '{region_type}'.")
            return None

        region_names = [normalize_text(r["nombre"]) for r in regions]
        match = process.extractOne(name, region_names, scorer=fuzz.token_sort_ratio)
        if not match:
            logger.warning(f"No close match found for {region_type} '{name}'.")
            return None

        _, _, index = match
        return regions[index]["regionId"]

    def get_transport_type_id(self, transport_name: str) -> Optional[int]:
        """Resolve a transport mode name to its NAP transport type id.

        Args:
            transport_name: Transport mode name or alias (e.g. "bus",
                "train", "ferry").

        Returns:
            The matched transport type's NAP id, or `None` if not found.
        """
        try:
            canonical = _resolve_alias(transport_name, _TRANSPORT_TYPE_ALIASES)
        except ValueError:
            canonical = normalize_text(transport_name)

        response = self._get("/TipoTransporte")
        if response.status_code != 200:
            logger.error(f"Error fetching transport types: {response.status_code}")
            return None

        for transport in response.json():
            if normalize_text(transport["nombre"]) == canonical:
                return transport["tipoTransporteId"]

        logger.warning(f"Transport type '{transport_name}' not found.")
        return None

    def get_file_type_id(self, file_type: str = "GTFS") -> Optional[int]:
        """Resolve a file format name (e.g. "GTFS") to its NAP file type id.

        Args:
            file_type: File format name, e.g. "GTFS" or "NeTEx".

        Returns:
            The matched file type's NAP id, or `None` if not found.
        """
        file_type = normalize_text(file_type)
        response = self._get("/TipoFichero")
        if response.status_code != 200:
            logger.error(f"Error fetching file types: {response.status_code}")
            return None

        for f in response.json():
            if normalize_text(f["nombre"]) == file_type:
                return f["tipoFicheroId"]

        logger.warning(f"File type '{file_type}' not found.")
        return None

    def get_organization_id(self, organization_name: str) -> Optional[int]:
        """Resolve an organization/operator name to its NAP organization id.

        Args:
            organization_name: Organization name to look up.

        Returns:
            The matched organization's NAP id, or `None` if not found.
        """
        organization_name = normalize_text(organization_name)
        response = self._get(f"/Organizacion/GetByName/{organization_name}")
        if response.status_code != 200:
            logger.error(f"Error fetching organizations: {response.status_code} - {response.text}")
            return None

        organizations = response.json()
        org_names = [normalize_text(o["nombre"]) for o in organizations]
        match = process.extractOne(organization_name, org_names, scorer=fuzz.token_sort_ratio)
        if not match:
            logger.warning(f"No close match found for organization '{organization_name}'.")
            return None

        _, _, index = match
        return organizations[index]["organizacionId"]

    # -------------------------------------------------------------------
    # Search feeds
    # -------------------------------------------------------------------

    def find_datasets(
        self,
        region: Union[int, str, List[Union[int, str]]] = (),
        transport_type: Union[int, str, List[Union[int, str]]] = (),
        organization: Union[int, str, List[Union[int, str]]] = (),
        file_type: Union[int, str] = "GTFS",
        region_type: Union[int, str, List[Union[int, str]]] = 3,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        file_description: Union[str, List[str]] = (),
        keep: str = "newest",
    ) -> List[Dict[str, Any]]:
        """Find NAP datasets ("conjuntos de datos") matching filters.

        Args:
            region: Region name(s) or NAP region id(s) to filter by.
            transport_type: Transport mode name(s) or NAP transport type
                id(s) to filter by.
            organization: Organization name(s) or NAP organization id(s)
                to filter by.
            file_type: File format name or NAP file type id. Defaults to
                "GTFS".
            region_type: Region category name(s)/id(s) that `region`
                values are resolved against (see `get_region_id`).
            start_date: If set (with `end_date`), only keep datasets with
                a published file whose validity interval covers this
                range. Accepts "today", `date`/`datetime`, or
                `"%d-%m-%Y"`/`"%d%m%Y"` strings.
            end_date: End of the validity range; see `start_date`.
            file_description: Substring(s) to match against each
                dataset's name/description.
            keep: When filtering by date and a dataset has multiple
                matching files, which to keep: "newest", "oldest", or
                "all".

        Returns:
            A list of raw NAP dataset dictionaries (each with a
            `ficherosDto` list of downloadable files).
        """
        start_date, end_date = input_date(start_date, end_date)

        organization = organization if isinstance(organization, list) else [organization]
        region_type = region_type if isinstance(region_type, list) else [region_type]
        region = region if isinstance(region, list) else [region]
        transport_type = transport_type if isinstance(transport_type, list) else [transport_type]
        file_description = (
            file_description if isinstance(file_description, list) else [file_description]
        )

        organization_ids = [
            self.get_organization_id(o) if isinstance(o, str) else o for o in organization
        ]
        region_ids = [
            self.get_region_id(r, rt) if isinstance(r, str) else r
            for r in region
            for rt in region_type
        ]
        transport_type_ids = [
            self.get_transport_type_id(t) if isinstance(t, str) else t for t in transport_type
        ]
        file_type_id = self.get_file_type_id(file_type) if isinstance(file_type, str) else file_type

        payload = {
            "provincias": region_ids,
            "comunidades": region_ids,
            "areasurbanas": region_ids,
            "municipios": region_ids,
            "tipotransportes": transport_type_ids,
            "tipoficheros": [file_type_id],
            "organizaciones": organization_ids,
        }
        response = requests.post(
            f"{self.BASE_URL}/Fichero/Filter",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
        )
        if response.status_code != 200:
            logger.error(f"Error filtering datasets: {response.status_code} - {response.text}")
            return []

        result = response.json()
        if result.get("filesNum", 0) <= 0:
            return []
        datasets = result["conjuntosDatoDto"]

        if start_date is not None:
            datasets = self._filter_by_dates(datasets, start_date, end_date, keep=keep)

        if file_description:
            needles = [normalize_text(d) for d in file_description]
            datasets = [
                d
                for d in datasets
                if any(
                    n in normalize_text(d["descripcion"]) or n in normalize_text(d["nombre"])
                    for n in needles
                )
            ]

        return datasets

    @staticmethod
    def _filter_by_dates(
        datasets: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
        keep: str = "newest",
    ) -> List[Dict[str, Any]]:
        """Keep only each dataset's file(s) valid over `[start_date, end_date]`.

        Args:
            datasets: Raw NAP dataset dictionaries.
            start_date: Start of the required validity range.
            end_date: End of the required validity range.
            keep: "newest", "oldest", or "all" matching files to keep per
                dataset.

        Returns:
            The datasets that have at least one matching file, with
            `ficherosDto` replaced by only the matching file(s).
        """
        filtered = []
        for dataset in datasets:
            matching = []
            best_upload_date = None
            for file_info in dataset["ficherosDto"]:
                file_start = datetime.strptime(file_info["fechaDesde"], "%Y-%m-%dT%H:%M:%S")
                file_end = datetime.strptime(file_info["fechaHasta"], "%Y-%m-%dT%H:%M:%S")
                if not (file_start <= start_date and file_end >= end_date):
                    continue

                upload_date = datetime.strptime(file_info["fechaActualizacion"], "%Y-%m-%dT%H:%M:%S")
                if keep == "all":
                    matching.append(file_info)
                elif best_upload_date is None:
                    matching, best_upload_date = [file_info], upload_date
                elif keep == "newest" and upload_date > best_upload_date:
                    matching, best_upload_date = [file_info], upload_date
                elif keep == "oldest" and upload_date < best_upload_date:
                    matching, best_upload_date = [file_info], upload_date
                elif keep not in ("newest", "oldest", "all"):
                    raise ValueError(f"Invalid 'keep' value: {keep}")

            if matching:
                dataset = {**dataset, "ficherosDto": matching}
                filtered.append(dataset)

        return filtered

    def search_feeds(
        self,
        region: Union[int, str, List[Union[int, str]]] = (),
        transport_type: Union[int, str, List[Union[int, str]]] = (),
        organization: Union[int, str, List[Union[int, str]]] = (),
        file_type: Union[int, str] = "GTFS",
        region_type: Union[int, str, List[Union[int, str]]] = 3,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        file_description: Union[str, List[str]] = (),
        keep: str = "newest",
    ) -> List[GTFSFeedMetadata]:
        """Search NAP for GTFS feeds matching filters.

        See `find_datasets` for the filter arguments. Each NAP dataset can
        bundle several downloadable files; this returns one
        `GTFSFeedMetadata` per file.

        Returns:
            A list of `GTFSFeedMetadata`, one per matching file.
        """
        datasets = self.find_datasets(
            region=region,
            transport_type=transport_type,
            organization=organization,
            file_type=file_type,
            region_type=region_type,
            start_date=start_date,
            end_date=end_date,
            file_description=file_description,
            keep=keep,
        )

        feeds: List[GTFSFeedMetadata] = []
        for dataset in datasets:
            main_name = dataset.get("nombre", "")
            for i, file_info in enumerate(dataset.get("ficherosDto", [])):
                suffix = f"_{i + 1}" if i > 0 else ""
                feeds.append(
                    GTFSFeedMetadata(
                        id=str(file_info["ficheroId"]),
                        download_url=f"{self.BASE_URL}/Fichero/download/{file_info['ficheroId']}",
                        name=f"{main_name}{suffix}",
                        provider=None,
                        country_code="ES",
                        source=self.SOURCE_NAME,
                        raw={"dataset": dataset, "file": file_info},
                    )
                )
        return feeds

    def download_feeds(
        self,
        feeds: List[GTFSFeedMetadata],
        download_folder: str,
        overwrite: bool = False,
        unzip: bool = True,
    ) -> List[str]:
        """Download previously found NAP feeds to a local folder.

        Overrides the base implementation to send the `ApiKey` header
        required by NAP's download endpoint.

        Args:
            feeds: Feeds to download, as returned by `search_feeds()`.
            download_folder: Directory to store (and, if `unzip`,
                extract) the downloaded feeds into.
            overwrite: If True, re-download and replace files that
                already exist on disk.
            unzip: If True, extract each ZIP after downloading and delete
                the ZIP file.

        Returns:
            Absolute paths to the downloaded feeds.
        """
        return download_feeds(
            feeds, download_folder, overwrite=overwrite, unzip=unzip, headers=self.headers
        )

    # -------------------------------------------------------------------
    # Historic-publication stitching (NAP-specific)
    # -------------------------------------------------------------------

    def download_historic_stack(
        self,
        output_path: str,
        datasets: Union[Dict[str, Any], List[Dict[str, Any]]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        day_separation: int = 1,
        overwrite: bool = False,
        aoi=None,
    ) -> List[str]:
        """Download and stitch a dataset's publication history into one feed.

        For each dataset, downloads every historic publication whose date
        falls within `[start_date, end_date]` (plus the publication
        immediately preceding it, to cover the start of the range), trims
        each publication's `calendar.txt`/`calendar_dates.txt` so
        consecutive publications don't overlap, and merges them via
        `pyGTFSHandler.utils.stack_gtfs.historic_stack`.

        Args:
            output_path: Directory to assemble the stitched feed(s) in.
            datasets: One or more raw NAP dataset dictionaries, as
                returned by `find_datasets`.
            start_date: Start of the date range to cover. Accepts
                "today", `date`/`datetime`, or `"%d-%m-%Y"`/`"%d%m%Y"`
                strings.
            end_date: End of the date range to cover.
            day_separation: Minimum number of days a publication is
                assumed to stay valid for, if service periods don't force
                it shorter.
            overwrite: If True, redo datasets whose stitched output
                already exists.
            aoi: Optional AOI passed through to
                `pyGTFSHandler.utils.stack_gtfs.historic_stack` to restrict
                stops.

        Returns:
            Paths to each dataset's stitched feed folder.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        os.makedirs(output_path, exist_ok=True)
        start_date, end_date = input_date(start_date, end_date)
        datasets = copy(datasets)

        for dataset in datasets:
            dataset_id = dataset["conjuntoDatoId"]
            name = normalize_text(dataset["nombre"])
            response = self._get(f"/Fichero/historico/{dataset_id}")
            if response.status_code != 200:
                warnings.warn(f"No historic results for dataset {dataset_id} {name}")

            history = response.json()
            first_entry, first_entry_date = None, None
            historic_entries = []
            for entry in history:
                entry_date = datetime.strptime(entry["fecha"], "%Y-%m-%dT%H:%M:%S")
                if entry_date < start_date and (
                    first_entry_date is None or first_entry_date < entry_date
                ):
                    first_entry_date, first_entry = entry_date, entry
                if start_date <= entry_date <= end_date:
                    historic_entries.append(entry)

            if first_entry is not None:
                historic_entries.append(first_entry)
            dataset["historic_entries"] = historic_entries

        all_paths = []
        for dataset in datasets:
            main_name = sanitize_filename(dataset["nombre"])
            main_path = os.path.normpath(os.path.join(output_path, main_name))
            if not overwrite and os.path.isdir(main_path):
                all_paths.append(main_path)
                logger.info(f"Dataset '{main_path}' already exists. Skipping.")
                continue

            entries = sorted(
                dataset["historic_entries"],
                key=lambda e: datetime.strptime(e["fecha"], "%Y-%m-%dT%H:%M:%S"),
            )
            if not entries:
                continue

            path_stack = self._download_and_stitch_history(
                entries, main_name, main_path, day_separation, end_date, overwrite
            )
            if not path_stack:
                continue

            if os.path.isfile(path_stack[-1] + ".zip"):
                os.remove(path_stack[-1] + ".zip")

            historic_stack(path_stack, main_path, aoi)
            logger.info(f"Finished stitching historic dataset '{main_name}'.")
            for f in path_stack:
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)

            all_paths.append(main_path)

        return all_paths

    def _download_and_stitch_history(
        self,
        entries: List[Dict[str, Any]],
        main_name: str,
        main_path: str,
        day_separation: int,
        end_date: datetime,
        overwrite: bool,
    ) -> List[str]:
        """Download each historic entry and trim calendars against the next one.

        Args:
            entries: Historic publication entries, sorted chronologically.
            main_name: Sanitized dataset name, used to build file paths.
            main_path: Base output path for this dataset.
            day_separation: See `download_historic_stack`.
            end_date: End of the requested date range.
            overwrite: If True, re-download entries that already exist.

        Returns:
            Paths (without extension) to each downloaded/extracted entry,
            in the order they should be stitched.
        """
        dates = [datetime.strptime(e["fecha"], "%Y-%m-%dT%H:%M:%S") for e in entries]
        path_stack: List[str] = []
        i = 0
        while i is not None:
            entry = entries[i]
            file_date = dates[i]
            file_path = os.path.normpath(f"{main_path}_start_date_{file_date.strftime('%Y%m%d')}")

            if os.path.isdir(file_path) and not overwrite:
                logger.info(f"File '{file_path}' already exists. Skipping download.")
            else:
                url = f"{self.BASE_URL}/Fichero/{entry['link']}"
                response = requests.get(url, headers=self.headers, stream=True)
                if response.status_code != 200:
                    warnings.warn(
                        f"Error downloading entry {entry['id']}: {response.status_code} - {response.text}"
                    )
                    i += 1
                    continue

                with open(file_path + ".zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if path_stack and os.path.isfile(path_stack[-1] + ".zip"):
                    if hashing.compare_paths(path_stack[-1] + ".zip", file_path + ".zip"):
                        os.remove(file_path + ".zip")
                        logger.info(f"Entry '{file_path}' is identical to the previous one. Skipping.")
                        i += 1
                        continue
                    os.remove(path_stack[-1] + ".zip")

                os.makedirs(file_path, exist_ok=True)
                with zipfile.ZipFile(file_path + ".zip", "r") as zip_ref:
                    zip_ref.extractall(file_path)

            next_index, min_end_date = self._trim_calendars(file_path, file_date, dates, day_separation)

            path_stack.append(file_path)
            if next_index is None or next_index <= i:
                next_index = i + 1
            elif dates[next_index] >= end_date or (min_end_date is not None and min_end_date >= end_date):
                break

            if next_index >= len(entries):
                break

            i = next_index

        return path_stack

    @staticmethod
    def _trim_calendars(
        file_path: str,
        file_date: datetime,
        dates: List[datetime],
        day_separation: int,
    ) -> tuple:
        """Trim `calendar.txt`/`calendar_dates.txt` for one historic entry.

        Args:
            file_path: Path to the extracted GTFS folder.
            file_date: Publication date of this entry.
            dates: Publication dates of all entries, chronologically.
            day_separation: See `download_historic_stack`.

        Returns:
            A tuple `(next_index, min_end_date)`, as returned by
            `pyGTFSHandler.downloaders.spain.utils.process_calendar`/
            `process_calendar_dates`.
        """
        calendar_path = os.path.normpath(os.path.join(file_path, "calendar.txt"))
        calendar_dates_path = os.path.normpath(os.path.join(file_path, "calendar_dates.txt"))

        next_index, min_end_date = None, None
        if os.path.isfile(calendar_path):
            next_index, min_end_date = process_calendar(
                calendar_path, file_date, dates, day_separation, calendar_path
            )

        if os.path.isfile(calendar_dates_path):
            next_index, min_end_date = process_calendar_dates(
                calendar_dates_path,
                file_date,
                dates,
                day_separation,
                calendar_dates_path,
                next_index=next_index,
                has_calendar=os.path.isfile(calendar_path),
            )

        if not os.path.isfile(calendar_path) and not os.path.isfile(calendar_dates_path):
            warnings.warn(f"File '{file_path}' has no calendar.txt or calendar_dates.txt")

        return next_index, min_end_date
