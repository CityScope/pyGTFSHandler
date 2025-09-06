import requests
import difflib
from typing import Union
from datetime import datetime, timedelta, date
import unicodedata
import os
import zipfile
import re
from copy import copy
from rapidfuzz import process, fuzz
import warnings
import csv
import pandas as pd
from ... import utils
from ... import stack_gtfs
import shutil


def input_date(start_date, end_date):
    if end_date is None:
        end_date = start_date
    elif start_date is None:
        start_date = end_date

    if (start_date is None) and (end_date is None):
        return None, None

    if start_date is not None:
        if start_date == "today":
            start_date = datetime.now().strftime("%d-%m-%Y")

        if end_date == "today":
            end_date = datetime.now().strftime("%d-%m-%Y")

        if isinstance(start_date, date):
            start_date = datetime.combine(start_date, datetime.min.time())

        if isinstance(end_date, date):
            end_date = datetime.combine(end_date, datetime.min.time())

        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%d%m%Y")
            except Exception as _:
                start_date = datetime.strptime(start_date, "%d-%m-%Y")

        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%d%m%Y")
            except Exception as _:
                end_date = datetime.strptime(end_date, "%d-%m-%Y")

    return start_date, end_date


def normalize_text(text):
    """Lowercase + strip accents from a string"""
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def sanitize_filename(name: str) -> str:
    """Replaces spaces and invalid filename characters with an underscore."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", normalize_text(name))


def process_calendar(path, date, possible_dates, day_separation, calendar_path):
    with open(path, "r") as f:
        sample = f.read(1024)  # read a chunk of the file
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    df = pd.read_csv(path, dtype=str, sep=sep)  # keep everything as string first
    df = df.rename(columns={c: utils.normalize_string(c) for c in df.columns})
    df["start_date"] = df["start_date"].astype(int)
    df["end_date"] = df["end_date"].astype(int)
    date_int = int(date.strftime("%Y%m%d"))
    df["start_date"] = df["start_date"].apply(lambda x: min(x, date_int))

    if len(possible_dates) == 0:
        return None

    min_end_date = pd.to_datetime(df["end_date"].astype(str), format="%Y%m%d").min()
    min_end_date = min(date + timedelta(days=day_separation), min_end_date)
    min_end_date = min_end_date + timedelta(days=1)

    id = None
    for i in range(len(possible_dates)):
        if possible_dates[i] <= min_end_date:
            id = i
        else:
            break

    if id is not None:
        end_date = possible_dates[id]
        end_date_int = int((end_date - timedelta(days=1)).strftime("%Y%m%d"))
        if end_date > date:
            df["end_date"] = df["end_date"].apply(lambda x: min(x, end_date_int))
            df.to_csv(calendar_path, index=False, sep=sep)
            return id, min_end_date
        else:
            df["end_date"] = df["end_date"].apply(lambda x: min(x, date_int))
            df.to_csv(calendar_path, index=False, sep=sep)
            return id, min_end_date

    df = df[df["start_date"] <= df["end_date"]]
    df.to_csv(calendar_path, index=False, sep=sep)
    return None, min_end_date


def process_calendar_dates(
    path,
    file_date,
    possible_dates,
    day_separation,
    calendar_dates_path,
    id=None,
    has_calendar: bool = True,
):
    with open(path, "r") as f:
        sample = f.read(1024)  # read a chunk of the file
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    df = pd.read_csv(path, dtype=str, sep=sep)  # keep everything as string first
    df = df.rename(columns={c: utils.normalize_string(c) for c in df.columns})
    df["date"] = df["date"].astype(int)
    date_int = int(file_date.strftime("%Y%m%d"))
    df = df[df["date"] >= date_int]
    if id is not None:
        end_date = possible_dates[id]
        end_date_int = int((end_date - timedelta(days=1)).strftime("%Y%m%d"))
        df = df[df["date"] <= end_date_int]
        df.to_csv(calendar_dates_path, index=False, sep=sep)
        return id, end_date

    df["exception_type"] = df["exception_type"].astype(int)
    dates = pd.to_datetime(
        df.loc[df["exception_type"] == 1, "date"].astype(str), format="%Y%m%d"
    )
    if len(dates) == 0:
        if not has_calendar:
            warnings.warn(
                f"File {path.replace('calendar_dates.txt', '')} does not have calendar.txt and calendar_dates.txt does not have any exception_type 1 so there are no dates with any trips."
            )

        if (id is not None) and (id < (len(possible_dates) - 1)):
            return id + 1, None
        else:
            return None, None

    counts = dates.value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index
    min_end_date = candidates.max()
    min_end_date = min(file_date + timedelta(days=day_separation), min_end_date)
    min_end_date = min_end_date + timedelta(days=1)

    id = None
    for i in range(len(possible_dates)):
        if possible_dates[i] <= min_end_date:
            id = i
        else:
            break

    if id is not None:
        end_date = possible_dates[id]
        end_date_int = int((end_date - timedelta(days=1)).strftime("%Y%m%d"))
        if end_date > file_date:
            df = df[df["date"] <= end_date_int]
            df.to_csv(calendar_dates_path, index=False, sep=sep)
            return id, min_end_date
        else:
            df = df[df["date"] <= date_int]
            df.to_csv(calendar_dates_path, index=False, sep=sep)
            return id, min_end_date

    df.to_csv(calendar_dates_path, index=False, sep=sep)
    return None, min_end_date


class APIClient:
    BASE_URL = "https://nap.transportes.gob.es/api"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NAP_API_KEY", "")
        self.headers = {
            "ApiKey": self.api_key,
            "accept": "application/json",
        }

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "ApiKey": self.api_key,
            "accept": "application/json",
        }

    def get_headers(self):
        return self.headers

    def get_region_id(self, name: str, region_type: Union[int, str] = 3):
        """Gets the region ID of the region by matching the closest name."""
        if (
            (region_type == 0)
            or (normalize_text(region_type) == "provincia")
            or (normalize_text(region_type) == "province")
            or (normalize_text(region_type) == "region")
        ):
            region_type = "Provincia"
        elif (
            (region_type == 1)
            or (normalize_text(region_type).replace(" ", "") == "comunidadautonoma")
            or (normalize_text(region_type) == "state")
        ):
            region_type = "ComunidadAutonoma"
        elif (
            (region_type == 2)
            or (normalize_text(region_type) == "ciudad")
            or (normalize_text(region_type).replace(" ", "") == "areaurbana")
            or (normalize_text(region_type) == "city")
            or (normalize_text(region_type).replace(" ", "") == "urbanarea")
        ):
            region_type = "AreaUrbana"
        elif (
            (region_type == 3)
            or (normalize_text(region_type) == "municipio")
            or (normalize_text(region_type) == "municipality")
        ):
            region_type = "Municipio"
        else:
            raise Exception(f"region type {region_type} not valid.")

        name = normalize_text(name)
        region_type = normalize_text(region_type)

        url = f"{self.BASE_URL}/Region"  # /GetByName/{name}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            regions = response.json()

            # Filter for regions that are municipalities based on 'tipoNombre'
            regions = [
                i for i in regions if normalize_text(i.get("tipoNombre")) == region_type
            ]

            if regions:
                # Extract the names of the municipalities
                region_names = [normalize_text(i["nombre"]) for i in regions]

                closest_match, score, index = process.extractOne(
                    name, region_names, scorer=fuzz.token_sort_ratio
                )

                print(f"Closest match for {region_type} {name}: {closest_match}")

                if closest_match:
                    return regions[index]["regionId"]

                else:
                    print(f"No close match found for {region_type} '{name}'.")
            else:
                print(
                    f"No {region_type} found with the region_type (tipoNombre) '{name}'."
                )

        return None

    def get_transport_type_id(self, transport_name: str):
        """Obtains the ID of the transport type."""
        transport_name = normalize_text(transport_name)
        if (transport_name == "bus") or (transport_name == "autobus"):
            transport_name = "autobus"
        elif (
            (transport_name == "tren")
            or (transport_name == "ferrocarril")
            or (transport_name == "rail")
            or (transport_name == "train")
        ):
            transport_name = "ferroviario"
        elif (
            (transport_name == "barco")
            or (transport_name == "boat")
            or (transport_name == "ferry")
        ):
            transport_name = "maritimo"
        elif (
            (transport_name == "avion")
            or (transport_name == "plane")
            or (transport_name == "air")
            or (transport_name == "aereo")
        ):
            transport_name = "aereo"

        url = f"{self.BASE_URL}/TipoTransporte"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            transport_types = response.json()
            for transport in transport_types:
                if (
                    normalize_text(transport["nombre"]) == transport_name
                ):  # .lower() == transport_name.lower():
                    return transport["tipoTransporteId"]
        print(f"Transport type '{transport_name}' not found.")
        return None

    def get_file_type_id(self, file_type: str = "GTFS"):
        """Obtains the ID of the file type, defaulting to 'GTFS'."""
        file_type = normalize_text(file_type)
        url = f"{self.BASE_URL}/TipoFichero"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            file_types = response.json()
            for f in file_types:
                if (
                    normalize_text(f["nombre"]) == file_type
                ):  # .lower() == file_name.lower():
                    return f["tipoFicheroId"]
        print(f"File type '{file_type}' not found.")
        return None

    def get_organization_id(self, organization_name: str):
        """Obtains the ID of the organization by its name."""
        organization_name = normalize_text(organization_name)
        url = f"{self.BASE_URL}/Organizacion/GetByName/{organization_name}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            organizations = response.json()

            # Extract the names of all organizations
            org_names = [normalize_text(org["nombre"]) for org in organizations]

            # Find the closest matching name to the input 'organization_name'
            closest_match = difflib.get_close_matches(organization_name, org_names, n=1)

            if closest_match:
                closest_match = normalize_text(closest_match[0])
                # Get the organization that matches the closest name
                selected_org = next(
                    org
                    for org in organizations
                    if normalize_text(org["nombre"]) == closest_match
                )
                return selected_org[
                    "organizacionId"
                ]  # Return the ID of the selected organization
            else:
                print(f"No close match found for organization '{organization_name}'.")
        else:
            print(
                f"Error fetching organizations: {response.status_code} - {response.text}"
            )

        return None

    def get_file_id(self, file_name: Union[str, list]):
        """Obtains the file ID by the dataset (conjunto de datos) name."""
        _file_name = copy(file_name)
        if isinstance(_file_name, str):
            _file_name = [_file_name]

        for i in range(len(_file_name)):
            if isinstance(_file_name[i], str):
                raise Exception(
                    f"file name should be str or list[str] but got an element {i} has type {type(_file_name[i])}."
                )

            _file_name[i] = sanitize_filename(
                _file_name[i]
            )  # Normalize the input dataset name

        url = f"{self.BASE_URL}/Fichero/GetList"  # Assuming dataset name is passed here
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            datasets = (
                response.json()
            )  # Parse the list of datasets (conjuntos de datos)

            # Extract the names of all datasets
            dataset_names = [
                sanitize_filename(dataset["nombre"])
                for dataset in datasets["conjuntosDatoDto"]
            ]

            file_ids = []
            for name in _file_name:
                # Find the closest matching dataset name to the input 'conjunto_dato_name'
                closest_match = difflib.get_close_matches(name, dataset_names, n=1)

                if closest_match:
                    closest_match = sanitize_filename(closest_match[0])
                    # Get the dataset that matches the closest name
                    selected_dataset = next(
                        dataset
                        for dataset in datasets["conjuntosDatoDto"]
                        if sanitize_filename(dataset["nombre"]) == closest_match
                    )
                    file_ids.append(
                        selected_dataset["conjuntoDatoId"]
                    )  # Return the file ID ('conjuntoDatoId') of the selected dataset
                else:
                    print(f"No close match found for dataset '{name}'.")

            return file_ids
        else:
            print(f"Error fetching datasets: {response.status_code} - {response.text}")

        return None

    def get_file_metadata(self, file_id: Union[str, int]):
        if isinstance(file_id, str):
            file_id = self.get_file_id(file_id)
            if len(file_id) == 0:
                return []
            else:
                file_id = file_id[0]

        url = (
            f"{self.BASE_URL}/Fichero/{file_id}"  # Assuming dataset name is passed here
        )
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            metadata = (
                response.json()
            )  # Parse the list of datasets (conjuntos de datos)
            return metadata  # ['conjuntosDatoDto']
        else:
            print(f"Error fetching metadata: {response.status_code} - {response.text}")

        return None

    def find_files(
        self,
        region: Union[int, str, list] = [],
        transport_type: Union[int, str, list] = [],
        organization: Union[int, str, list] = [],
        file_type: Union[int, str] = "GTFS",
        region_type: Union[int, str, list[int], list[str]] = 3,
        start_date: str = None,
        end_date: str = None,
        file_description: Union[str, list] = [],
        metadata: bool = True,
        keep: str = "newest",
    ):
        """Filters and obtains the list of files by municipality, transport type, and file type.
        Filtering by date could not work if metadata=False."""

        start_date, end_date = input_date(start_date, end_date)

        _organization = copy(organization)
        if isinstance(_organization, list):
            _organization = [_organization]

        _region_type = copy(region_type)
        if isinstance(_region_type, list):
            _region_type = [_region_type]

        _region = copy(region)
        if isinstance(_region, list):
            _region = [_region]

        _transport_type = copy(transport_type)
        if isinstance(_transport_type, list):
            _transport_type = [_transport_type]

        _file_description = copy(file_description)
        if isinstance(_file_description, list):
            _file_description = [_file_description]

        for i in range(len(_organization)):
            if isinstance(_organization[i], str):
                _organization[i] = self.get_organization_id(_organization[i])

        new_region = []
        for i in range(len(_region)):
            for j in range(len(_region_type)):
                if isinstance(_region[i], str):
                    new_region.append(self.get_region_id(_region[i], _region_type[j]))
                else:
                    new_region.append(_region[i])

        _region = new_region

        for i in range(len(_transport_type)):
            if isinstance(_transport_type[i], str):
                _transport_type[i] = self.get_transport_type_id(_transport_type[i])

        if isinstance(file_type, str):
            file_type = self.get_file_type_id(file_type)

        url = f"{self.BASE_URL}/Fichero/Filter"

        data = {
            "provincias": _region,
            "comunidades": _region,
            "areasurbanas": _region,
            "municipios": _region,
            "tipotransportes": _transport_type,
            "tipoficheros": [file_type],
            "organizaciones": _organization,
        }
        response = requests.post(
            url, headers={**self.headers, "Content-Type": "application/json"}, json=data
        )
        if response.status_code == 200:
            files = response.json()

            if files["filesNum"] > 0:
                files = files["conjuntosDatoDto"]
            else:
                return []

            if start_date is not None:
                files = self.filter_by_dates(files, start_date, end_date, keep=keep)

            if len(_file_description) > 0:
                for i in range(len(_file_description)):
                    _file_description[i] = normalize_text(_file_description[i])

                new_files = []
                for i in range(len(files)):
                    data = normalize_text(files[i]["descripcion"])
                    name = normalize_text(files[i]["nombre"])
                    contains_descr = False
                    for desc in _file_description:
                        if (desc in data) or (desc in name):
                            contains_descr = True
                            break

                    if contains_descr:
                        new_files.append(files[i])

                files = new_files

            if metadata:
                return files
            else:
                file_ids = []
                for i in range(len(files)):
                    file_ids.append(files[i]["conjuntoDatoId"])

                return file_ids

        print("Error filtering files:", response.status_code, response.text)
        return None

    def filter_by_dates(self, files, start_date, end_date, keep="newest"):
        start_date, end_date = input_date(start_date, end_date)

        filtered_files = []
        for file in files:
            new_data = []
            upload_date = None
            for data in file["ficherosDto"]:
                file_start_date = datetime.strptime(
                    data["fechaDesde"], "%Y-%m-%dT%H:%M:%S"
                )
                file_end_date = datetime.strptime(
                    data["fechaHasta"], "%Y-%m-%dT%H:%M:%S"
                )

                if file_start_date <= start_date and file_end_date >= end_date:
                    new_upload_date = datetime.strptime(
                        data["fechaActualizacion"], "%Y-%m-%dT%H:%M:%S"
                    )
                    if upload_date:
                        if keep == "newest":
                            if upload_date < new_upload_date:
                                new_data = [data]
                                upload_date = new_upload_date
                        elif keep == "oldest":
                            if upload_date > new_upload_date:
                                new_data = [data]
                                upload_date = new_upload_date
                        elif keep == "all":
                            new_data.append(data)
                        else:
                            raise Exception(f"keep key {keep} not valid")
                    else:
                        new_data = [data]
                        upload_date = new_upload_date

            if len(new_data) > 0:
                file["ficherosDto"] = new_data
                filtered_files.append(file)

        return filtered_files

    def download_historic(
        self,
        output_path,
        files,
        start_date,
        end_date,
        day_separation=1,
        overwrite=False,
        aoi=None,
    ):
        if not isinstance(files, list):
            files = [files]

        os.makedirs(output_path, exist_ok=True)
        start_date, end_date = input_date(start_date, end_date)

        for i in range(len(files)):
            historic_files = []
            file_id = files[i]["conjuntoDatoId"]
            name = normalize_text(files[i]["nombre"])
            url = f"{self.BASE_URL}/Fichero/historico/{file_id}"
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                warnings.warn(f"No historic results for file {file_id} {name}")

            response_data = response.json()
            first_file = None
            first_file_date = None
            for data in response_data:
                file_date = datetime.strptime(data["fecha"], "%Y-%m-%dT%H:%M:%S")
                if file_date < start_date:
                    if first_file_date is None:
                        first_file_date = file_date
                        first_file = data
                    elif first_file_date < file_date:
                        first_file_date = file_date
                        first_file = data

                if (file_date >= start_date) & (file_date <= end_date):
                    historic_files.append(data)

            if first_file is not None:
                historic_files.append(first_file)

            files[i]["historic_files"] = historic_files

        all_paths = []
        for i in range(len(files)):
            file = files[i]
            main_name = sanitize_filename(file["nombre"])
            main_path = os.path.normpath(output_path + "/" + main_name)
            if (not overwrite) and os.path.isdir(main_path):
                all_paths.append(main_path)
                print(f"File {main_path} already exists. Skipping.")
                continue

            file_ids = []
            links = []
            dates = []
            for j in range(len(file["historic_files"])):
                historic_file = file["historic_files"][j]
                file_id = historic_file["id"]
                file_url = historic_file["link"]
                file_date = datetime.strptime(
                    historic_file["fecha"], "%Y-%m-%dT%H:%M:%S"
                )
                file_ids.append(file_id)
                links.append(file_url)
                dates.append(file_date)

            if len(links) == 0:
                continue

            sorted_data = sorted(zip(dates, file_ids, links), key=lambda x: x[0])
            dates, file_ids, links = zip(*sorted_data)

            j = 0
            path_stack = []
            while j is not None:
                file_id = file_ids[j]
                file_url = links[j]
                file_date = dates[j]
                file_name = main_name + "_start_date_" + file_date.strftime("%Y%m%d")
                file_path = os.path.normpath(
                    main_path + "_start_date_" + file_date.strftime("%Y%m%d")
                )

                if os.path.isdir(file_path) and (not overwrite):
                    print(f"File {file_path} already exists. Skipping.")
                else:
                    url = f"{self.BASE_URL}/Fichero/{file_url}"
                    response = requests.get(url, headers=self.headers, stream=True)
                    if response.status_code == 200:
                        with open(file_path + ".zip", "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024):
                                if chunk:
                                    f.write(chunk)

                        if j > 0:
                            if os.path.isfile(path_stack[-1] + ".zip"):
                                if utils.compare_paths(
                                    path_stack[-1] + ".zip", file_path + ".zip"
                                ):
                                    os.remove(file_path + ".zip")
                                    print(
                                        f"File {file_name} is not different from previous files so it was skipped."
                                    )
                                    j += 1
                                    continue
                                else:
                                    os.remove(path_stack[-1] + ".zip")

                        os.makedirs(file_path, exist_ok=True)
                        with zipfile.ZipFile(file_path + ".zip", "r") as zip_ref:
                            zip_ref.extractall(file_path)

                        print(f"File {file_name} downloaded successfully.")
                    else:
                        warnings.warn(
                            f"Error downloading file {file_name} with ID {file_id}: {response.status_code} - {response.text}"
                        )

                calendar_path = os.path.normpath(file_path + "/calendar.txt")
                calendar_dates_path = os.path.normpath(
                    file_path + "/calendar_dates.txt"
                )
                new_j = None
                if os.path.isfile(calendar_path):
                    new_j, min_end_date = process_calendar(
                        calendar_path,
                        file_date,
                        possible_dates=dates,
                        day_separation=day_separation,
                        calendar_path=calendar_path,
                    )

                if os.path.isfile(calendar_dates_path):
                    new_j, min_end_date = process_calendar_dates(
                        calendar_dates_path,
                        file_date,
                        possible_dates=dates,
                        day_separation=day_separation,
                        calendar_dates_path=calendar_dates_path,
                        id=new_j,
                        has_calendar=os.path.isfile(calendar_path),
                    )

                if (not os.path.isfile(calendar_path)) and (
                    not os.path.isfile(calendar_dates_path)
                ):
                    warnings.warn(
                        f"File {file_name} does not have calendar.txt or calendar_dates.txt"
                    )
                    new_j = j + 1

                path_stack.append(file_path)
                if new_j is None:
                    new_j = j + 1
                elif new_j <= j:
                    new_j = j + 1
                elif dates[new_j] >= end_date:
                    break
                elif (min_end_date is not None) and (min_end_date >= end_date):
                    break

                if new_j >= len(links):
                    break

                if new_j > (j + 1):
                    for jj in range(j + 1, new_j):
                        print(
                            f"Skipping file {main_name}{dates[jj].strftime('%Y%m%d')} as it is too near to last file {file_name}"
                        )

                j = new_j

            if os.path.isfile(path_stack[-1] + ".zip"):
                os.remove(path_stack[-1] + ".zip")

            stack_gtfs.historic_stack(path_stack, main_path, aoi)
            print("Finished post-process")
            for f in path_stack:
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)

            all_paths.append(main_path)

        return all_paths

    def find_file_names(
        self,
        base_path: str = "",
        region: Union[int, str, list] = [],
        transport_type: Union[int, str, list] = [],
        organization: Union[int, str, list] = [],
        file_type: Union[int, str] = "GTFS",
        region_type: Union[int, str, list] = 3,
        start_date: str = None,
        end_date: str = None,
        file_description: Union[str, list] = [],
    ):
        """Filters and obtains the list of files by municipality, transport type, and file type."""
        files = self.find_files(
            region=region,
            transport_type=transport_type,
            organization=organization,
            file_type=file_type,
            region_type=region_type,
            start_date=start_date,
            end_date=end_date,
            file_description=file_description,
            metadata=True,
        )
        file_names = []
        for i in range(len(files)):
            main_name = sanitize_filename(files[i]["nombre"])
            data = files[i]["ficherosDto"]
            for j in range(len(data)):
                if j > 0:
                    name = main_name + f"_{j + 1}"
                else:
                    name = main_name

                name = os.path.normpath(base_path + "/" + name)
                file_names.append(name)

        return file_names

    def download_files(
        self,
        file_ids: Union[list, int, str, dict],
        output_path: str,
        overwrite: bool = False,
        update: bool = True,
    ):
        """Downloads a specific file given its ID."""
        os.makedirs(output_path, exist_ok=True)

        _file_ids = file_ids
        if isinstance(_file_ids, list):
            _file_ids = [_file_ids]
        if len(file_ids) == 0:
            return []

        if isinstance(_file_ids[0], str):
            _file_ids = self.get_file_id(_file_ids)

        if isinstance(_file_ids[0], int):
            _file_ids = [self.get_file_metadata(i) for i in _file_ids]

        file_names = []
        new_file_ids = []
        dataset_dates = []
        for i in range(len(_file_ids)):
            main_name = sanitize_filename(_file_ids[i]["nombre"])
            data = _file_ids[i]["ficherosDto"]
            for j in range(len(data)):
                if j > 0:
                    name = main_name + f"_{j + 1}"
                else:
                    name = main_name

                name = os.path.normpath(output_path + "/" + name)
                file_names.append(name)
                new_file_ids.append(data[j]["ficheroId"])
                dataset_dates.append(_file_ids[i]["fechaCreacion"])

        _file_ids = new_file_ids

        for i in range(len(_file_ids)):
            if os.path.isdir(file_names[i]) and (not overwrite):
                if update:
                    file_creation_date = datetime.fromtimestamp(
                        os.path.getctime(file_names[i])
                    )
                    try:
                        dataset_creation_date = datetime.strptime(
                            dataset_dates[i], "%Y-%m-%dT%H:%M:%S.%f"
                        )
                    except ValueError:
                        dataset_creation_date = datetime.strptime(
                            dataset_dates[i], "%Y-%m-%dT%H:%M:%S"
                        )

                    if file_creation_date >= dataset_creation_date:
                        print(f"File {file_names[i]} already exists. Skipping.")
                        continue
                    else:
                        print(
                            f"Updated file {file_names[i]} is available. Overwriting."
                        )

                else:
                    print(f"File {file_names[i]} already exists. Skipping.")
                    continue

            url = f"{self.BASE_URL}/Fichero/download/{_file_ids[i]}"
            response = requests.get(url, headers=self.headers, stream=True)
            if response.status_code == 200:
                with open(file_names[i] + ".zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                os.makedirs(file_names[i], exist_ok=True)
                with zipfile.ZipFile(file_names[i] + ".zip", "r") as zip_ref:
                    zip_ref.extractall(file_names[i])

                os.remove(file_names[i] + ".zip")

                print(f"File {file_names[i]} downloaded successfully.")
            else:
                print(
                    f"Error downloading file with ID {_file_ids[i]}: {response.status_code} - {response.text}"
                )

        return file_names
