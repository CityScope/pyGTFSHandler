import requests
import difflib
from typing import Union
from datetime import datetime
import unicodedata
import os
import zipfile
import re 
from copy import copy

def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Remove tildes (accent marks) by decomposing and stripping accents
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    return s

def clean_filename(filename):
    # Replace special characters and multiple spaces/underscores with a single underscore
    filename = normalize_string(filename)
    cleaned_filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)  # Replace non-alphanumeric characters with underscores
    cleaned_filename = re.sub(r'_+', '_', cleaned_filename)    # Replace multiple underscores with a single one
    cleaned_filename = cleaned_filename.strip('_')             # Remove leading/trailing underscores
    
    return cleaned_filename

class APIClient:
    BASE_URL = "https://nap.transportes.gob.es/api"

    def __init__(self, api_key:str = None):
        self.api_key = api_key or os.getenv('NAP_API_KEY', "")
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

    def get_region_id(self,name:str,region_type:Union[int, str]=3):     
        """Gets the region ID of the region by matching the closest name."""
        if (region_type == 0) or (normalize_string(region_type) == 'provincia') or (normalize_string(region_type) == 'province'
                                                                            ) or (normalize_string(region_type) == 'region'):
            region_type = 'Provincia'
        elif (region_type == 1) or (normalize_string(region_type).replace(" ","") == 'comunidadautonoma'
                                    ) or (normalize_string(region_type) == 'state'): 
            region_type = 'ComunidadAutonoma'
        elif (region_type == 2) or (normalize_string(region_type) == 'ciudad') or (normalize_string(region_type).replace(" ","") == 'areaurbana'
                            ) or (normalize_string(region_type) == 'city') or (normalize_string(region_type).replace(" ","") == 'urbanarea'):
            region_type = 'AreaUrbana'
        elif (region_type == 3) or (normalize_string(region_type) == 'municipio') or (normalize_string(region_type) == 'municipality'):
            region_type = 'Municipio'
        else:
            raise Exception(f"region type {region_type} not valid.")
        
        name = normalize_string(name)
        region_type = normalize_string(region_type)
        
        url = f"{self.BASE_URL}/Region"#/GetByName/{name}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            regions = response.json()

            # Filter for regions that are municipalities based on 'tipoNombre'
            regions = [i for i in regions if normalize_string(i.get('tipoNombre')) == region_type]
            
            if regions:
                # Extract the names of the municipalities
                region_names = [normalize_string(i['nombre']) for i in regions]
                
                # Find the closest matching name to the input 'municipality_name'
                closest_match = difflib.get_close_matches(name, region_names, n=1)
                
                if closest_match:
                    closest_match = normalize_string(closest_match[0])
                    # Get the municipality that matches the closest name
                    selected_region = next(i for i in regions if normalize_string(i['nombre']) == closest_match)
                    return selected_region['regionId']  # Return the ID of the selected municipality
                else:
                    print(f"No close match found for municipality '{name}'.")
            else:
                print(f"No municipalities found with the name '{name}'.")
        
        return None

    def get_transport_type_id(self,transport_name:str):
        """Obtains the ID of the transport type."""
        transport_name = normalize_string(transport_name)
        if (transport_name == 'bus') or (transport_name == 'autobus'):
            transport_name = 'autobus'
        elif (transport_name == 'tren') or (transport_name == 'ferrocarril') or (transport_name == 'rail') or (transport_name == 'train'): 
            transport_name = 'ferroviario'
        elif (transport_name == 'barco') or (transport_name == 'boat') or (transport_name == 'ferry'):
            transport_name = 'maritimo'
        elif (transport_name == 'avion') or (transport_name == 'plane') or (transport_name == 'air') or (transport_name == 'aereo'):
            transport_name = 'aereo'
            
        url = f"{self.BASE_URL}/TipoTransporte"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            transport_types = response.json()
            for transport in transport_types:
                if normalize_string(transport['nombre']) == transport_name:#.lower() == transport_name.lower():
                    return transport['tipoTransporteId']
        print(f"Transport type '{transport_name}' not found.")
        return None

    def get_file_type_id(self,file_type:str="GTFS"):
        """Obtains the ID of the file type, defaulting to 'GTFS'."""
        file_type = normalize_string(file_type)
        url = f"{self.BASE_URL}/TipoFichero"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            file_types = response.json()
            for f in file_types:
                if normalize_string(f['nombre']) == file_type:#.lower() == file_name.lower():
                    return f['tipoFicheroId']
        print(f"File type '{file_type}' not found.")
        return None

    def get_organization_id(self,organization_name:str):
        """Obtains the ID of the organization by its name."""
        organization_name = normalize_string(organization_name)
        url = f"{self.BASE_URL}/Organizacion/GetByName/{organization_name}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            organizations = response.json()
            
            # Extract the names of all organizations
            org_names = [normalize_string(org['nombre']) for org in organizations]
            
            # Find the closest matching name to the input 'organization_name'
            closest_match = difflib.get_close_matches(organization_name, org_names, n=1)
            
            if closest_match:
                closest_match = normalize_string(closest_match[0])
                # Get the organization that matches the closest name
                selected_org = next(org for org in organizations if normalize_string(org['nombre']) == closest_match)
                return selected_org['organizacionId']  # Return the ID of the selected organization
            else:
                print(f"No close match found for organization '{organization_name}'.")
        else:
            print(f"Error fetching organizations: {response.status_code} - {response.text}")
        
        return None

    def get_file_id(self,file_name: Union[str,list]):
        """Obtains the file ID by the dataset (conjunto de datos) name."""
        _file_name = copy(file_name)
        if type(_file_name) == str:
            _file_name = [_file_name]

        for i in range(len(_file_name)):
            if type(_file_name[i]) != str:
                raise Exception(f"file name should be str or list[str] but got an element {i} has type {type(_file_name[i])}.")
            
            _file_name[i] = clean_filename(_file_name[i])  # Normalize the input dataset name

        url = f"{self.BASE_URL}/Fichero/GetList"  # Assuming dataset name is passed here
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            datasets = response.json()  # Parse the list of datasets (conjuntos de datos)
            
            # Extract the names of all datasets
            dataset_names = [clean_filename(dataset['nombre']) for dataset in datasets['conjuntosDatoDto']]
            
            file_ids = []
            for name in _file_name:
                # Find the closest matching dataset name to the input 'conjunto_dato_name'
                closest_match = difflib.get_close_matches(name, dataset_names, n=1)
                
                if closest_match:
                    closest_match = clean_filename(closest_match[0])
                    # Get the dataset that matches the closest name
                    selected_dataset = next(dataset for dataset in datasets['conjuntosDatoDto'] if clean_filename(dataset['nombre']) == closest_match)
                    file_ids.append(selected_dataset['conjuntoDatoId'])# Return the file ID ('conjuntoDatoId') of the selected dataset
                else:
                    print(f"No close match found for dataset '{name}'.")

            return file_ids
        else:
            print(f"Error fetching datasets: {response.status_code} - {response.text}")
        
        return None

    def get_file_metadata(self,file_id:Union[str,int]):
        if type(file_id) == str:
            file_id = self.get_file_id(file_id)
            if len(file_id) == 0:
                return []
            else:
                file_id = file_id[0]

        url = f"{self.BASE_URL}/Fichero/{file_id}"  # Assuming dataset name is passed here
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            metadata = response.json()  # Parse the list of datasets (conjuntos de datos)
            return metadata#['conjuntosDatoDto']
        else:
            print(f"Error fetching metadata: {response.status_code} - {response.text}")

        return None

    def find_files(self,region:Union[int, str, list]=[],
                    transport_type:Union[int, str,list]=[],
                    organization:Union[int, str,list]=[], 
                    file_type:Union[int, str]='GTFS',
                    region_type:Union[int, str]=3, 
                    start_date:str=None,end_date:str=None, 
                    file_description:Union[str,list]=[], metadata:bool=True):
        
        """Filters and obtains the list of files by municipality, transport type, and file type.
            Filtering by date could not work if metadata=False."""

        if type(end_date) == type(None):
            end_date = start_date 
        elif type(start_date) == type(None):
            start_date = end_date

        if type(start_date) != type(None):
            if start_date == 'today':
                start_date = datetime.now().strftime("%d-%m-%Y")

            if end_date == 'today':
                end_date = datetime.now().strftime("%d-%m-%Y")

            try:
                start_date = datetime.strptime(start_date, "%d%m%Y")
            except:
                start_date = datetime.strptime(start_date, "%d-%m-%Y")

            try:
                end_date = datetime.strptime(end_date, "%d%m%Y")
            except:
                end_date = datetime.strptime(end_date, "%d-%m-%Y")

        _organization = copy(organization)
        if type(_organization) != list:
            _organization = [_organization]

        _region = copy(region)
        if type(_region) != list:
            _region = [_region]

        _transport_type = copy(transport_type)
        if type(_transport_type) != list:
            _transport_type = [_transport_type]

        _file_description = copy(file_description)
        if type(_file_description) != list:
            _file_description = [_file_description]

        for i in range(len(_organization)):
            if type(_organization[i]) == str:
                _organization[i] = self.get_organization_id(_organization[i])
        
        for i in range(len(_region)):
            if type(_region[i]) == str:
                _region[i] = self.get_region_id(_region[i],region_type)
        
        for i in range(len(_transport_type)):
            if type(_transport_type[i]) == str:
                _transport_type[i] = self.get_transport_type_id(_transport_type[i])

        if type(file_type) == str:
            file_type = self.get_file_type_id(file_type)

        url = f"{self.BASE_URL}/Fichero/Filter"
        
        data = {
            "provincias": _region,
            "comunidades": _region,
            "areasurbanas": _region,
            "municipios": _region,
            "tipotransportes": _transport_type,
            "tipoficheros": [file_type],
            "organizaciones": _organization
        }
        response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, json=data)
        if response.status_code == 200:
            files = response.json()

            if files['filesNum'] > 0:
                files = files['conjuntosDatoDto']
            else:
                return []
        
            if type(start_date) != type(None):
                filtered_files = []
                for file in files:
                    new_data = []
                    for data in file['ficherosDto']:
                        file_start_date = datetime.strptime(data['fechaDesde'], "%Y-%m-%dT%H:%M:%S")
                        file_end_date = datetime.strptime(data['fechaHasta'], "%Y-%m-%dT%H:%M:%S")
                        
                        if file_start_date <= start_date and file_end_date >= end_date:
                            new_data.append(data)

                    if len(new_data) > 0:
                        file['ficherosDto'] = new_data
                        filtered_files.append(file)

                files = filtered_files
                
            
            if len(_file_description) > 0:
                for i in range(len(_file_description)):
                    _file_description[i] = normalize_string(_file_description[i])

                new_files = []
                for i in range(len(files)):
                    data = normalize_string(files[i]['descripcion'])
                    name = normalize_string(files[i]['nombre'])
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
                    file_ids.append(files[i]['conjuntoDatoId'])

                return file_ids
        
        print("Error filtering files:", response.status_code, response.text)
        return None

    def find_file_names(self,base_path:str="", region:Union[int, str, list]=[],
                    transport_type:Union[int, str,list]=[],
                    organization:Union[int, str,list]=[], 
                    file_type:Union[int, str]='GTFS',
                    region_type:Union[int, str]=3, 
                    start_date:str=None,end_date:str=None, 
                    file_description:Union[str,list]=[]):
        
        """Filters and obtains the list of files by municipality, transport type, and file type."""
        files = self.find_files(region=region,transport_type=transport_type,organization=organization,file_type=file_type,
                                    region_type=region_type,start_date=start_date,end_date=end_date,file_description=file_description,metadata=True)
        file_names = []
        for i in range(len(files)):
            main_name = clean_filename(files[i]['nombre'])
            data = files[i]['ficherosDto']
            for j in range(len(data)):
                if j > 0:
                    name = main_name + f"_{j+1}"
                else:
                    name = main_name
                
                name = os.path.normpath(base_path + "/" + name)
                file_names.append(name)

        return file_names

    def download_file(self,file_ids:Union[list,int,str,dict],output_path:str,overwrite:bool=False,update:bool=True):
        
        """Downloads a specific file given its ID."""
        _file_ids = file_ids
        if type(_file_ids) != list:
            _file_ids = [_file_ids]
        if len(file_ids) == 0:
            return [] 
        
        if type(_file_ids[0]) == str:
            _file_ids = self.get_file_id(_file_ids)

        if type(_file_ids[0]) == int:
            _file_ids = [self.get_file_metadata(i) for i in _file_ids] 
        
        file_names = []
        new_file_ids = []
        dataset_dates = []
        for i in range(len(_file_ids)):
            main_name = clean_filename(_file_ids[i]['nombre'])
            data = _file_ids[i]['ficherosDto']
            for j in range(len(data)):
                if j > 0:
                    name = main_name + f"_{j+1}"
                else:
                    name = main_name
                
                name = os.path.normpath(output_path + "/" + name)
                file_names.append(name)
                new_file_ids.append(data[j]['ficheroId'])
                dataset_dates.append(_file_ids[i]['fechaCreacion'])

        _file_ids = new_file_ids

        for i in range(len(_file_ids)):
            if os.path.isdir(file_names[i]) and (overwrite == False):
                if update: 
                    file_creation_date = datetime.fromtimestamp(os.path.getctime(file_names[i]))
                    dataset_creation_date = datetime.strptime(dataset_dates[i], '%Y-%m-%dT%H:%M:%S.%f')
                    if file_creation_date >= dataset_creation_date:
                        print(f"File {file_names[i]} already exists. Skipping.")
                        continue 
                    else:
                        print(f"Updated file {file_names[i]} is available. Overwriting.")

                else:
                    print(f"File {file_names[i]} already exists. Skipping.")
                    continue 

            url = f"{self.BASE_URL}/Fichero/download/{_file_ids[i]}"
            response = requests.get(url, headers=self.headers, stream=True)
            if response.status_code == 200:
                with open(file_names[i]+".zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                os.makedirs(file_names[i], exist_ok=True)
                with zipfile.ZipFile(file_names[i]+".zip", 'r') as zip_ref:
                    zip_ref.extractall(file_names[i])

                os.remove(file_names[i]+'.zip')

                print(f"File {file_names[i]} downloaded successfully.")
            else:
                print(f"Error downloading file with ID {_file_ids[i]}: {response.status_code} - {response.text}")

        return file_names