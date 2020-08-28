import requests
import requests_cache
import pandas as pd
import json
import pyodbc

from scripts.InsolverDataFrame import InsolverDataFrame


class InsolverGeoFrame(InsolverDataFrame):
    """
    Dataframe class for Insolver Geography module. Is similar to pandas' DataFrame.
    """
    def __init__(
            self,
            data=None,
            sep=',',
            encoding=None,
            driver='{SQL Server}',
            server=None,
            database=None,
            username=None,
            password=None,
            table=None
    ):
        super().__init__(data, sep, encoding, driver, server, database, username, password, table)

    _df_columns_default = {
        'json': '_df_columns_geo',
        'columns': [
            {'name': 'kladr',},
            {'name': 'policies_count'},
            {'name': 'claims_count',},
            {'name': 'claims_count_adj',},
            {'name': 'claims_sum_infl',},
        ]
    }

    # ---------------------------------------------------
    # Get data methods
    # ---------------------------------------------------

    def get_pd(self, columns=None):
        """
        Gets loaded data.

        :param columns: Columns of dataframe to get.
        :returns: Pandas Dataframe.
        """
        if self._is_frame is None:
            return None
        if columns is None:
            columns = self._df.columns
        return self._df[columns].copy()


# ---------------------------------------------------
# Get coordinates methods
# ---------------------------------------------------


class KLADRtoAddress:
    """
    Gets addresses from KLADRs by api https://dadata.ru/
    """
    def __init__(self):
        self.priority = 1

    def __call__(self, kladr, token):
        data = {
            "query": kladr
        }
        api_url = 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/fias'
        headers = {
            'content-type': 'application/json',
            'Authorization': token,  # 'Token 79abf89d58871ed1df79b83126f8f8c2362e51db'
        }
        response = requests.post(api_url, json=data, headers=headers)
        adress_json = response.json()
        adress_str = adress_json['suggestions'][0]['value']
        return adress_str


class address_to_geo_coord_api:
    """
    Gets geo coordinates from address by api http://api.sputnik.ru/maps/geocoder/
    """
def (address):
    r = requests.get(f'http://search.maps.sputnik.ru/search/addr?q={address}')
    response = r.json()
    coordinates = response['result']['address'][0]['features'][0]['geometry']['geometries'][0]['coordinates']
    coordinates = coordinates[::-1]
    return coordinates

addresses = []
coordinates = []
for kladr_id in kladr_list:
    address = kladr_to_address_api(kladr_id)
    addresses.append(address)
    coord = address_to_geo_coord_api(address)
    coordinates.append(coord)
    print(address, coord)
df['address'] = addresses
df['coordinates'] = coordinates

