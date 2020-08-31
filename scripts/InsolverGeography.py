import requests
import requests_cache
import pandas as pd
import json
import pyodbc
import math

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
            {'name': ['kladr', 'address'],
             'type': ['number', 'number'],
             },
            {'name': 'policies_count', 'type': 'number',},
            {'name': 'claims_count', 'type': 'number',},
            {'name': 'claims_count_adj', 'type': 'number',},
            {'name': 'claims_sum_infl', 'type': 'number',},
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


def kladr_to_address(kladr, token):
    """
    Gets addresses from KLADRs by api https://dadata.ru/.

    :param kladr: KLADR.
    :param token: Token for api https://dadata.ru/.
    :returns: Address.
    """
    data = {
        "query": kladr
    }
    api_url = 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/fias'
    headers = {
        'content-type': 'application/json',
        'Authorization': token,  # 'Token 79abf89d58871ed1df79b83126f8f8c2362e51db'
    }
    _response = requests.post(api_url, json=data, headers=headers)
    _address_json = _response.json()
    _address_str = _address_json['suggestions'][0]['value']
    return _address_str


def address_to_geo_coordinates(address):
    """
    Gets geo coordinates from addresses by api http://api.sputnik.ru/maps/geocoder/.

    :param address: Address.
    :returns: Coordinates ['latitude', 'longitude'].
    """
    request = requests.get(f'http://search.maps.sputnik.ru/search/addr?q={address}')
    response = request.json()
    _coordinates = response['result']['address'][0]['features'][0]['geometry']['geometries'][0]['coordinates']
    _coordinates = _coordinates[::-1]
    return _coordinates


class GetCoordinates:
    """
    Gets geo coordinates in columns 'latitude' and 'longitude' of dataframe.

    :param type: 'kladr' or 'address' to get coordinates from.
    :param token: Token for api https://dadata.ru/ (only if 'type' is 'kladr').
    """
    def __init__(self):
        self._priority = 0

    def __call__(self, df, type, token=None):

        _addresses = []
        _coordinates = []

        if type == 'kladr':
            for kladr in df['kladr']:
                _address = kladr_to_address(kladr, token)
                _addresses.append(_address)
            df['address'] = _addresses

        for address in df['address']:
            _coord = address_to_geo_coordinates(address)
            _coordinates.append(_coord)

        df['latitude'], df['longitude'] = zip(*_coordinates)

        return df


# ---------------------------------------------------
# Geo methods
# ---------------------------------------------------


def geo_dist(src_lat, src_lon, dst_lat, dst_lon):
    dist = 6371 * 2 * math.asin(
        math.sqrt(
            math.pow( math.sin( ( src_lat - dst_lat ) * math.pi/180 / 2 ), 2 )
            +
            math.cos( src_lat * math.pi/180 ) * math.cos( dst_lat * math.pi/180 ) *
                math.pow( math.sin( ( src_lon - dst_lon ) * math.pi/180 / 2 ), 2 )
            )
    )
    return round(dist, 0)


class GeoPointPoint:

    def __init__(self, geo_points_df):
        self._geo_points_df = geo_points_df
        self._geo_points_points_df = geo_points_df.join(geo_points_df, on=None, how='outer', lsuffix='1', rsuffix='2')
        self._geo_points_points_df['geo_dist'] = geo_dist(self._geo_points_points_df[[
            'latitude_1', 'longitude_1', 'latitude_2', 'longitude_2']])


---------------------------------

update [dbo].[data_kladr]
set
	 count_pol_200 = T.count_pol
	,premium_200 = T.premium
	,count_pol_with_claim_200 = T.count_pol_with_claim
	,claim_count_200 = T.claim_count
	,claim_count_adj_200 = T.claim_count_adj
	,claim_sum_200 = T.claim_sum
	,paid_sum_200 = T.paid_sum
	,claim_sum_infl_200 = T.claim_sum_infl
	,paid_sum_infl_200 = T.paid_sum_infl
from
	(
		select
			 [kladr_1]
			,sum([data_kladr_kladr].count_pol_2) as count_pol
			,sum([data_kladr_kladr].premium_2) as premium
			,sum([data_kladr_kladr].count_pol_with_claim_2) as count_pol_with_claim
			,sum([data_kladr_kladr].claim_count_2) as claim_count
			,sum([data_kladr_kladr].claim_count_adj_2) as claim_count_adj
			,sum([data_kladr_kladr].claim_sum_2) as claim_sum
			,sum([data_kladr_kladr].paid_sum_2) as paid_sum
			,sum([data_kladr_kladr].claim_sum_infl_2) as claim_sum_infl
			,sum([data_kladr_kladr].paid_sum_infl_2) as paid_sum_infl
		from [dbo].[data_kladr_kladr]
		where isnull(dist,10000) <= 200
		group by [kladr_1]
	) as T
where T.[kladr_1] = [data_kladr].[insurerKLADR]

update [dbo].[data_kladr]
set
	 count_pol_500 = T.count_pol
	,premium_500 = T.premium
	,count_pol_with_claim_500 = T.count_pol_with_claim
	,claim_count_500 = T.claim_count
	,claim_count_adj_500 = T.claim_count_adj
	,claim_sum_500 = T.claim_sum
	,paid_sum_500 = T.paid_sum
	,claim_sum_infl_500 = T.claim_sum_infl
	,paid_sum_infl_500 = T.paid_sum_infl
from
	(
		select
			 [kladr_1]
			,sum([data_kladr_kladr].count_pol_2) as count_pol
			,sum([data_kladr_kladr].premium_2) as premium
			,sum([data_kladr_kladr].count_pol_with_claim_2) as count_pol_with_claim
			,sum([data_kladr_kladr].claim_count_2) as claim_count
			,sum([data_kladr_kladr].claim_count_adj_2) as claim_count_adj
			,sum([data_kladr_kladr].claim_sum_2) as claim_sum
			,sum([data_kladr_kladr].paid_sum_2) as paid_sum
			,sum([data_kladr_kladr].claim_sum_infl_2) as claim_sum_infl
			,sum([data_kladr_kladr].paid_sum_infl_2) as paid_sum_infl
		from [dbo].[data_kladr_kladr]
		where isnull(dist,10000) <= 500
		group by [kladr_1]
	) as T
where T.[kladr_1] = [data_kladr].[insurerKLADR]
