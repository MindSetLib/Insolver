import requests
import requests_cache
import pandas as pd
import math

from scripts.InsolverDataFrame import InsolverDataFrame


class InsolverGeoFrame(InsolverDataFrame):
    """
    Dataframe class for Insolver Geo module.
    """
    def __init__(self, df_points):
        super().__init__(df_points)

    # ---------------------------------------------------
    # Coordinates methods
    # ---------------------------------------------------

    @staticmethod
    def _get_address_from_kladr(_kladr, _token):
        """
        Gets addresses from KLADRs by api https://dadata.ru/.

        :param _kladr: KLADR.
        :param _token: Token for api https://dadata.ru/.
        :returns: Address.
        """
        _data = {
            "query": _kladr
        }
        _api_url = 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/fias'
        _headers = {
            'content-type': 'application/json',
            'Authorization': _token,
        }
        _response = requests.post(_api_url, json=_data, headers=_headers)
        _address_json = _response.json()
        _address_str = _address_json['suggestions'][0]['value']
        return _address_str

    def get_address_from_kladr(self, column_kladr, column_address, token):
        """
        Gets addresses from KLADRs by api https://dadata.ru/.

        :param column_kladr: Column in InsolverDataFrame with KLADRs.
        :param column_address: Column in InsolverDataFrame for addresses.
        :param token: Token for api https://dadata.ru/.
        :returns: None.
        """
        _addresses = []
        for _kladr in self._df[column_kladr]:
            _address = self._get_address_from_kladr(_kladr, token)  # token='Token 79abf89d58871ed1df79b83126f8f8c2362e51db'
            _addresses.append(_address)
        self._df[column_address] = _addresses

    @staticmethod
    def _get_coordinates_from_address(_address):
        """
        Gets geo coordinates from addresses by api http://api.sputnik.ru/maps/geocoder/.

        :param _address: Address.
        :returns: Coordinates ['latitude', 'longitude'].
        """
        _request = requests.get(f'http://search.maps.sputnik.ru/search/addr?q={_address}')
        _response = _request.json()
        _coordinates = _response['result']['address'][0]['features'][0]['geometry']['geometries'][0]['coordinates']
        _coordinates = _coordinates[::-1]
        return _coordinates

    def get_coordinates_from_address(self, column_address, column_lat, column_lon):
        """
        Gets geo coordinates from addresses by api http://api.sputnik.ru/maps/geocoder/.

        :param column_address: Column in InsolverDataFrame with addresses.
        :param column_lat: Column in InsolverDataFrame for latitudes.
        :param column_lon: Column in InsolverDataFrame for longitudes.
        :returns: None.
        """
        _coordinates = []
        for _address in self._df[column_address]:
            _coord = self._get_coordinates_from_address(_address)
            _coordinates.append(_coord)
            self._df[column_lat], self._df[column_lon] = zip(*_coordinates)

    @staticmethod
    def _get_geo_dist(_start_lat, _start_lon, dst_lat, dst_lon):
        """
        Gets geographical distance between two points.

        :param _start_lat: Latitude of the start point.
        :param _start_lon: Longitude of the start point.
        :param dst_lat: Latitude of the destination point.
        :param dst_lon: Longitude of the destination point.
        :return: Distance in kilometers.
        """
        _dist = 6371 * 2 * math.asin(
            math.sqrt(
                math.pow( math.sin( ( src_lat - dst_lat ) * math.pi/180 / 2 ), 2 )
                +
                math.cos( src_lat * math.pi/180 ) * math.cos( dst_lat * math.pi/180 ) *
                    math.pow( math.sin( ( src_lon - dst_lon ) * math.pi/180 / 2 ), 2 )
                )
        )
        return round(_dist, 0)

    # ---------------------------------------------------
    # Geo methods
    # ---------------------------------------------------



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
