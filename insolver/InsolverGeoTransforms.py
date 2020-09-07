import requests
import requests_cache
import pandas as pd
import math
import datetime

from .InsolverDataFrame import InsolverDataFrame
from .InsolverMain import InsolverTransformMain


# ---------------------------------------------------
# Dataframe with geo points
# ---------------------------------------------------


class InsolverGeoPointsFrame(InsolverDataFrame):
    """
    Dataframe class with geo points for Insolver Geo module.
    """
    def __init__(self, geo_points):
        super().__init__(geo_points)

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


# ---------------------------------------------------
# Dataframe with relations geo points to geo points
# ---------------------------------------------------


class InsolverGeoPointsToPointsFrame(InsolverDataFrame):
    """
    Dataframe class with relations geo points to geo points for Insolver Geo module.

    :param geo_points_main: InsolverGeoPointsFrame with geo points.
    :param geo_points_rel: InsolverGeoPointsFrame with geo points to make relations with.
    """
    def __init__(self, geo_points_main, geo_points_rel):
        now = str(datetime.datetime.now())
        geo_points_main[f'key_{now}'] = 0
        geo_points_rel[f'key_{now}'] = 0
        super().__init__(geo_points_main.merge(geo_points_rel, on=f'key_{now}', how='outer', suffixes=('_1', '_2')))
        self._df.drop(columns=f'key_{now}', inplace=True)


# ---------------------------------------------------
# Geo methods
# ---------------------------------------------------


class TransformGeoDistGet(InsolverTransformMain):
    """
    Gets geographical distance between two points in kilometers.

    :param column_start_lat: Column in InsolverGeoPointsToPointsFrame with start points' latitudes.
    :param column_start_lon: Column in InsolverGeoPointsToPointsFrame with start points' longitudes.
    :param column_dst_lat: Column in InsolverGeoPointsToPointsFrame with destination points' latitudes.
    :param column_dst_lon: Column in InsolverGeoPointsToPointsFrame with destination points' longitudes.
    :param column_dist: Column in InsolverGeoPointsToPointsFrame for distance.
    """
    def __init__(self, column_start_lat, column_start_lon, column_dst_lat, column_dst_lon, column_dist):
        self.priority = 0
        super().__init__()
        self.column_start_lat = column_start_lat
        self.column_start_lon = column_start_lon
        self.column_dst_lat = column_dst_lat
        self.column_dst_lon = column_dst_lon
        self.column_dist = column_dist

    @staticmethod
    def _get_geo_dist(_start_lat_lon_dst_lat_lon):
        _start_lat = _start_lat_lon_dst_lat_lon[0]
        _start_lon = _start_lat_lon_dst_lat_lon[1]
        _dst_lat = _start_lat_lon_dst_lat_lon[2]
        _dst_lon = _start_lat_lon_dst_lat_lon[3]
        _dist = 6371 * 2 * math.asin(
            math.sqrt(
                math.pow(math.sin((_start_lat - _dst_lat) * math.pi / 180 / 2), 2)
                +
                math.cos(_start_lat * math.pi / 180) * math.cos(_dst_lat * math.pi / 180)
                *
                math.pow(math.sin((_start_lon - _dst_lon) * math.pi / 180 / 2), 2)
            )
        )
        return round(_dist, 0)

    def __call__(self, df):
        df[self.column_dist] = df[[self.column_start_lat, self.column_start_lon, self.column_dst_lat,
                                   self.column_dst_lon]].apply(self._get_geo_dist, axis=1)


class TransformParamsSumAround(InsolverTransformMain):
    """
    Gets sum of numeric parameter's values in surround of each point.

    :param column_link_p: Link column in InsolverGeoPointsFrame.
    :param column_link_ptp: Link column in InsolverGeoPointsToPointsFrame.
    :param columns_params_ptp: List of columns in InsolverGeoPointsToPointsFrame to calculate sums of values.
    :param column_dist_ptp: Column in InsolverGeoPointsToPointsFrame with distances between points.
    :param dist_max: Maximum distance between points for include into surround.
    :param columns_results_p: List of columns in InsolverGeoPointsFrame for calculations' results.
    """
    def __init__(self, column_link_p, column_link_ptp, columns_params_ptp, column_dist_ptp, dist_max, columns_results_p):
        self.priority = 1
        super().__init__()
        self.column_link_p = column_link_p
        self.column_link_ptp = column_link_ptp
        self.columns_params_ptp = columns_params_ptp
        self.column_dist_ptp = column_dist_ptp
        self.dist_max = dist_max
        self.columns_results_p = columns_results_p

    def __call__(self, df_points, df_points_to_points):
        _df = (df_points[[self.column_link_p, ]].merge(
            df_points_to_points.loc[df_points_to_points[self.column_dist_ptp] <= self.dist_max,
                                    [self.column_link_ptp] + self.columns_params_ptp],
            left_on=self.column_link_p, right_on=self.column_link_ptp, how='outer')).groupby(
            self.column_link_p).sum().reset_index()
        _df.rename(columns=dict(zip(self.columns_params_ptp, self.columns_results_p)), inplace=True)
        df_points = df_points.merge(_df, left_on=self.column_link_p, right_on=self.column_link_p, how='outer')
        return df_points


class TransformAddParams(InsolverTransformMain):
    """
    Adds parameters' values from InsolverGeoPointsFrame to InsolverDataFrame.

    :param column_link_df: Link column in InsolverDataFrame.
    :param column_link_p: Link column in InsolverGeoPointsFrame.
    :param columns_params: List of columns in InsolverGeoPointsFrame to merge into InsolverDataFrame.
    """
    def __init__(self, column_link_df, column_link_p, columns_params):
        self.priority = 2
        super().__init__()
        self.column_link_df = column_link_df
        self.column_link_p = column_link_p
        self.columns_params = columns_params

    def __call__(self, df, df_points):
        df = df.merge(df_points[[self.column_link_p] + self.columns_params], left_on=self.column_link_df,
                      right_on=self.column_link_p, how='left')
        return df
