import pandas as pd
import re
import datetime
import numpy as np
import json

from scripts.InsolverMain import InsolverMain


class InsolverTransformer(InsolverMain):

    def __init__(self, df):
        self._is_frame = False
        if type(df) == pd.DataFrame:
            self._df = df
            self._is_frame = True

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
        return self._df[columns]

    # ---------------------------------------------------
    # Person data methods
    # ---------------------------------------------------

    _client_type_dict = {
        'person': float(0),
        'company': float(1),
        '0': float(0),
        '1': float(1),
        0: float(0),
        1: float(1)
    }

    def transform_client_type(self):
        """
        Transforms values in column 'client_type' from {'person','company'} to {0,1}.

        :returns: None.
        """
        self._df['client_type'] = self._df['client_type'].map(self._client_type_dict)

    @staticmethod
    def _gender(client_type_name_gender):

        _client_type = client_type_name_gender[0]
        _client_name = client_type_name_gender[1]
        _client_gender = client_type_name_gender[2]

        if _client_type == 'company':  # juridic
            _gender_m = 0
            _gender_f = 0

        elif _client_type == '1':  # juridic
            _gender_m = 0
            _gender_f = 0

        elif _client_type == 1:  # juridic
            _gender_m = 0
            _gender_f = 0

        elif _client_gender == 'male':
            _gender_m = 1
            _gender_f = 0

        elif _client_gender == 'female':
            _gender_m = 0
            _gender_f = 1

        else:
            try:
                if len(_client_name) < 2:
                    _gender_m = 0
                    _gender_f = 0
                elif _client_name[-2:].upper() == 'ИЧ':
                    _gender_m = 1
                    _gender_f = 0
                elif _client_name[-4:].upper() == 'ОГЛЫ':
                    _gender_m = 1
                    _gender_f = 0
                elif _client_name[-2:].upper() == 'НА':
                    _gender_m = 0
                    _gender_f = 1
                elif _client_name[-4:].upper() == 'КЫЗЫ':
                    _gender_m = 0
                    _gender_f = 1
                else:
                    _gender_m = 0
                    _gender_f = 0
            except:
                _gender_m = 0
                _gender_f = 0

        return [_gender_m, _gender_f]

    def transform_gender(self):
        """
        Gets values in dummy columns 'gender_m' and 'gender_f' from columns 'client_type', 'client_name' and 'client_gender'.

        :returns: None.
        """
        self._df['gender_m'], self._df['gender_f'] = zip(
            *self._df[['client_type', 'client_name', 'client_gender']].apply(self._gender, axis=1).to_frame()[0])

    @staticmethod
    def _age_get(datebirth_datestart):
        _client_date_birth = datebirth_datestart[0]
        _p_date_start = datebirth_datestart[1]
        _age = None
        if _client_date_birth > datetime.datetime.now():
            _age = None
        elif _client_date_birth.year < datetime.datetime.now().year - 120:
            _age = None
        elif _client_date_birth > _p_date_start:
            _age = None
        else:
            _age = (_p_date_start - _client_date_birth).days // 365.25
        return _age

    def age_get(self):
        """
        Gets values of age in column 'driver_minage' from columns 'client_date_birth' and 'p_date_start'.

        :returns: None.
        """
        self._df['driver_minage'] = self._df[['client_date_birth', 'p_date_start']].apply(self._age_get, axis=1)

    @staticmethod
    def _age(age, age_max):
        if pd.isnull(age):
            age = None
        elif age < 18:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def transform_age(self, age_max=70):
        """
        Transforms values of drivers' minimum age in column 'driver_minage' with values over 'age_max' grouped.

        :param age_max: Maximum value of drivers' age, bigger values will be grouped (70 by default).
        :returns: None.
        """
        self._df['driver_minage'] = self._df['driver_minage'].apply(self._age, args=(age_max,))

    @staticmethod
    def _age_gender(age_gender):
        _age = age_gender[0]
        _gender = age_gender[1]
        if _gender == 0:
            _age = 18
        return _age

    def transform_age_gender(self):
        """
        Gets intersections of drivers' minimum age and gender in columns 'driver_minage_m' and 'driver_minage_f' from
        columns 'driver_minage', 'gender_m' and 'gender_f'.

        :returns: None.
        """
        self._df['driver_minage_m'] = self._df[['driver_minage', 'gender_m']].apply(_age_gender, axis=1)
        self._df['driver_minage_f'] = self._df[['driver_minage', 'gender_f']].apply(_age_gender, axis=1)

    @staticmethod
    def _exp(exp, exp_max):
        if pd.isnull(exp):
            exp = None
        elif exp < 0:
            exp = None
        elif exp > exp_max:
            exp = exp_max
        return exp

    def transform_exp(self, exp_max=52):
        """
        Transforms values of drivers' minimum experience in column 'driver_minexp' with values over 'exp_max' grouped.

        :param exp_max: Maximum value of drivers' experience, bigger values will be grouped (52 by default).
        :returns: None.
        """
        self._df['driver_minexp'] = self._df['driver_minexp'].apply(self._exp, args=(exp_max,))

    def transform_age_exp_18(self):
        """
        Transforms records with difference between drivers' minimum age and minimum experience less then 18 years,
        sets drivers' minimum experience equal to drivers' minimum age minus 18 years.

        :returns: Number of records modified.
        """
        n = len(self._df.loc[(self._df['driver_minage'] - self._df['driver_minexp']) < 18])
        self._df['driver_minexp'].loc[(self._df['driver_minage'] - self._df['driver_minexp']) < 18] = self._df[
                                                                                                          'driver_minage'] - 18
        return n

    @staticmethod
    def _name_get(client_name):
        _tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            _name = _tokenize_re.findall(str(client_name))[1].upper()
            return _name
        except Exception:
            return 'ERROR'

    def transform_name_check(self, names_list):
        """
        Checks if clients' first names from column 'client_name' are in 'white list',
        strings in column 'client_name' should concatenate surname, first name and second name.

        :param names_list: A list of 'good' clients' first names.
        :returns: None.
        """
        self._df['client_name_check'] = 1 * self._df['client_name'].apply(self._name_get).isin(names_list)

    # ---------------------------------------------------
    # Vehicle data methods
    # ---------------------------------------------------

    @staticmethod
    def _power(power, power_min, power_max, power_group):
        if pd.isnull(power):
            power = None
        elif power < power_min:
            power = power_min
        elif power > power_max:
            power = power_max
        else:
            power = round(power / power_group, 0)
        return power

    def transform_veh_power(self, power_min=10, power_max=500, power_group=10):
        """
        Transforms values of vehicles' power in column 'vehocle_power' with values under 'power_min' and over 'power_max'
        grouped, values between 'power_min' and 'power_max' are grouped with group size 'power_group'.

        :param power_min: Minimum value of vehicles' power, lower values will be grouped (10 by default).
        :param power_max: Maximum value of vehicles' power, bigger values will be grouped (500 by default).
        :param power_group: Values of vehicles' power are divided by this parameter, rounded to integers.
        :returns: None.
        """
        self._df['vehicle_power'] = self._df['vehicle_power'].apply(self._power,
                                                                    args=(power_min, power_max, power_group,))

    @staticmethod
    def _veh_age_get(issueyear_datestart):
        _vehicle_issue_year = issueyear_datestart[0]
        _p_date_start = issueyear_datestart[1]
        _veh_age = None
        if _vehicle_issue_year > datetime.datetime.now().year:
            _veh_age = None
        elif _vehicle_issue_year < datetime.datetime.now().year - 70:
            _veh_age = None
        elif _vehicle_issue_year > _p_date_start.year:
            _veh_age = None
        else:
            _veh_age = _p_date_start.year - _vehicle_issue_year
        return _veh_age

    def veh_age_get(self):
        """
        Gets values of vehicles' age in column 'vehicle_age' from columns 'vehicle_issue year' and 'p_date_start'.

        :returns: None.
        """
        self._df['vehicle_age'] = self._df[['vehicle_issue_year', 'p_date_start']].apply(self._veh_age_get, axis=1)

    @staticmethod
    def _veh_age(age, age_max):
        if pd.isnull(age):
            age = None
        elif age < 0:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def transform_veh_age(self, veh_age_max=25):
        """
        Transforms values of vehicles' age in column 'vehicle_age' with values over 'veh_age_max' grouped.

        :param veh_age_max: Maximum value of vehicles' age, bigger values will be grouped (25 by default).
        :returns: None.
        """
        self._df['vehicle_age'] = self._df['vehicle_age'].apply(self._veh_age, args=(veh_age_max,))

    def transform_veh_type_sort_freq(self):
        """
        Gets sorted by claims' frequency vehicles' types in column 'vehicle_type_freq' from columns
        'vehicle_type' and 'p_claims_count_adj'.

        :returns: Dict of sorted vehicles' types.
        """
        self._df['count'] = 1

        _veh_type_freq = self._df.groupby(['vehicle_type']).sum()[['p_claims_count_adj', 'count']]

        _veh_type_freq['freq'] = _veh_type_freq['p_claims_count_adj'] / _veh_type_freq['count']

        _keys = []
        _values = []
        for i in enumerate(_veh_type_freq.sort_values('freq', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))

        _veh_type_freq_dict = dict(zip(_keys, _values))

        self._df['vehicle_type_freq'] = self._df['vehicle_type'].map(_veh_type_freq_dict)

        return _veh_type_freq_dict

    def transform_veh_type_sort_ac(self):
        """
        Gets sorted by claims' average sum vehicles' types in column 'vehicle_type_ac' from columns
        'vehicle_type', 'p_claims_sum_infl' and 'p_claims_count_adj'.

        :returns: Dict of sorted vehicles' types.
        """
        _veh_type_ac = self._df.groupby(['vehicle_type']).sum()[['p_claims_sum_infl', 'p_claims_count_adj']]

        _veh_type_ac['avg_claim'] = _veh_type_ac['p_claims_sum_infl'] / _veh_type_ac['p_claims_count_adj']

        _keys = []
        _values = []
        for i in enumerate(_veh_type_ac.sort_values('avg_claim', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))

        _veh_type_ac_dict = dict(zip(_keys, _values))

        self._df['veh_type_ac'] = self._df['vehicle_type'].map(_veh_type_ac_dict)

        return _veh_type_ac_dict

    # ---------------------------------------------------
    # Region data methods
    # ---------------------------------------------------

    @staticmethod
    def _region_get(kladr):
        if pd.isnull(kladr):
            _region_num = None
        else:
            _region_num = kladr[0:2]

        try:
            _region_num = int(_region_num)
        except Exception:
            _region_num = None

        return _region_num

    def region_get(self):
        """
        Gets regions' numbers in column 'region_num' from column 'kladr'.

        :return: None.
        """
        self._df['region_num'] = self._df['kladr'].apply(self._region_get)

    def region_useless(self, size_min=1000):
        """
        Checks the amount of data in regions.

        :param size_min: Minimum allowed number of records for each region (1000 by default).
        :returns: List of regions with few data.
        """
        if 'region_num' not in list(self._df.columns):
            self.region_get()
        _region_size = pd.DataFrame(self._df.groupby('region_num').size().reset_index(name='region_size'))
        return list(_region_size['region_num'].loc[_region_size['region_size'] < size_min])

    def transform_region_useless_group(self, size_min=1000):
        """
        Groups all regions with few data to one group with number = 0.

        :param size_min: Minimum allowed number of records for each region (1000 by default).
        :returns: None.
        """
        if 'region_num' not in list(self._df.columns):
            self.region_get()
        self._df.loc[self._df['region_num'].isin(self.region_useless(size_min)), 'region_num'] = 0

    def transform_region_sort_freq(self):
        """
        Gets sorted by claims' frequency regions in column 'region_freq' from columns
        'region_num' and 'p_claims_count_adj'.

        :returns: Dict of sorted regions.
        """
        self._df['count'] = 1

        _region_freq = self._df.groupby(['region_num']).sum()[['p_claims_count_adj', 'count']]

        _region_freq['freq'] = _region_freq['p_claims_count_adj'] / _region_freq['count']

        _keys = []
        _values = []
        for i in enumerate(_region_freq.sort_values('freq', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))

        _region_freq_dict = dict(zip(_keys, _values))

        self._df['region_freq'] = self._df['region_num'].map(_region_freq_dict)

        return _region_freq_dict

    def transform_region_sort_ac(self):
        """
        Gets sorted by claims' average sum regions in column 'region_ac' from columns
        'region_num', 'p_claims_sum_infl' and 'p_claims_count_adj'.

        :returns: Dict of sorted regions.
        """
        _region_ac = self._df.groupby(['region_num']).sum()[['p_claims_sum_infl', 'p_claims_count_adj']]

        _region_ac['avg_claim'] = _region_ac['p_claims_sum_infl'] / _region_ac['p_claims_count_adj']

        _keys = []
        _values = []
        for i in enumerate(_region_ac.sort_values('avg_claim', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))

        _region_ac_dict = dict(zip(_keys, _values))

        self._df['region_ac'] = self._df['region_num'].map(_region_ac_dict)

        return _region_ac_dict

    # ---------------------------------------------------
    # Other data methods
    # ---------------------------------------------------

    def polynomizer(self, column, n=2):
        """
        Gets polynomial of feature.

        :param column: Feature's column name.
        :param n: Polinomial's degree.
        :returns: None.
        """
        if column in list(self._df.columns):
            for i in range(2, n + 1):
                self._df[column + '_' + str(i)] = self._df[column] ** i

    def get_dummies(self, columns):
        """
        Gets dummy columns of the features.

        :param columns: List of columns to transforme.
        :returns: None.
        """
        self._df = pd.get_dummies(self._df, columns=columns)


