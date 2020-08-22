import pandas as pd
import re
import datetime

from scripts.InsolverMain import InsolverMain


class InsolverTransforms(InsolverMain):
    """
    Class to compose transforms.
    'transforms_all' represents the list of all available transforms with priorities.

    :param df: Pandas' DataFrame to transform.
    :param transforms: List of transforms to be done.
    :returns: Transformed data as pandas' DataFrame.
    """
    def __init__(self, df, transforms):
        self._is_frame = False
        if type(df) == pd.DataFrame:
            self._df = df
            self._is_frame = True
            if type(transforms) == list:
                self.transforms = transforms

    transforms_all = {
        'prior_zero': [
            'age_get',
            'veh_age_get',
            'region_get',
        ],
        'prior_first': [
            'transform_client_type',
            'transform_gender',
            'transform_age',
            'transform_exp',
            'transform_name_check',
            'transform_veh_power',
            'transform_veh_age',
            'transform_region_useless_group',
        ],
        'prior_second': [
            'transform_age_gender',
            'transform_age_exp_18',
            'transform_veh_type_sort_freq',
            'transform_veh_type_sort_ac',
            'transform_region_sort_freq',
            'transform_region_sort_ac',
        ],
        'prior_third': [
            'polynomizer',
        ]
    }

    def get_pd(self, transforms=None, columns=None):
        """
        Gets transformed data as pandas' DataFrame.

        :param transforms: List of transforms to be done.
        :param columns: Columns of dataframe to get.
        :returns: Transformed data as pandas' DataFrame.
        """
        if self._is_frame is None:
            return None

        if transforms is not None:
            if type(transforms) == list:
                self.transforms = transforms

        for idx, t in enumerate(self.transforms):
            if type(t).__name__ in self.transforms_all['prior_zero']:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        for idx, t in enumerate(self.transforms):
            if type(t).__name__ in self.transforms_all['prior_first']:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        for idx, t in enumerate(self.transforms):
            if type(t).__name__ in self.transforms_all['prior_second']:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        for idx, t in enumerate(self.transforms):
            if type(t).__name__ in self.transforms_all['prior_third']:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        if columns is None:
            columns = self._df.columns

        return self._df[columns]


# ---------------------------------------------------
# Person data methods
# ---------------------------------------------------


class transform_client_type:
    """
    Transforms values in column 'client_type' from {'person','company'} to {0,1}.
    """
    def __init__(self):
        self._apply = True

    _client_type_dict = {
        'person': float(0),
        'company': float(1),
        '0': float(0),
        '1': float(1),
        0: float(0),
        1: float(1)
    }

    def __call__(self, df):
        df['client_type'] = df['client_type'].map(self._client_type_dict)
        return df


class transform_gender:
    """
    Gets values in dummy columns 'gender_m' and 'gender_f' from columns 'client_type', 'client_name' and 'client_gender'.
    """
    def __init__(self):
        self._apply = True

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

    def __call__(self, df):
        df['gender_m'], df['gender_f'] = zip(
            *df[['client_type', 'client_name', 'client_gender']].apply(self._gender, axis=1).to_frame()[0])
        return df


class age_get:
    """
    Gets values of age in column 'driver_minage' from columns 'client_date_birth' and 'p_date_start'.
    """
    def __init__(self):
        self._apply = True

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

    def __call__(self, df):
        df['driver_minage'] = df[['client_date_birth', 'p_date_start']].apply(self._age_get, axis=1)
        return df


class transform_age:
    """
    Transforms values of drivers' minimum age in column 'driver_minage' with values over 'age_max' grouped.

    :param age_max: Maximum value of drivers' age, bigger values will be grouped (70 by default).
    """
    def __init__(self, age_max=70):
        self._apply = True
        self._age_max = age_max

    @staticmethod
    def _age(age, age_max):
        if pd.isnull(age):
            age = None
        elif age < 18:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def __call__(self, df, age_max=None):
        if age_max is not None:
            self._age_max = age_max
        df['driver_minage'] = df['driver_minage'].apply(self._age, args=(self._age_max,))
        return df


class transform_age_gender:
    """
    Gets intersections of drivers' minimum age and gender in columns 'driver_minage_m' and 'driver_minage_f' from
    columns 'driver_minage', 'gender_m' and 'gender_f'.
    """
    def __init__(self):
        self._apply = True

    @staticmethod
    def _age_gender(age_gender):
        _age = age_gender[0]
        _gender = age_gender[1]
        if _gender == 0:  # male
            _driver_minage_m = _age
            _driver_minage_f = 18
        elif _gender == 1:  # female
            _driver_minage_m = 18
            _driver_minage_f = _age
        else:
            _driver_minage_m = 18
            _driver_minage_f = 18
        return [_driver_minage_m, _driver_minage_f]

    def __call__(self, df):
        df['driver_minage_m'], df['driver_minage_f'] = zip(
            *df[['driver_minage','Gender']].apply(self._age_gender, axis=1).to_frame()[0])
        return df


class transform_exp:
    """
    Transforms values of drivers' minimum experience in column 'driver_minexp' with values over 'exp_max' grouped.

    :param exp_max: Maximum value of drivers' experience, bigger values will be grouped (52 by default).
    """
    def __init__(self, exp_max=70):
        self._apply = True
        self._exp_max = exp_max

    @staticmethod
    def _exp(exp, exp_max):
        if pd.isnull(exp):
            exp = None
        elif exp < 0:
            exp = None
        elif exp > exp_max:
            exp = exp_max
        return exp

    def __call__(self, df, exp_max=None):
        if exp_max is not None:
            self._exp_max = exp_max
        df['driver_minexp'] = df['driver_minexp'].apply(self._exp, args=(self._exp_max,))
        return df


class transform_age_exp_18:
    """
    Transforms records with difference between drivers' minimum age and minimum experience less then 18 years,
    sets drivers' minimum experience equal to drivers' minimum age minus 18 years.
    """
    def __init__(self):
        self._apply = True

    def __call__(self):
        self._n = len(df.loc[(df['driver_minage'] - df['driver_minexp']) < 18])
        df['driver_minexp'].loc[(df['driver_minage'] - df['driver_minexp']) < 18] = df['driver_minage'] - 18
        return df


class transform_name_check:
    """
    Checks if clients' first names from column 'client_name' are in 'white list',
    strings in column 'client_name' should concatenate surname, first name and second name.

    :param names_list: A list of 'good' clients' first names.
    """
    def __init__(self, names_list):
        self._apply = True
        self._names_list = names_list

    @staticmethod
    def _name_get(client_name):
        _tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            _name = _tokenize_re.findall(str(client_name))[1].upper()
            return _name
        except Exception:
            return 'ERROR'

    def __call__(self, df, names_list=None):
        if names_list is not None:
            self._names_list = names_list
        df['client_name_check'] = 1 * df['client_name'].apply(self._name_get).isin(self._names_list)
        return df


# ---------------------------------------------------
# Vehicle data methods
# ---------------------------------------------------


class transform_veh_power:
    """
    Transforms values of vehicles' power in column 'vehocle_power' with values under 'power_min' and over 'power_max'
    grouped, values between 'power_min' and 'power_max' are grouped with group size 'power_group'.

    :param power_min: Minimum value of vehicles' power, lower values will be grouped (10 by default).
    :param power_max: Maximum value of vehicles' power, bigger values will be grouped (500 by default).
    :param power_group: Values of vehicles' power are divided by this parameter, rounded to integers.
    """
    def __init__(self, power_min=10, power_max=500, power_group=10):
        self._apply = True
        self._power_min = power_min
        self._power_max = power_max
        self._power_group = power_group

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

    def __call__(self, df, power_min=None, power_max=None, power_group=None):
        if power_min is not None:
            self._power_min = power_min
        if power_max is not None:
            self._power_max = power_max
        if power_group is not None:
            self._power_group = power_group
        df['vehicle_power'] = df['vehicle_power'].apply(self._power, args=(self._power_min, self._power_max, self._power_group,))
        return df


class veh_age_get:
    """
    Gets values of vehicles' age in column 'vehicle_age' from columns 'vehicle_issue year' and 'p_date_start'.
    """
    def __init__(self):
        self._apply = True

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

    def __call__(self, df):
        df['vehicle_age'] = df[['vehicle_issue_year', 'p_date_start']].apply(self._veh_age_get, axis=1)
        return df


class transform_veh_age:
    """
    Transforms values of vehicles' age in column 'vehicle_age' with values over 'veh_age_max' grouped.

    :param veh_age_max: Maximum value of vehicles' age, bigger values will be grouped (25 by default).
    """
    def __init__(self, veh_age_max=25):
        self._apply = True
        self._veh_age_max = veh_age_max

    @staticmethod
    def _veh_age(age, age_max):
        if pd.isnull(age):
            age = None
        elif age < 0:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def __call__(self, df, veh_age_max=None):
        if veh_age_max is not None:
            self._veh_age_max = veh_age_max
        df['vehicle_age'] = df['vehicle_age'].apply(self._veh_age, args=(self._veh_age_max,))
        return df


class transform_veh_type_sort_freq:
    """
    Gets sorted by claims' frequency vehicles' types in column 'vehicle_type_freq' from columns
    'vehicle_type' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self._apply = True
        self.veh_type_freq_dict = {}

    def __call__(self, df):
        df['count'] = 1
        self._veh_type_freq = df.groupby(['vehicle_type']).sum()[['p_claims_count_adj', 'count']]
        self._veh_type_freq['freq'] = self._veh_type_freq['p_claims_count_adj'] / self._veh_type_freq['count']
        _keys = []
        _values = []
        for i in enumerate(self._veh_type_freq.sort_values('freq', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))
        self.veh_type_freq_dict = dict(zip(_keys, _values))
        df['vehicle_type_freq'] = df['vehicle_type'].map(self.veh_type_freq_dict)
        return df


class transform_veh_type_sort_ac:
    """
    Gets sorted by claims' average sum vehicles' types in column 'vehicle_type_ac' from columns
    'vehicle_type', 'p_claims_sum_infl' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self._apply = True
        self.veh_type_ac_dict = {}

    def __call__(self, df):
        self._veh_type_ac = df.groupby(['vehicle_type']).sum()[['p_claims_sum_infl', 'p_claims_count_adj']]
        self._veh_type_ac['avg_claim'] = self._veh_type_ac['p_claims_sum_infl'] / self._veh_type_ac['p_claims_count_adj']
        _keys = []
        _values = []
        for i in enumerate(self._veh_type_ac.sort_values('avg_claim', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))
        self.veh_type_ac_dict = dict(zip(_keys, _values))
        df['veh_type_ac'] = df['vehicle_type'].map(self.veh_type_ac_dict)
        return df


# ---------------------------------------------------
# Region data methods
# ---------------------------------------------------


class region_get:
    """
    Gets regions' numbers in column 'region_num' from column 'kladr'.
    """
    def __init__(self):
        self._apply = True

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

    def __call__(self, df):
        df['region_num'] = df['kladr'].apply(self._region_get)
        return df


class transform_region_useless_group:
    """
    Groups all regions with few data to one group with number = 0.

    :param size_min: Minimum allowed number of records for each region (1000 by default).
    """
    def __init__(self, size_min=1000):
        self._apply = True
        self._size_min = size_min
        self.region_useless = {}

    def region_useless_get(self, df, size_min):
        """
        Checks the amount of data in regions.

        :param df: Dataframe to explore.
        :param size_min: Minimum allowed number of records for each region (1000 by default).
        :returns: List of regions with few data.
        """
        self._region_size = pd.DataFrame(df.groupby('region_num').size().reset_index(name='region_size'))
        self.region_useless = list(self._region_size['region_num'].loc[self._region_size['region_size'] < size_min])
        return self.region_useless

    def __call__(self, df, size_min=None):
        if size_min is not None:
            self._size_min = size_min
        df.loc[df['region_num'].isin(self.region_useless_get(df, self._size_min)), 'region_num'] = 0
        return df


class transform_region_sort_freq:
    """
    Gets sorted by claims' frequency regions in column 'region_freq' from columns
    'region_num' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self._apply = True
        self.region_freq_dict = {}

    def __call__(self, df):
        df['count'] = 1
        self._region_freq = df.groupby(['region_num']).sum()[['p_claims_count_adj', 'count']]
        self._region_freq['freq'] = self._region_freq['p_claims_count_adj'] / self._region_freq['count']
        _keys = []
        _values = []
        for i in enumerate(self._region_freq.sort_values('freq', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))
        self.region_freq_dict = dict(zip(_keys, _values))
        df['region_freq'] = df['region_num'].map(self.region_freq_dict)
        return df


class transform_region_sort_ac:
    """
    Gets sorted by claims' average sum regions in column 'region_ac' from columns
    'region_num', 'p_claims_sum_infl' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self._apply = True
        self.region_ac_dict = {}

    def __call__(self, df):
        self._region_ac = df.groupby(['region_num']).sum()[['p_claims_sum_infl', 'p_claims_count_adj']]
        self._region_ac['avg_claim'] = self._region_ac['p_claims_sum_infl'] / self._region_ac['p_claims_count_adj']
        _keys = []
        _values = []
        for i in enumerate(self._region_ac.sort_values('avg_claim', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))
        self.region_ac_dict = dict(zip(_keys, _values))
        df['region_ac'] = df['region_num'].map(self.region_ac_dict)
        return df


# ---------------------------------------------------
# Other data methods
# ---------------------------------------------------


class polynomizer:
    """
    Gets polynomial of feature.

    :param column: Feature's column name.
    :param n: Polinomial's degree.
    """
    def __init__(self, column, n=2):
        self._apply = True
        self._column = column
        self._n = n

    def __call__(self, df, column=None, n=None):
        if column is not None:
            self._column = column
        if n is not None:
            self._n = n
        if self._column in list(df.columns):
            for i in range(2, self._n + 1):
                df[self._column + '_' + str(i)] = df[self._column] ** i
        return df


class get_dummies:
    """
    Gets dummy columns of the features.

    :param column: A columns to transform.
    """
    def __init__(self, column):
        self._apply = True
        self._column = column

    def __call__(self, df, column=None):
        if column is not None:
            self._column = column
        df = pd.get_dummies(df, columns=self._column)
        return df
