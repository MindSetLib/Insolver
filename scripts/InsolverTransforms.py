import pandas as pd
import re
import datetime

from scripts.InsolverMain import InsolverMain


class InsolverTransforms(InsolverMain):
    """
    Class to compose transforms.
    Each transform must have the priority param.
    Priority = 0: transforms witch get values from other (TransformAgeGetFromBirthday, TransformRegionGetFromKladr, ets).
    Priority = 1: main transforms of values (TransformAge, TransformVehPower, ets).
    Priority = 2: transforms witch get intersections of features (TransformAgeGender, ets);
        transforms witch sort values (TransformRegionSortFreq, ets).
    Priority = 3: transforms witch get functions of values (TransformPolinomizer, ets).

    :param df: InsolverDataFrame to transform.
    :param transforms: List of transforms to be done.
    :returns: Transformed InsolverDataFrame.
    """
    def __init__(self, df, transforms):
        self._is_frame = False
        if isinstance(df, pd.DataFrame):
            self._df = df
            self._is_frame = True
            if isinstance(transforms, list):
                self.transforms = transforms

    def get_pd(self, transforms=None, columns=None):
        """
        Gets transformed data as InsolverDataFrame.

        :param transforms: List of transforms to be done.
        :param columns: Columns of dataframe to get.
        :returns: Transformed data as InsolverDataFrame.
        """
        if self._is_frame is None:
            return None

        if transforms is not None:
            if isinstance(transforms, list):
                self.transforms = transforms

        for t in self.transforms:
            if t.priority == 0:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        for t in self.transforms:
            if t.priority == 1:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        for t in self.transforms:
            if t.priority == 2:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        for t in self.transforms:
            if t.priority == 3:
                self._df = t(self._df)
                print(f"Transformation '{type(t).__name__}' is done")

        if columns is None:
            columns = self._df.columns

        return self._df[columns].copy()


# ---------------------------------------------------
# Person data methods
# ---------------------------------------------------


class TransformGenderGetFromName:
    """
    Gets clients' genders in from russian second names.

    :param column_name: Column in InsolverDataFrame with clients' names, type is string.
    :param column_gender: Column in InsolverDataFrame for clients' genders, type is string.
    :param dict_gender: Dict for return values, {'male':'male', 'female':'female'} by default.
    """
    def __init__(self, column_name, column_gender, dict_gender={'male':'male', 'female':'female'}):
        self.priority = 0
        self.column_name = column_name
        self.column_gender = column_gender
        self.dict_gender = dict_gender

    @staticmethod
    def _gender(_client_name, _dict_gender):
        if pd.isnull(_client_name):
            _gender = None
        elif len(_client_name) < 2:
            _gender = None
        elif _client_name.upper().endswith(('ИЧ', 'ОГЛЫ')):
            _gender = _dict_gender['male']
        elif _client_name.upper().endswith(('НА', 'КЫЗЫ')):
            _gender = _dict_gender['female']
        else:
            _gender = None
        return _gender

    def __call__(self, df):
        df[self.column_gender] = df[self.column_name].apply(self._gender, args=(self.dict_gender,))
        return df


class TransformAgeGetFromBirthday:
    """
    Gets clients' ages from birth dates and policies' start dates.

    :param column_date_birth: Column in InsolverDataFrame with clients' birth dates, type is date.
    :param column_date_start: Column in InsolverDataFrame with policies' start dates, type is date.
    :param column_age: Column in InsolverDataFrame for clients' ages, type is integer.
    """
    def __init__(self, column_date_birth, column_date_start, column_age):
        self.priority = 0
        self.column_date_birth = column_date_birth
        self.column_date_start = column_date_start
        self.column_age = column_age

    @staticmethod
    def _age_get(_datebirth_datestart):
        _date_birth = _datebirth_datestart[0]
        _date_start = _datebirth_datestart[1]
        if pd.isnull(_date_birth):
            _age = None
        elif pd.isnull(_date_start):
            _age = None
        if _date_birth > datetime.datetime.now():
            _age = None
        elif _date_birth.year < datetime.datetime.now().year - 120:
            _age = None
        elif _date_birth > _date_start:
            _age = None
        else:
            _age = int((_date_start - _date_birth).days // 365.25)
        return _age

    def __call__(self, df):
        df[self.column_age] = df[[self.column_date_birth, self.column_date_start]].apply(self._age_get, axis=1)
        return df


class TransformAge:
    """
    Transforms values of drivers' minimum ages.
    Values under 'age_min' are invalid.
    Values over 'age_max' will be grouped.

    :param column_driver_minage: Column in InsolverDataFrame with drivers' minimum ages, type is integer.
    :param age_min: Minimum value of drivers' ages, lower values are invalid, type is integer, 18 by default.
    :param age_max: Maximum value of drivers' ages, bigger values will be grouped, type is integer, 70 by default.
    """
    def __init__(self, column_driver_minage, age_min=18, age_max=70):
        self.priority = 1
        self.column_driver_minage = column_driver_minage
        self.age_min = age_min
        self.age_max = age_max

    @staticmethod
    def _age(_age, _age_min, _age_max):
        if pd.isnull(_age):
            _age = None
        elif _age < _age_min:
            _age = None
        elif _age > _age_max:
            _age = _age_max
        return _age

    def __call__(self, df):
        df[self.column_driver_minage] = df[self.column_driver_minage].apply(self._age, args=(self.age_min, self.age_max,))
        return df


class TransformAgeGender:
    """
    Gets intersections of drivers' minimum ages and genders.

    :param column_age: Column in InsolverDataFrame with clients' ages, type is integer.
    :param column_gender: Column in InsolverDataFrame with clients' genders.
    :param column_age_m: Column in InsolverDataFrame for males' ages, for females default value is applied, type is integer.
    :param column_age_f: Column in InsolverDataFrame for females' ages, for males default value is applied, type is integer.
    :param age_default: Default value of the age, type is integer, 18 by default.
    :param dict_gender: Dict for genders' values, {'male':'male', 'female':'female'} by default.
    """
    def __init__(self, column_age, column_gender, column_age_m, column_age_f, age_default=18,
                 dict_gender={'male':'male', 'female':'female'}):
        self.priority = 2
        self.column_age = column_age
        self.column_gender = column_gender
        self.column_age_m = column_age_m
        self.column_age_f = column_age_f
        self.age_default = age_default
        self.dict_gender = dict_gender

    @staticmethod
    def _age_gender(_age_gender, _age_default, _dict_gender):
        _age = _age_gender[0]
        _gender = _age_gender[1]
        if pd.isnull(_age):
            _age_m = None
            _age_f = None
        elif pd.isnull(_gender):
            _age_m = None
            _age_f = None
        elif _gender == _dict_gender['male']
            _age_m = _age
            _age_f = _age_default
        elif _gender == _dict_gender['female']
            _age_m = _age_default
            _age_f = _age
        else:
            _age_m = None
            _age_f = None
        return [_age_m, _age_f]

    def __call__(self, df):
        df[self.column_age_m], df[self.column_age_f] = zip(*df[[self.column_age,self.column_gender]].apply(
            self._age_gender, axis=1, args=(self.age_default, self.dict_gender)).to_frame()[0])
        return df


class TransformExp:
    """
    Transforms values of drivers' minimum experiences with values over 'exp_max' grouped.

    :param column_driver_minexp: Column in InsolverDataFrame with drivers' minimum experiences, type is integer.
    :param exp_max: Maximum value of drivers' experiences, bigger values will be grouped, type is integer, 52 by default.
    """
    def __init__(self, column_driver_minexp, exp_max=70):
        self.priority = 1
        self.column_driver_minexp = column_driver_minexp
        self.exp_max = exp_max

    @staticmethod
    def _exp(_exp, _exp_max):
        if pd.isnull(_exp):
            _exp = None
        elif _exp < 0:
            _exp = None
        elif _exp > _exp_max:
            _exp = _exp_max
        return _exp

    def __call__(self, df):
        df[self.column_driver_minexp] = df[self.column_driver_minexp].apply(self._exp, args=(self.exp_max,))
        return df


class TransformAgeExp18:
    """
    Transforms records with difference between drivers' minimum age and minimum experience less then 18 years,
    sets drivers' minimum experience equal to drivers' minimum age minus 18 years.
    """
    def __init__(self, column_driver_minage, column_driver_minexp, diff_min=18):
        self.priority = 2
        self.column_driver_minage = column_driver_minage
        self.column_driver_minexp = column_driver_minexp
        self.diff_min = diff_min

    def __call__(self, df):
        self.num_errors = len(df.loc[(df[self.column_driver_minage] - df[self.column_driver_minexp]) < self.diff_min])
        df[self.column_driver_minexp].loc[(df[self.column_driver_minage] - df[self.column_driver_minexp])
                                          < self.diff_min] = df[self.column_driver_minage] - self.diff_min
        return df


class TransformNameCheck:
    """
    Checks if clients' first names from column 'client_name' are in 'white list',
    strings in column 'client_name' should concatenate surname, first name and second name.

    :param names_list: A list of 'good' clients' first names.
    """
    def __init__(self, names_list):
        self.priority = 1
        self.names_list = names_list

    @staticmethod
    def _name_get(_client_name):
        _tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            _name = _tokenize_re.findall(str(_client_name))[1].upper()
            return _name
        except Exception:
            return 'ERROR'

    def __call__(self, df, names_list=None):
        if names_list is not None:
            self.names_list = names_list
        df['client_name_check'] = 1 * df['client_name'].apply(self._name_get).isin(self.names_list)
        return df


# ---------------------------------------------------
# Vehicle data methods
# ---------------------------------------------------


class TransformVehPower:
    """
    Transforms values of vehicles' power in column 'vehocle_power' with values under 'power_min' and over 'power_max'
    grouped, values between 'power_min' and 'power_max' are grouped with group size 'power_group'.

    :param power_min: Minimum value of vehicles' power, lower values will be grouped (10 by default).
    :param power_max: Maximum value of vehicles' power, bigger values will be grouped (500 by default).
    :param power_group: Values of vehicles' power are divided by this parameter, rounded to integers.
    """
    def __init__(self, power_min=10, power_max=500, power_group=10):
        self.priority = 1
        self.power_min = power_min
        self.power_max = power_max
        self.power_group = power_group

    @staticmethod
    def _power(_power, _power_min, _power_max, _power_group):
        if pd.isnull(_power):
            _power = None
        elif _power < _power_min:
            _power = _power_min
        elif _power > _power_max:
            _power = _power_max
        else:
            _power = round(_power / _power_group, 0)
        return _power

    def __call__(self, df, power_min=None, power_max=None, power_group=None):
        if power_min is not None:
            self.power_min = power_min
        if power_max is not None:
            self.power_max = power_max
        if power_group is not None:
            self.power_group = power_group
        df['vehicle_power'] = df['vehicle_power'].apply(self._power, args=(self.power_min, self.power_max, self.power_group,))
        return df


class VehAgeGet:
    """
    Gets values of vehicles' age in column 'vehicle_age' from columns 'vehicle_issue year' and 'p_date_start'.
    """
    def __init__(self):
        self._apply = True
        self.priority = 0

    @staticmethod
    def _veh_age_get(_issueyear_datestart):
        _vehicle_issue_year = _issueyear_datestart[0]
        _p_date_start = _issueyear_datestart[1]
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


class TransformVehAge:
    """
    Transforms values of vehicles' age in column 'vehicle_age' with values over 'veh_age_max' grouped.

    :param veh_age_max: Maximum value of vehicles' age, bigger values will be grouped (25 by default).
    """
    def __init__(self, veh_age_max=25):
        self.priority = 1
        self.veh_age_max = veh_age_max

    @staticmethod
    def _veh_age(_age, _age_max):
        if pd.isnull(_age):
            _age = None
        elif _age < 0:
            _age = None
        elif _age > _age_max:
            _age = _age_max
        return _age

    def __call__(self, df, veh_age_max=None):
        if veh_age_max is not None:
            self.veh_age_max = veh_age_max
        df['vehicle_age'] = df['vehicle_age'].apply(self._veh_age, args=(self.veh_age_max,))
        return df


class TransformVehTypeSortFreq:
    """
    Gets sorted by claims' frequency vehicles' types in column 'vehicle_type_freq' from columns
    'vehicle_type' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self.priority = 2
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


class TransformVehTypeSortAC:
    """
    Gets sorted by claims' average sum vehicles' types in column 'vehicle_type_ac' from columns
    'vehicle_type', 'p_claims_sum_infl' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self.priority = 2
        self.veh_type_ac_dict = {}

    def __call__(self, df):
        self._veh_type_ac = df.groupby(['vehicle_type']).sum()[['p_claims_sum_infl', 'p_claims_count']]
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


class RegionGet:
    """
    Gets regions' numbers in column 'region_num' from column 'kladr'.
    """
    def __init__(self):
        self.priority = 0

    @staticmethod
    def _region_get(_kladr):
        if pd.isnull(_kladr):
            _region_num = None
        else:
            _region_num = _kladr[0:2]

        try:
            _region_num = int(_region_num)
        except Exception:
            _region_num = None

        return _region_num

    def __call__(self, df):
        df['region_num'] = df['kladr'].apply(self._region_get)
        return df


class TransformRegionUselessGroup:
    """
    Groups all regions with few data to one group with number = 0.

    :param size_min: Minimum allowed number of records for each region (1000 by default).
    """
    def __init__(self, size_min=1000):
        self.priority = 1
        self.size_min = size_min
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
            self.size_min = size_min
        df.loc[df['region_num'].isin(self.region_useless_get(df, self.size_min)), 'region_num'] = 0
        return df


class TransformRegionSortFreq:
    """
    Gets sorted by claims' frequency regions in column 'region_freq' from columns
    'region_num' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self.priority = 2
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


class TransformRegionSortAC:
    """
    Gets sorted by claims' average sum regions in column 'region_ac' from columns
    'region_num', 'p_claims_sum_infl' and 'p_claims_count_adj'.
    """
    def __init__(self):
        self.priority = 2
        self.region_ac_dict = {}

    def __call__(self, df):
        self._region_ac = df.groupby(['region_num']).sum()[['p_claims_sum_infl', 'p_claims_count']]
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


class TransformMapValues:
    """
    Transforms values in 'column' according to the 'dictionary'.

    :param column: The column to map.
    :param dict: The dictionary for mapping.
    """
    def __init__(self, column, dictionary):
        self.priority = 1
        self.column = column
        self.dict = dictionary

    def __call__(self, df):
        df[self.column] = df[self.column].map(self.dictionary)
        return df


class TransformPolynomizer:
    """
    Gets polynomial of feature.

    :param column: Feature's column name.
    :param n: Polinomial's degree.
    """
    def __init__(self, column, n=2):
        self.priority = 3
        self.column = column
        self.n = n

    def __call__(self, df, column=None, n=None):
        if column is not None:
            self.column = column
        if n is not None:
            self.n = n
        if self.column in list(df.columns):
            for i in range(2, self.n + 1):
                df[self.column + '_' + str(i)] = df[self.column] ** i
        return df


class TransformGetDummies:
    """
    Gets dummy columns of the features.

    :param column: A columns to transform.
    """
    def __init__(self, column):
        self.priority = 3
        self.column = column

    def __call__(self, df, column=None):
        if column is not None:
            self.column = column
        df = pd.get_dummies(df, columns=self.column)
        return df
