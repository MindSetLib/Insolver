import pandas as pd
import re
import datetime

from scripts.InsolverDataFrame import InsolverDataFrame


class InsolverTransforms(InsolverDataFrame):
    """
    Class to compose transforms.
    Each transform must have the priority param.
    Priority = 0: transforms witch get values from other (TransformAgeGetFromBirthday, TransformRegionGetFromKladr, ets).
    Priority = 1: main transforms of values (TransformAge, TransformVehPower, ets).
    Priority = 2: transforms witch get intersections of features (TransformAgeGender, ets);
        transforms witch sort values (TransformParamSortFreq, TransformParamSortAC).
    Priority = 3: transforms witch get functions of values (TransformPolinomizer, TransformGetDummies, ets).

    :param df: InsolverDataFrame to transform.
    :param transforms: List of transforms to be done.
    :returns: Transformed InsolverDataFrame.
    """
    def __init__(self, df, transforms):
        super().__init__(df)
        if isinstance(transforms, list):
            self.transforms = transforms

    def transform(self):
        """
        Transforms data in InsolverDataFrame.

        :returns: List of transforms have been done.
        """
        _transforms_done = []

        if self._is_frame is None:
            raise NotImplementedError("No data loaded.")

        if self.transforms:

            for t in self.transforms:
                if t.priority == 0:
                    self._df = t(self._df)
                    _transforms_done.append(type(t).__name__)

            for t in self.transforms:
                if t.priority == 1:
                    self._df = t(self._df)
                    _transforms_done.append(type(t).__name__)

            for t in self.transforms:
                if t.priority == 2:
                    self._df = t(self._df)
                    _transforms_done.append(type(t).__name__)

            for t in self.transforms:
                if t.priority == 3:
                    self._df = t(self._df)
                    _transforms_done.append(type(t).__name__)

        return _transforms_done


# ---------------------------------------------------
# Person data methods
# ---------------------------------------------------


class TransformGenderGetFromName:
    """
    Gets clients' genders from russian second names.

    :param column_name: Column in InsolverDataFrame with clients' names, type is string.
    :param column_gender: Column in InsolverDataFrame for clients' genders, type is string.
    :param gender_male: Return value for male gender in InsolverDataFrame, 'male' by default.
    :param gender_female: Return value for female gender in InsolverDataFrame, 'female' by default.
    """
    def __init__(self, column_name, column_gender, gender_male='male', gender_female='female'):
        self.priority = 0
        self.column_name = column_name
        self.column_gender = column_gender
        self.gender_male = gender_male
        self.gender_female = gender_female

    @staticmethod
    def _gender(_client_name, _dict_gender, _gender_male, _gender_female):
        if pd.isnull(_client_name):
            _gender = None
        elif len(_client_name) < 2:
            _gender = None
        elif _client_name.upper().endswith(('ИЧ', 'ОГЛЫ')):
            _gender = _gender_male
        elif _client_name.upper().endswith(('НА', 'КЫЗЫ')):
            _gender = _gender_female
        else:
            _gender = None
        return _gender

    def __call__(self, df):
        df[self.column_gender] = df[self.column_name].apply(self._gender, args=(self.gender_male, self.gender_female,))
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
        elif _date_birth > datetime.datetime.now():
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
    :param age_min: Minimum value of drivers' age, lower values are invalid, type is integer, 18 by default.
    :param age_max: Maximum value of drivers' age, bigger values will be grouped, type is integer, 70 by default.
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
    :param gender_male: Value for male gender in InsolverDataFrame, 'male' by default.
    :param gender_female: Value for male gender in InsolverDataFrame, 'female' by default.
    """
    def __init__(self, column_age, column_gender, column_age_m, column_age_f, age_default=18,
                 gender_male='male', gender_female='female'):
        self.priority = 2
        self.column_age = column_age
        self.column_gender = column_gender
        self.column_age_m = column_age_m
        self.column_age_f = column_age_f
        self.age_default = age_default
        self.gender_male = gender_male
        self.gender_female = gender_female

    @staticmethod
    def _age_gender(_age_gender, _age_default, _gender_male, _gender_female):
        _age = _age_gender[0]
        _gender = _age_gender[1]
        if pd.isnull(_age):
            _age_m = None
            _age_f = None
        elif pd.isnull(_gender):
            _age_m = None
            _age_f = None
        elif _gender == _gender_male:
            _age_m = _age
            _age_f = _age_default
        elif _gender == _gender_female:
            _age_m = _age_default
            _age_f = _age
        else:
            _age_m = None
            _age_f = None
        return [_age_m, _age_f]

    def __call__(self, df):
        df[self.column_age_m], df[self.column_age_f] = zip(*df[[self.column_age, self.column_gender]].apply(
            self._age_gender, axis=1, args=(self.age_default, self.gender_male, self.gender_female)).to_frame()[0])
        return df


class TransformExp:
    """
    Transforms values of drivers' minimum experiences with values over 'exp_max' grouped.

    :param column_driver_minexp: Column in InsolverDataFrame with drivers' minimum experiences, type is integer.
    :param exp_max: Maximum value of drivers' experience, bigger values will be grouped, type is integer, 52 by default.
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


class TransformAgeExpDiff:
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
    Checks if clients' first names are in special list.
    Names should concatenate surnames, first names and second names.

    :param column_name: Column in InsolverDataFrame with clients' names, type is string.
    :param column_name_check: Column in InsolverDataFrame for bool values are first names in the list or not.
    :param names_list: The list of clients' first names, type is list with upper strings.
    """
    def __init__(self, column_name, column_name_check, names_list):
        self.priority = 1
        self.column_name = column_name
        self.column_name_check = column_name_check
        self.names_list = names_list

    @staticmethod
    def _name_get(_client_name):
        _tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            _name = _tokenize_re.findall(str(_client_name))[1].upper()
            return _name
        except Exception:
            return 'ERROR'

    def __call__(self, df):
        df[self.column_name_check] = 1 * df[self.column_name].apply(self._name_get).isin(self.names_list)
        return df


# ---------------------------------------------------
# Vehicle data methods
# ---------------------------------------------------


class TransformVehPower:
    """
    Transforms values of vehicles' powers.
    Values under 'power_min' and over 'power_max' will be grouped.
    Values between 'power_min' and 'power_max' will be grouped with step 'power_step'.

    :param column_veh_power: Column in InsolverDataFrame with vehicles' powers, type is integer or float.
    :param power_min: Minimum value of vehicles' power, lower values will be grouped, type is integer, 10 by default.
    :param power_max: Maximum value of vehicles' power, bigger values will be grouped, type is integer, 500 by default.
    :param power_step: Values of vehicles' power will be divided by this parameter, rounded to integers, 10 by default.
    """
    def __init__(self, column_veh_power, power_min=10, power_max=500, power_step=10):
        self.priority = 1
        self.column_veh_power = column_veh_power
        self.power_min = power_min
        self.power_max = power_max
        self.power_step = power_step

    @staticmethod
    def _power(_power, _power_min, _power_max, _power_step):
        if pd.isnull(_power):
            _power = None
        elif _power < _power_min:
            _power = _power_min
        elif _power > _power_max:
            _power = _power_max
        else:
            _power = int(round(_power / _power_step, 0))
        return _power

    def __call__(self, df):
        df[self.column_veh_power] = df[self.column_veh_power].apply(self._power, args=(self.power_min, self.power_max, self.power_step,))
        return df


class TransformVehAgeGetFromIssueYear:
    """
    Gets vehicles' ages from issue years and policies' start dates.

    :param column_veh_issue_year: Column in InsolverDataFrame with vehicles' issue years, type is integer.
    :param column_date_start: Column in InsolverDataFrame with policies' start dates, type is date.
    :param column_veh_age: Column in InsolverDataFrame for vehicles' ages, type is integer.
    """
    def __init__(self, column_veh_issue_year, column_date_start, column_veh_age):
        self.priority = 0
        self.column_veh_issue_year = column_veh_issue_year
        self.column_date_start = column_date_start
        self.column_veh_age = column_veh_age

    @staticmethod
    def _veh_age_get(_issueyear_datestart):
        _veh_issue_year = _issueyear_datestart[0]
        _date_start = _issueyear_datestart[1]
        _veh_age = None
        if pd.isnull(_veh_issue_year):
            _veh_age = None
        elif pd.isnull(_date_start):
            _veh_age = None
        elif _veh_issue_year > datetime.datetime.now().year:
            _veh_age = None
        elif _veh_issue_year < datetime.datetime.now().year - 90:
            _veh_age = None
        elif _veh_issue_year > _date_start.year:
            _veh_age = None
        else:
            _veh_age = _date_start.year - _veh_issue_year
        return _veh_age

    def __call__(self, df):
        df[self.column_veh_age] = df[[self.column_veh_issue_year, self.column_date_start]].apply(self._veh_age_get, axis=1)
        return df


class TransformVehAge:
    """
    Transforms values of vehicles' ages.
    Values over 'veh_age_max' will be grouped.

    :param column_veh_age: Column in InsolverDataFrame with vehicles' ages, type is integer.
    :param veh_age_max: Maximum value of vehicles' age, bigger values will be grouped, type is integer, 25 by default.
    """
    def __init__(self, column_veh_age, veh_age_max=25):
        self.priority = 1
        self.column_veh_age = column_veh_age
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

    def __call__(self, df):
        df[self.column_veh_age] = df[self.column_veh_age].apply(self._veh_age, args=(self.veh_age_max,))
        return df


# ---------------------------------------------------
# Region data methods
# ---------------------------------------------------


class TransformRegionGetFromKladr:
    """
    Gets regions' numbers from column KLADRs.

    :param column_kladr: Column in InsolverDataFrame with KLADRs, type is string.
    :param column_region_num: Column in InsolverDataFrame for regions, type is integer.
    """
    def __init__(self, column_kladr, column_region_num):
        self.priority = 0
        self.column_kladr = column_kladr
        self.column_region_num = column_region_num

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
        df[self.column_region_num] = df[self.column_kladr].apply(self._region_get)
        return df


# ---------------------------------------------------
# Sorting data methods
# ---------------------------------------------------


class TransformParamUselessGroup:
    """
    Groups all parameter's values with few data to one group.

    :param column_param: Column in InsolverDataFrame with parameter.
    :param size_min: Minimum allowed number of records for each parameter value, type is integer, 1000 by default.
    :param group_name: Name of the group for parameter's values with few data.
    """
    def __init__(self, column_param, size_min=1000, group_name=0):
        self.priority = 1
        self.column_param = column_param
        self.size_min = size_min
        self.group_name = group_name
        self.param_useless = {}

    @staticmethod
    def _param_useless_get(_df, _column_param, _size_min):
        """
        Checks the amount of data for each parameter's value.

        :param _df: InsolverDataFrame to explore.
        :param _column_param: Column in InsolverDataFrame with parameter.
        :param _size_min: Minimum allowed number of records for each parameter's value, type is integer, 1000 by default.
        :returns: List of parameter's values with few data.
        """
        _param_size = pd.DataFrame(_df.groupby(_column_param).size().reset_index(name='param_size'))
        _param_useless = list(_param_size[_column_param].loc[_param_size['param_size'] < _size_min])
        return _param_useless

    def __call__(self, df):
        self.param_useless = self._param_useless_get(df, self.column_param, self.size_min)
        df.loc[df[self.column_param].isin(self.param_useless), self.column_param] = self.group_name
        return df


class TransformParamSortFreq:
    """
    Gets sorted by claims' frequency parameter's values.

    :param column_param: Column in InsolverDataFrame with parameter.
    :param column_param_sort_freq: Column in InsolverDataFrame for sorted values of parameter, type is integer.
    :param column_policies_count: Column in InsolverDataFrame with number of policies, type is integer or float.
    :param column_claims_count: Column in InsolverDataFrame with number of claims, type is integer or float.
    """
    def __init__(self, column_param, column_param_sort_freq, column_policies_count, column_claims_count):
        self.priority = 2
        self.column_param = column_param
        self.column_param_sort_freq = column_param_sort_freq
        self.column_policies_count = column_policies_count
        self.column_claims_count = column_claims_count
        self.param_freq = pd.DataFrame
        self.param_freq_dict = {}

    def __call__(self, df):
        self.param_freq = df.groupby([self.column_param]).sum()[[self.column_claims_count, self.column_policies_count]]
        self.param_freq['freq'] = self.param_freq[self.column_claims_count] / self.param_freq[self.column_policies_count]
        _keys = []
        _values = []
        for i in enumerate(self.param_freq.sort_values('freq', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))
        self.param_freq_dict = dict(zip(_keys, _values))
        df[self.column_param_sort_freq] = df[self.column_param].map(self.param_freq_dict)
        return df


class TransformParamSortAC:
    """
    Gets sorted by claims' average sum parameter's values.

    :param column_param: Column in InsolverDataFrame with parameter.
    :param column_param_sort_ac: Column in InsolverDataFrame for sorted values of parameter, type is integer.
    :param column_claims_count: Column in InsolverDataFrame with number of claims, type is integer or float.
    :param column_claims_sum: Column in InsolverDataFrame with sum of claims, type is integer or float.
    """
    def __init__(self, column_param, column_param_sort_ac, column_claims_count, column_claims_sum):
        self.priority = 2
        self.column_param = column_param
        self.column_param_sort_ac = column_param_sort_ac
        self.column_claims_count = column_claims_count
        self.column_claims_sum = column_claims_sum
        self.param_ac = pd.DataFrame
        self.param_ac_dict = {}

    def __call__(self, df):
        self.param_ac = df.groupby([self.column_param]).sum()[[self.column_claims_sum, self.column_claims_count]]
        self.param_ac['avg_claim'] = self.param_ac[self.column_claims_sum] / self.param_ac[self.column_claims_count]
        _keys = []
        _values = []
        for i in enumerate(self.param_ac.sort_values('avg_claim', ascending=False).index.values):
            _keys.append(i[1])
            _values.append(float(i[0]))
        self.param_ac_dict = dict(zip(_keys, _values))
        df[self.column_param_sort_ac] = df[self.column_param].map(self.param_ac_dict)
        return df


# ---------------------------------------------------
# Other data methods
# ---------------------------------------------------


class TransformMapValues:
    """
    Transforms parameter's values according to the dictionary.

    :param column_param: Column in InsolverDataFrame with parameter to map.
    :param dict: The dictionary for mapping.
    """
    def __init__(self, column_param, dictionary):
        self.priority = 1
        self.column_param = column_param
        self.dictionary = dictionary

    def __call__(self, df):
        df[self.column_param] = df[self.column_param].map(self.dictionary)
        return df


class TransformPolynomizer:
    """
    Gets polynomials of parameter's values.

    :param column_param: Column in InsolverDataFrame with parameter to polinomize.
    :param n: Polinomial's degree, type is integer.
    """
    def __init__(self, column_param, n=2):
        self.priority = 3
        self.column_param = column_param
        self.n = n

    def __call__(self, df):
        for i in range(2, self.n + 1):
            _a = self.column_param + '_' + str(i)
            while _a in list(df.columns):
                _a = _a + '_'
            df[_a] = df[self.column_param] ** i
        return df


class TransformGetDummies:
    """
    Gets dummy columns of the parameter.

    :param column_param: Column in InsolverDataFrame with parameter to transform.
    """
    def __init__(self, column_param):
        self.priority = 3
        self.column_param = column_param

    def __call__(self, df):
        df = pd.get_dummies(df, columns=self.column)
        return df
