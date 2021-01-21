import re
import datetime

import numpy as np
import pandas as pd

from insolver.frame import InsolverDataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ---------------------------------------------------
# Person data methods
# ---------------------------------------------------


class TransformGenderGetFromName:
    """Gets clients' genders from theirs russian second names.

    Attributes:
        column_name (str): Column name in InsolverDataFrame containing clients' names, column type is string.
        column_gender (str): Column name in InsolverDataFrame for clients' genders.
        gender_male (str): Return value for male gender in InsolverDataFrame, 'male' by default.
        gender_female (str): Return value for female gender in InsolverDataFrame, 'female' by default.
    """
    def __init__(self, column_name, column_gender, gender_male='male', gender_female='female'):
        self.priority = 0
        self.column_name = column_name
        self.column_gender = column_gender
        self.gender_male = gender_male
        self.gender_female = gender_female

    @staticmethod
    def _gender(client_name, gender_male, gender_female):
        if pd.isnull(client_name):
            gender = None
        elif len(client_name) < 2:
            gender = None
        elif client_name.upper().endswith(('ИЧ', 'ОГЛЫ')):
            gender = gender_male
        elif client_name.upper().endswith(('НА', 'КЫЗЫ')):
            gender = gender_female
        else:
            gender = None
        return gender

    def __call__(self, df):
        df[self.column_gender] = df[self.column_name].apply(self._gender, args=(self.gender_male, self.gender_female,))
        return df


class TransformAgeGetFromBirthday:
    """Gets clients' ages in years from theirs birth dates and policies' start dates.

    Attributes:
        column_date_birth (str): Column name in InsolverDataFrame containing clients' birth dates, column type is date.
        column_date_start (str): Column name in InsolverDataFrame containing policies' start dates, column type is date.
        column_age (str): Column name in InsolverDataFrame for clients' ages in years, column type is int.
    """
    def __init__(self, column_date_birth, column_date_start, column_age):
        self.priority = 0
        self.column_date_birth = column_date_birth
        self.column_date_start = column_date_start
        self.column_age = column_age

    @staticmethod
    def _age_get(datebirth_datestart):
        date_birth = datebirth_datestart[0]
        date_start = datebirth_datestart[1]
        if pd.isnull(date_birth):
            age = None
        elif pd.isnull(date_start):
            age = None
        elif date_birth > datetime.datetime.now():
            age = None
        elif date_birth.year < datetime.datetime.now().year - 120:
            age = None
        elif date_birth > date_start:
            age = None
        else:
            age = int((date_start - date_birth).days // 365.25)
        return age

    def __call__(self, df):
        df[self.column_age] = df[[self.column_date_birth, self.column_date_start]].apply(self._age_get, axis=1)
        return df


class TransformAge:
    """Transforms values of drivers' minimum ages in years.
    Values under 'age_min' are invalid. Values over 'age_max' will be grouped.

    Attributes:
        column_driver_minage (str): Column name in InsolverDataFrame containing drivers' minimum ages in years,
            column type is integer.
        age_min (int): Minimum value of drivers' age in years, lower values are invalid, 18 by default.
        age_max (int): Maximum value of drivers' age in years, bigger values will be grouped, 70 by default.
    """
    def __init__(self, column_driver_minage, age_min=18, age_max=70):
        self.priority = 1
        self.column_driver_minage = column_driver_minage
        self.age_min = age_min
        self.age_max = age_max

    @staticmethod
    def _age(age, age_min, age_max):
        if pd.isnull(age):
            age = None
        elif age < age_min:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def __call__(self, df):
        df[self.column_driver_minage] = df[self.column_driver_minage].apply(self._age,
                                                                            args=(self.age_min, self.age_max))
        return df


class TransformAgeGender:
    """Gets intersections of drivers' minimum ages and genders.

    Attributes:
        column_age (str): Column name in InsolverDataFrame containing clients' ages in years, column type is integer.
        column_gender (str): Column name in InsolverDataFrame containing clients' genders.
        column_age_m (str): Column name in InsolverDataFrame for males' ages, for females default value is applied,
            column type is integer.
        column_age_f (str): Column name in InsolverDataFrame for females' ages, for males default value is applied,
            column type is integer.
        age_default (int): Default value of the age in years,18 by default.
        gender_male: Value for male gender in InsolverDataFrame, 'male' by default.
        gender_female: Value for male gender in InsolverDataFrame, 'female' by default.
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
    def _age_gender(age_gender, age_default, gender_male, gender_female):
        age = age_gender[0]
        gender = age_gender[1]
        if pd.isnull(age):
            age_m = None
            age_f = None
        elif pd.isnull(gender):
            age_m = None
            age_f = None
        elif gender == gender_male:
            age_m = age
            age_f = age_default
        elif gender == gender_female:
            age_m = age_default
            age_f = age
        else:
            age_m = None
            age_f = None
        return [age_m, age_f]

    def __call__(self, df):
        df[self.column_age_m], df[self.column_age_f] = zip(*df[[self.column_age, self.column_gender]].apply(
            self._age_gender, axis=1, args=(self.age_default, self.gender_male, self.gender_female)).to_frame()[0])
        return df


class TransformExp:
    """Transforms values of drivers' minimum experiences in years with values over 'exp_max' grouped.

    Attributes:
        column_driver_minexp (str): Column name in InsolverDataFrame containing drivers' minimum experiences in years,
            column type is integer.
        exp_max (int): Maximum value of drivers' experience in years, bigger values will be grouped, 52 by default.
    """
    def __init__(self, column_driver_minexp, exp_max=52):
        self.priority = 1
        self.column_driver_minexp = column_driver_minexp
        self.exp_max = exp_max

    @staticmethod
    def _exp(exp, exp_max):
        if pd.isnull(exp):
            exp = None
        elif exp < 0:
            exp = None
        elif exp > exp_max:
            exp = exp_max
        return exp

    def __call__(self, df):
        df[self.column_driver_minexp] = df[self.column_driver_minexp].apply(self._exp, args=(self.exp_max,))
        return df


class TransformAgeExpDiff:
    """Transforms records with difference between drivers' minimum age and minimum experience less then 'diff_min'
     years, sets drivers' minimum experience equal to drivers' minimum age minus 'diff_min' years.

    Attributes:
        column_driver_minage (str): Column name in InsolverDataFrame containing drivers' minimum ages in years,
            column type is integer.
        column_driver_minexp (str): Column name in InsolverDataFrame containing drivers' minimum experiences in years,
            column type is integer.
        diff_min (int): Minimum allowed difference between age and experience in years.
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
    """Checks if clients' first names are in special list.
    Names may concatenate surnames, first names and last names.

    Attributes:
        column_name (str): Column name in InsolverDataFrame containing clients' names, column type is string.
        name_full (bool): Sign if name is the concatenation of surname, first name and last name, False by default.
        column_name_check (str): Column name in InsolverDataFrame for bool values if first names are in the list or not.
        names_list (list): The list of clients' first names.
    """
    def __init__(self, column_name, column_name_check, names_list, name_full=False):
        self.priority = 1
        self.column_name = column_name
        self.name_full = name_full
        self.column_name_check = column_name_check
        self.names_list = [n.upper() for n in names_list]

    @staticmethod
    def _name_get(client_name):
        tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            name = tokenize_re.findall(str(client_name))[1].upper()
            return name
        except Exception:
            return 'ERROR'

    def __call__(self, df):
        if not self.name_full:
            df[self.column_name_check] = 1 * df[self.column_name].isin(self.names_list)
        else:
            df[self.column_name_check] = 1 * df[self.column_name].apply(self._name_get).isin(self.names_list)
        return df


# ---------------------------------------------------
# Vehicle data methods
# ---------------------------------------------------


class TransformVehPower:
    """Transforms values of vehicles' powers.
    Values under 'power_min' and over 'power_max' will be grouped.
    Values between 'power_min' and 'power_max' will be grouped with step 'power_step'.

    Attributes:
        column_veh_power (str): Column name in InsolverDataFrame containing vehicles' powers,
            column type is float.
        power_min (float): Minimum value of vehicles' power, lower values will be grouped, 10 by default.
        power_max (float): Maximum value of vehicles' power, bigger values will be grouped, 500 by default.
        power_step (int): Values of vehicles' power will be divided by this parameter, rounded to integers,
            10 by default.
    """
    def __init__(self, column_veh_power, power_min=10, power_max=500, power_step=10):
        self.priority = 1
        self.column_veh_power = column_veh_power
        self.power_min = power_min
        self.power_max = power_max
        self.power_step = power_step

    @staticmethod
    def _power(power, power_min, power_max, power_step):
        if pd.isnull(power):
            power = None
        elif power < power_min:
            power = power_min
        elif power > power_max:
            power = power_max
        else:
            power = int(round(power / power_step, 0))
        return power

    def __call__(self, df):
        df[self.column_veh_power] = df[self.column_veh_power].apply(self._power, args=(self.power_min, self.power_max,
                                                                                       self.power_step,))
        return df


class TransformVehAgeGetFromIssueYear:
    """Gets vehicles' ages in years from issue years and policies' start dates.

    Attributes:
        column_veh_issue_year (str): Column name in InsolverDataFrame containing vehicles' issue years,
            column type is integer.
        column_date_start (str): Column name in InsolverDataFrame containing policies' start dates, column type is date.
        column_veh_age (str): Column name in InsolverDataFrame for vehicles' ages in years, column type is integer.
    """
    def __init__(self, column_veh_issue_year, column_date_start, column_veh_age):
        self.priority = 0
        self.column_veh_issue_year = column_veh_issue_year
        self.column_date_start = column_date_start
        self.column_veh_age = column_veh_age

    @staticmethod
    def _veh_age_get(issueyear_datestart):
        veh_issue_year = issueyear_datestart[0]
        date_start = issueyear_datestart[1]
        if pd.isnull(veh_issue_year):
            veh_age = None
        elif pd.isnull(date_start):
            veh_age = None
        elif veh_issue_year > datetime.datetime.now().year:
            veh_age = None
        elif veh_issue_year < datetime.datetime.now().year - 90:
            veh_age = None
        elif veh_issue_year > date_start.year:
            veh_age = None
        else:
            veh_age = date_start.year - veh_issue_year
        return veh_age

    def __call__(self, df):
        df[self.column_veh_age] = df[[self.column_veh_issue_year,
                                      self.column_date_start]].apply(self._veh_age_get, axis=1)
        return df


class TransformVehAge:
    """Transforms values of vehicles' ages in years. Values over 'veh_age_max' will be grouped.

    Attributes:
        column_veh_age (str): Column name in InsolverDataFrame containing vehicles' ages in years,
            column type is integer.
        veh_age_max (int): Maximum value of vehicles' age in years, bigger values will be grouped, 25 by default.
    """
    def __init__(self, column_veh_age, veh_age_max=25):
        self.priority = 1
        self.column_veh_age = column_veh_age
        self.veh_age_max = veh_age_max

    @staticmethod
    def _veh_age(age, age_max):
        if pd.isnull(age):
            age = None
        elif age < 0:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def __call__(self, df):
        df[self.column_veh_age] = df[self.column_veh_age].apply(self._veh_age, args=(self.veh_age_max,))
        return df


# ---------------------------------------------------
# Region data methods
# ---------------------------------------------------


class TransformRegionGetFromKladr:
    """Gets regions' numbers from KLADRs.

    Attributes:
        column_kladr (str): Column name in InsolverDataFrame containing KLADRs, column type is string.
        column_region_num (str): Column name in InsolverDataFrame for regions' numbers, column type is integer.
    """
    def __init__(self, column_kladr, column_region_num):
        self.priority = 0
        self.column_kladr = column_kladr
        self.column_region_num = column_region_num

    @staticmethod
    def _region_get(kladr):
        if pd.isnull(kladr):
            region_num = None
        else:
            region_num = kladr[0:2]

        try:
            region_num = int(region_num)
        except Exception:
            region_num = None

        return region_num

    def __call__(self, df):
        df[self.column_region_num] = df[self.column_kladr].apply(self._region_get)
        return df


# ---------------------------------------------------
# Sorting data methods
# ---------------------------------------------------


class TransformParamUselessGroup:
    """Groups all parameter's values with few data to one group.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter.
        size_min (int): Minimum allowed number of records for each parameter value, 1000 by default.
        group_name: Name of the group for parameter's values with few data.
        inference (bool): Sign if the transformation is used for inference, False by default.
        param_useless (list): The list of useless values of the parameter, for inference only.
    """
    def __init__(self, column_param, size_min=1000, group_name=0, inference=False, param_useless=None):
        self.priority = 1
        self.column_param = column_param
        self.size_min = size_min
        self.group_name = group_name
        self.inference = inference
        if inference:
            if param_useless is None:
                raise NotImplementedError("'param_useless' should contain the list of useless values.")
            self.param_useless = param_useless
        else:
            self.param_useless = {}

    @staticmethod
    def _param_useless_get(df, column_param, size_min):
        """Checks the amount of data for each parameter's value.

        Args:
            df: InsolverDataFrame to explore.
            column_param (str): Column name in InsolverDataFrame containing parameter.
            size_min (int): Minimum allowed number of records for each parameter's value, 1000 by default.

        Returns:
            list: List of parameter's values with few data.
        """
        param_size = pd.DataFrame(df.groupby(column_param).size().reset_index(name='param_size'))
        param_useless = list(param_size[column_param].loc[param_size['param_size'] < size_min])
        return param_useless

    def __call__(self, df):
        if not self.inference:
            self.param_useless = self._param_useless_get(df, self.column_param, self.size_min)
        df.loc[df[self.column_param].isin(self.param_useless), self.column_param] = self.group_name
        return df


class TransformParamSortFreq:
    """Gets sorted by claims' frequency parameter's values.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter.
        column_param_sort_freq (str): Column name in InsolverDataFrame for sorted values of parameter,
            column type is integer.
        column_policies_count (str): Column name in InsolverDataFrame containing numbers of policies,
            column type is integer or float.
        column_claims_count (str): Column name in InsolverDataFrame containing numbers of claims,
            column type is integer or float.
        inference (bool): Sign if the transformation is used for inference, False by default.
        param_freq_dict (dict): The dictionary of sorted values of the parameter, for inference only.
    """
    def __init__(self, column_param, column_param_sort_freq, column_policies_count, column_claims_count,
                 inference=False, param_freq_dict=None):
        self.priority = 2
        self.column_param = column_param
        self.column_param_sort_freq = column_param_sort_freq
        self.column_policies_count = column_policies_count
        self.column_claims_count = column_claims_count
        self.param_freq = pd.DataFrame
        self.inference = inference
        if inference:
            if param_freq_dict is None:
                raise NotImplementedError("'param_freq_dict' should contain the dictionary of sorted values.")
            self.param_freq_dict = param_freq_dict
        else:
            self.param_freq_dict = {}

    def __call__(self, df):
        if not self.inference:
            self.param_freq = df.groupby([self.column_param]).sum()[[self.column_claims_count,
                                                                     self.column_policies_count]]
            self.param_freq['freq'] = (self.param_freq[self.column_claims_count] /
                                       self.param_freq[self.column_policies_count])
            keys = []
            values = []
            for i in enumerate(self.param_freq.sort_values('freq', ascending=False).index.values):
                keys.append(i[1])
                values.append(float(i[0]))
            self.param_freq_dict = dict(zip(keys, values))
        df[self.column_param_sort_freq] = df[self.column_param].map(self.param_freq_dict)
        return df


class TransformParamSortAC:
    """Gets sorted by claims' average sum parameter's values.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter.
        column_param_sort_ac (str): Column name in InsolverDataFrame for sorted values of parameter,
            column type is integer.
        column_claims_count (str): Column name in InsolverDataFrame containing numbers of claims,
            column type is integer or float.
        column_claims_sum (str): Column name in InsolverDataFrame containing sums of claims,
            column type is integer or float.
        inference (bool): Sign if the transformation is used for inference, False by default.
        param_ac_dict (dict): The dictionary of sorted values of the parameter, for inference only.
    """
    def __init__(self, column_param, column_param_sort_ac, column_claims_count, column_claims_sum,
                 inference=False, param_ac_dict=None):
        self.priority = 2
        self.column_param = column_param
        self.column_param_sort_ac = column_param_sort_ac
        self.column_claims_count = column_claims_count
        self.column_claims_sum = column_claims_sum
        self.param_ac = pd.DataFrame
        self.inference = inference
        if inference:
            if param_ac_dict is None:
                raise NotImplementedError("'param_ac_dict' should contain the dictionary of sorted values.")
            self.param_ac_dict = param_ac_dict
        else:
            self.param_ac_dict = {}

    def __call__(self, df):
        if not self.inference:
            self.param_ac = df.groupby([self.column_param]).sum()[[self.column_claims_sum, self.column_claims_count]]
            self.param_ac['avg_claim'] = self.param_ac[self.column_claims_sum] / self.param_ac[self.column_claims_count]
            keys = []
            values = []
            for i in enumerate(self.param_ac.sort_values('avg_claim', ascending=False).index.values):
                keys.append(i[1])
                values.append(float(i[0]))
            self.param_ac_dict = dict(zip(keys, values))
        df[self.column_param_sort_ac] = df[self.column_param].map(self.param_ac_dict)
        return df


# ---------------------------------------------------
# Other data methods
# ---------------------------------------------------


class TransformToNumeric:
    """Transforms parameter's values to numeric types, uses Pandas' 'to_numeric'.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter to transform.
        downcast: Target numeric dtype, equal to Pandas' 'downcast' in the 'to_numeric' function, 'integer' by default.
    """
    def __init__(self, column_param, downcast='integer'):
        self.priority = 0
        self.column_param = column_param
        self.downcast = downcast

    def __call__(self, df):
        df[self.column_param] = pd.to_numeric(df[self.column_param], downcast=self.downcast)
        return df


class TransformMapValues:
    """Transforms parameter's values according to the dictionary.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter to map.
        dictionary (dict): The dictionary for mapping.
    """
    def __init__(self, column_param, dictionary):
        self.priority = 1
        self.column_param = column_param
        self.dictionary = dictionary

    def __call__(self, df):
        df[self.column_param] = df[self.column_param].map(self.dictionary)
        return df


class TransformPolynomizer:
    """Gets polynomials of parameter's values.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter to polynomize.
        n (int): Polynomial degree.
    """
    def __init__(self, column_param, n=2):
        self.priority = 3
        self.column_param = column_param
        self.n = n

    def __call__(self, df):
        for i in range(2, self.n + 1):
            a = self.column_param + '_' + str(i)
            while a in list(df.columns):
                a = a + '_'
            df[a] = df[self.column_param] ** i
        return df


class TransformGetDummies:
    """Gets dummy columns of the parameter, uses Pandas' 'get_dummies'.

    Attributes:
        column_param (str): Column name in InsolverDataFrame containing parameter to transform.
        drop_first (bool): Whether to get k-1 dummies out of k categorical levels by removing the first level,
            False by default.
        inference (bool): Sign if the transformation is used for inference, False by default.
        dummy_columns (list): List of the dummy columns, for inference only.
    """
    def __init__(self, column_param, drop_first=False, inference=False, dummy_columns=None):
        self.priority = 3
        self.column_param = column_param
        self.drop_first = drop_first
        self.inference = inference
        if inference:
            if dummy_columns is None:
                raise NotImplementedError("'dummy_columns' should contain the list of dummy columns.")
            self.dummy_columns = dummy_columns
        else:
            self.dummy_columns = []

    def __call__(self, df):
        if not self.inference:
            df_dummy = pd.get_dummies(df[[self.column_param]], prefix_sep='_', drop_first=self.drop_first)
            self.dummy_columns = list(df_dummy.columns)
            df = pd.concat([df, df_dummy], axis=1)
        else:
            for column in self.dummy_columns:
                df[column] = 1 * ((self.column_param + '_' + df[self.column_param]) == column)
        return df


class TransformCarFleetSize:
    """Calculates fleet sizes for policyholders.

    Attributes:
        column_id (str): Column name in InsolverDataFrame containing policyholders' IDs.
        column_date_start (str): Column name in InsolverDataFrame containing policies' start dates, column type is date.
        column_fleet_size (str): Column name in InsolverDataFrame for fleet sizes, column type is int.
    """
    def __init__(self, column_id, column_date_start, column_fleet_size):
        self.priority = 3
        self.column_id = column_id
        self.column_date_start = column_date_start
        self.column_fleet_size = column_fleet_size

    def __call__(self, df):
        cp = pd.merge(df[[self.column_id, self.column_date_start]], df[[self.column_id, self.column_date_start]],
                      on=self.column_id, how='left')
        cp = cp[(cp[f'{self.column_date_start}_y'] > cp[f'{self.column_date_start}_x'] - np.timedelta64(1, 'Y')) &
                (cp[f'{self.column_date_start}_y'] <= cp[f'{self.column_date_start}_y'])]
        cp = cp.groupby(self.column_id).size().to_dict()
        df[self.column_fleet_size] = df[self.column_id].map(cp)
        return df


class AutoFillNATransforms:
    """Fill NA values

    Attributes:
        numerical_columns (list): List of numerical columns
        categorical_columns (list): List of categorical columns
        medians (dict): Dictionary of median for each numerical column
        freq_categories (dict): Dictionary of frequency for each categorical column
    """
    def __init__(self, numerical_columns=None, categorical_columns=None, medians=None, freq_categories=None):
        self.priority = 0
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.medians = medians
        self.freq_categories = freq_categories

    def _find_num_cat_features(self, df):
        self.categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']
        self.numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']

    def _fillna_numerical(self, df):
        """Replace nan values with median values"""
        if not self.numerical_columns:
            return
        self.medians = {}
        for column in self.numerical_columns:
            if df[column].isnull().all():
                self.medians[column] = 1
            else:
                self.medians[column] = df[column].median()
            df[column].fillna(self.medians[column], axis=0, inplace=True)

    def _fillnan_categorical(self, df):
        """Replace nan values with most occurred category"""
        if not self.categorical_columns:
            return
        self.freq_categories = {}
        for column in self.categorical_columns:
            if df[column].mode().values.size > 0:
                most_frequent_category = df[column].mode()[0]
            else:
                most_frequent_category = 1
            self.freq_categories[column] = most_frequent_category
            df[column].fillna(most_frequent_category, inplace=True)

    def __call__(self, df):
        self._find_num_cat_features(df)
        self._fillna_numerical(df)
        self._fillnan_categorical(df)
        return df


class EncoderTransforms:
    """Label Encoder

     Attributes:
         column_names (list): columns for label encoding
         le_classes (dict): dictionary with label encoding classes for each column

    """
    def __init__(self, column_names, le_classes=None):
        self.priority = 3
        self.column_names = column_names
        self.le_classes = le_classes

    @staticmethod
    def _encode_column(column):
        le = LabelEncoder()
        le.fit(column)
        le_classes = le.classes_.tolist()
        column = le.transform(column)
        return column, le_classes

    def __call__(self, df):
        self.le_classes = {}
        for column_name in self.column_names:
            df[column_name], self.le_classes[column_name] = self._encode_column(df[column_name])
        return df


class OneHotEncoderTransforms:
    """OneHotEncoder Transformations

    Attributes:
        column_names (list): columns for one hot encoding
        encoder_dict (dict): dictionary with encoder_params for each column
    """
    def __init__(self, column_names, encoder_dict=None):
        self.priority = 3
        self.column_names = column_names
        self.encoder_dict = encoder_dict

    def _encode_column(self, df, column_name):
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(df[[column_name]])
        encoder_params = encoder.categories_
        encoder_params = [x.tolist() for x in encoder_params]
        column_encoded = pd.DataFrame(encoder.transform(df[[column_name]]))
        column_encoded.columns = encoder.get_feature_names([column_name])
        for column in column_encoded.columns:
            df[column] = column_encoded[column]
        return encoder_params

    def __call__(self, df):
        self.encoder_dict = {}
        for column in self.column_names:
            encoder_params = self._encode_column(df, column)
            self.encoder_dict[column] = encoder_params
            df.drop([column], axis=1, inplace=True)
        return df
