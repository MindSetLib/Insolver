import datetime
import pickle
import re
import traceback
import json

import pandas as pd

from .InsolverDataFrame import InsolverDataFrame
from .InsolverMain import InsolverTransformMain


class InsolverTransforms(InsolverDataFrame):
    """
    Class to compose transforms to be done on InsolverDataFrame.
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
        self.transforms_done = {}

    def transform(self):
        """
        Transforms data in InsolverDataFrame.

        :returns: List of transforms have been done.
        """
        if self._is_frame is None:
            raise NotImplementedError("No data loaded.")

        if self.transforms:

            try:

                priority_max = 0
                for transform in self.transforms:
                    if transform.priority > priority_max:
                        priority_max = transform.priority

                for priority in range(priority_max + 1):
                    for transform in self.transforms:
                        if transform.priority == priority:
                            self._df = transform(self._df)
                            attributes = {}
                            for attribute in dir(transform):
                                if attribute[0] != '_':
                                    exec("attributes.update({attribute: transform.%s})" % attribute)
                            self.transforms_done.update({type(transform).__name__: attributes})

            except Exception:
                traceback.print_last()

        return self.transforms_done

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.transforms_done, file)
            
    def save_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.transforms_done, file, separators=(',', ':'), sort_keys=True, indent=4)


# ---------------------------------------------------
# Person data methods
# ---------------------------------------------------


class TransformGenderGetFromName(InsolverTransformMain):
    """
    Gets clients' genders from russian second names.

    :param column_name: Column in InsolverDataFrame with clients' names, type is string.
    :param column_gender: Column in InsolverDataFrame for clients' genders, type is string.
    :param gender_male: Return value for male gender in InsolverDataFrame, 'male' by default.
    :param gender_female: Return value for female gender in InsolverDataFrame, 'female' by default.
    """
    def __init__(self, column_name, column_gender, gender_male='male', gender_female='female'):
        self.priority = 0
        super().__init__()
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


class TransformAgeGetFromBirthday(InsolverTransformMain):
    """
    Gets clients' ages from birth dates and policies' start dates.

    :param column_date_birth: Column in InsolverDataFrame with clients' birth dates, type is date.
    :param column_date_start: Column in InsolverDataFrame with policies' start dates, type is date.
    :param column_age: Column in InsolverDataFrame for clients' ages, type is integer.
    """
    def __init__(self, column_date_birth, column_date_start, column_age):
        self.priority = 0
        super().__init__()
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


class TransformAge(InsolverTransformMain):
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
        super().__init__()
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
        df[self.column_driver_minage] = df[self.column_driver_minage].apply(self._age, args=(self.age_min, self.age_max,))
        return df


class TransformAgeGender(InsolverTransformMain):
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
        super().__init__()
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


class TransformExp(InsolverTransformMain):
    """
    Transforms values of drivers' minimum experiences with values over 'exp_max' grouped.

    :param column_driver_minexp: Column in InsolverDataFrame with drivers' minimum experiences, type is integer.
    :param exp_max: Maximum value of drivers' experience, bigger values will be grouped, type is integer, 52 by default.
    """
    def __init__(self, column_driver_minexp, exp_max=52):
        self.priority = 1
        super().__init__()
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


class TransformAgeExpDiff(InsolverTransformMain):
    """
    Transforms records with difference between drivers' minimum age and minimum experience less then 18 years,
    sets drivers' minimum experience equal to drivers' minimum age minus 18 years.
    """
    def __init__(self, column_driver_minage, column_driver_minexp, diff_min=18):
        self.priority = 2
        super().__init__()
        self.column_driver_minage = column_driver_minage
        self.column_driver_minexp = column_driver_minexp
        self.diff_min = diff_min

    def __call__(self, df):
        self.num_errors = len(df.loc[(df[self.column_driver_minage] - df[self.column_driver_minexp]) < self.diff_min])
        df[self.column_driver_minexp].loc[(df[self.column_driver_minage] - df[self.column_driver_minexp])
                                          < self.diff_min] = df[self.column_driver_minage] - self.diff_min
        return df


class TransformNameCheck(InsolverTransformMain):
    """
    Checks if clients' first names are in special list.
    Names should concatenate surnames, first names and second names.

    :param column_name: Column in InsolverDataFrame with clients' names, type is string.
    :param column_name_check: Column in InsolverDataFrame for bool values are first names in the list or not.
    :param names_list: The list of clients' first names, type is list with upper strings.
    """
    def __init__(self, column_name, column_name_check, names_list):
        self.priority = 1
        super().__init__()
        self.column_name = column_name
        self.column_name_check = column_name_check
        self.names_list = names_list

    @staticmethod
    def _name_get(client_name):
        tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            name = tokenize_re.findall(str(client_name))[1].upper()
            return name
        except Exception:
            return 'ERROR'

    def __call__(self, df):
        df[self.column_name_check] = 1 * df[self.column_name].apply(self._name_get).isin(self.names_list)
        return df


# ---------------------------------------------------
# Vehicle data methods
# ---------------------------------------------------


class TransformVehPower(InsolverTransformMain):
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
        super().__init__()
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
        df[self.column_veh_power] = df[self.column_veh_power].apply(self._power, args=(self.power_min, self.power_max, self.power_step,))
        return df


class TransformVehAgeGetFromIssueYear(InsolverTransformMain):
    """
    Gets vehicles' ages from issue years and policies' start dates.

    :param column_veh_issue_year: Column in InsolverDataFrame with vehicles' issue years, type is integer.
    :param column_date_start: Column in InsolverDataFrame with policies' start dates, type is date.
    :param column_veh_age: Column in InsolverDataFrame for vehicles' ages, type is integer.
    """
    def __init__(self, column_veh_issue_year, column_date_start, column_veh_age):
        self.priority = 0
        super().__init__()
        self.column_veh_issue_year = column_veh_issue_year
        self.column_date_start = column_date_start
        self.column_veh_age = column_veh_age

    @staticmethod
    def _veh_age_get(issueyear_datestart):
        veh_issue_year = issueyear_datestart[0]
        date_start = issueyear_datestart[1]
        veh_age = None
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
        df[self.column_veh_age] = df[[self.column_veh_issue_year, self.column_date_start]].apply(self._veh_age_get, axis=1)
        return df


class TransformVehAge(InsolverTransformMain):
    """
    Transforms values of vehicles' ages.
    Values over 'veh_age_max' will be grouped.

    :param column_veh_age: Column in InsolverDataFrame with vehicles' ages, type is integer.
    :param veh_age_max: Maximum value of vehicles' age, bigger values will be grouped, type is integer, 25 by default.
    """
    def __init__(self, column_veh_age, veh_age_max=25):
        self.priority = 1
        super().__init__()
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


class TransformRegionGetFromKladr(InsolverTransformMain):
    """
    Gets regions' numbers from column KLADRs.

    :param column_kladr: Column in InsolverDataFrame with KLADRs, type is string.
    :param column_region_num: Column in InsolverDataFrame for regions, type is integer.
    """
    def __init__(self, column_kladr, column_region_num):
        self.priority = 0
        super().__init__()
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


class TransformParamUselessGroup(InsolverTransformMain):
    """
    Groups all parameter's values with few data to one group.

    :param column_param: Column in InsolverDataFrame with parameter.
    :param size_min: Minimum allowed number of records for each parameter value, type is integer, 1000 by default.
    :param group_name: Name of the group for parameter's values with few data.
    """
    def __init__(self, column_param, size_min=1000, group_name=0):
        self.priority = 1
        super().__init__()
        self.column_param = column_param
        self.size_min = size_min
        self.group_name = group_name
        self.param_useless = {}

    @staticmethod
    def _param_useless_get(df, column_param, size_min):
        """
        Checks the amount of data for each parameter's value.

        :param _df: InsolverDataFrame to explore.
        :param _column_param: Column in InsolverDataFrame with parameter.
        :param _size_min: Minimum allowed number of records for each parameter's value, type is integer, 1000 by default.
        :returns: List of parameter's values with few data.
        """
        param_size = pd.DataFrame(df.groupby(column_param).size().reset_index(name='param_size'))
        param_useless = list(param_size[column_param].loc[param_size['param_size'] < size_min])
        return param_useless

    def __call__(self, df):
        self.param_useless = self._param_useless_get(df, self.column_param, self.size_min)
        df.loc[df[self.column_param].isin(self.param_useless), self.column_param] = self.group_name
        return df


class TransformParamSortFreq(InsolverTransformMain):
    """
    Gets sorted by claims' frequency parameter's values.

    :param column_param: Column in InsolverDataFrame with parameter.
    :param column_param_sort_freq: Column in InsolverDataFrame for sorted values of parameter, type is integer.
    :param column_policies_count: Column in InsolverDataFrame with number of policies, type is integer or float.
    :param column_claims_count: Column in InsolverDataFrame with number of claims, type is integer or float.
    """
    def __init__(self, column_param, column_param_sort_freq, column_policies_count, column_claims_count):
        self.priority = 2
        super().__init__()
        self.column_param = column_param
        self.column_param_sort_freq = column_param_sort_freq
        self.column_policies_count = column_policies_count
        self.column_claims_count = column_claims_count
        self.param_freq = pd.DataFrame
        self.param_freq_dict = {}

    def __call__(self, df):
        self.param_freq = df.groupby([self.column_param]).sum()[[self.column_claims_count, self.column_policies_count]]
        self.param_freq['freq'] = self.param_freq[self.column_claims_count] / self.param_freq[self.column_policies_count]
        keys = []
        values = []
        for i in enumerate(self.param_freq.sort_values('freq', ascending=False).index.values):
            keys.append(i[1])
            values.append(float(i[0]))
        self.param_freq_dict = dict(zip(keys, values))
        df[self.column_param_sort_freq] = df[self.column_param].map(self.param_freq_dict)
        return df


class TransformParamSortAC(InsolverTransformMain):
    """
    Gets sorted by claims' average sum parameter's values.

    :param column_param: Column in InsolverDataFrame with parameter.
    :param column_param_sort_ac: Column in InsolverDataFrame for sorted values of parameter, type is integer.
    :param column_claims_count: Column in InsolverDataFrame with number of claims, type is integer or float.
    :param column_claims_sum: Column in InsolverDataFrame with sum of claims, type is integer or float.
    """
    def __init__(self, column_param, column_param_sort_ac, column_claims_count, column_claims_sum):
        self.priority = 2
        super().__init__()
        self.column_param = column_param
        self.column_param_sort_ac = column_param_sort_ac
        self.column_claims_count = column_claims_count
        self.column_claims_sum = column_claims_sum
        self.param_ac = pd.DataFrame
        self.param_ac_dict = {}

    def __call__(self, df):
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


class TransformToNumeric(InsolverTransformMain):
    """
    Transforms parameter's values to numeric types, equal to Pandas' 'to_numeric'.

    :param column_param: Column in InsolverDataFrame with parameter to transform.
    :param downcast: Target numeric dtype, equal to Pandas' 'downcast' in the 'to_numeric' function, 'integer' by default.
    """
    def __init__(self, column_param, downcast='integer'):
        self.priority = 0
        super().__init__()
        self.column_param = column_param
        self.downcast = downcast

    def __call__(self, df):
        df[self.column_param] = pd.to_numeric(df[self.column_param], downcast=self.downcast)
        return df


class TransformMapValues(InsolverTransformMain):
    """
    Transforms parameter's values according to the dictionary.

    :param column_param: Column in InsolverDataFrame with parameter to map.
    :param dictionary: The dictionary for mapping.
    """
    def __init__(self, column_param, dictionary):
        self.priority = 1
        super().__init__()
        self.column_param = column_param
        self.dictionary = dictionary

    def __call__(self, df):
        df[self.column_param] = df[self.column_param].map(self.dictionary)
        return df


class TransformPolynomizer(InsolverTransformMain):
    """
    Gets polynomials of parameter's values.

    :param column_param: Column in InsolverDataFrame with parameter to polinomize.
    :param n: Polinomial's degree, type is integer.
    """
    def __init__(self, column_param, n=2):
        self.priority = 3
        super().__init__()
        self.column_param = column_param
        self.n = n

    def __call__(self, df):
        for i in range(2, self.n + 1):
            _a = self.column_param + '_' + str(i)
            while _a in list(df.columns):
                _a = _a + '_'
            df[_a] = df[self.column_param] ** i
        return df


class TransformGetDummies(InsolverTransformMain):
    """
    Gets dummy columns of the parameter.

    :param column_param: Column in InsolverDataFrame with parameter to transform.
    """
    def __init__(self, column_param):
        self.priority = 3
        super().__init__()
        self.column_param = column_param

    def __call__(self, df):
        df = pd.get_dummies(df, columns=self.column_param)
        return df
