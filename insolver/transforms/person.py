import re
import datetime

import pandas as pd


class TransformGenderGetFromName:
    """Gets clients' genders from theirs russian second names.

    Parameters:
        column_name (str): Column name in InsolverDataFrame containing clients' names, column type is string.
        column_gender (str): Column name in InsolverDataFrame for clients' genders.
        gender_male (str): Return value for male gender in InsolverDataFrame, 'male' by default.
        gender_female (str): Return value for female gender in InsolverDataFrame, 'female' by default.
    """

    def __init__(self, column_name, column_gender, gender_male='male', gender_female='female', priority=0):
        self.priority = priority
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
        df[self.column_gender] = df[self.column_name].apply(
            self._gender,
            args=(
                self.gender_male,
                self.gender_female,
            ),
        )
        return df


class TransformAgeGetFromBirthday:
    """Gets clients' ages in years from theirs birth dates and policies' start dates.

    Parameters:
        column_date_birth (str): Column name in InsolverDataFrame containing clients' birth dates, column type is date.
        column_date_start (str): Column name in InsolverDataFrame containing policies' start dates, column type is date.
        column_age (str): Column name in InsolverDataFrame for clients' ages in years, column type is int.
    """

    def __init__(self, column_date_birth, column_date_start, column_age, priority=0):
        self.priority = priority
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

    Parameters:
        column_driver_minage (str): Column name in InsolverDataFrame containing drivers' minimum ages in years,
            column type is integer.
        age_min (int): Minimum value of drivers' age in years, lower values are invalid, 18 by default.
        age_max (int): Maximum value of drivers' age in years, bigger values will be grouped, 70 by default.
    """

    def __init__(self, column_driver_minage, age_min=18, age_max=70, priority=1):
        self.priority = priority
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
        df[self.column_driver_minage] = df[self.column_driver_minage].apply(
            self._age, args=(self.age_min, self.age_max)
        )
        return df


class TransformAgeGender:
    """Gets intersections of drivers' minimum ages and genders.

    Parameters:
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

    def __init__(
        self,
        column_age,
        column_gender,
        column_age_m,
        column_age_f,
        age_default=18,
        gender_male='male',
        gender_female='female',
        priority=2,
    ):
        self.priority = priority
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
        df[self.column_age_m], df[self.column_age_f] = zip(
            *df[[self.column_age, self.column_gender]]
            .apply(self._age_gender, axis=1, args=(self.age_default, self.gender_male, self.gender_female))
            .to_frame()[0]
        )
        return df


class TransformNameCheck:
    """Checks if clients' first names are in special list.
    Names may concatenate surnames, first names and last names.

    Parameters:
        column_name (str): Column name in InsolverDataFrame containing clients' names, column type is string.
        name_full (bool): Sign if name is the concatenation of surname, first name and last name, False by default.
        column_name_check (str): Column name in InsolverDataFrame for bool values if first names are in the list or not.
        names_list (list): The list of clients' first names.
        name_position (int): The position of the name in full name. For example, argument should be 0 for notation such
        as 'John Doe', but 1 for notation like 'Ivanov Ivan'.
    """

    def __init__(self, column_name, column_name_check, names_list, name_full=False, name_position=1, priority=1):
        self.priority = priority
        self.column_name = column_name
        self.name_full = name_full
        self.column_name_check = column_name_check
        self.name_position = name_position
        self.names_list = [n.upper() for n in names_list]

    @staticmethod
    def _name_get(client_name, name_position):
        tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            name = tokenize_re.findall(str(client_name))[name_position].upper()
            return name
        except IndexError:
            return 'ERROR'

    def __call__(self, df):
        if not self.name_full:
            df[self.column_name_check] = 1 * df[self.column_name].isin(self.names_list)
        else:
            df[self.column_name_check] = 1 * df[self.column_name].apply(self._name_get).isin(self.names_list)
        return df
