import pandas as pd
import re
import datetime
import numpy as np
import pyodbc


class InsolverDataFrame:

    def __init__(self):
        self._is_frame = True

    ''' Load data methods '''

    def load_pd(self, pd_dataframe):
        self._df = pd_dataframe

    def load_csv(self, csv_dataframe):
        self._df = pd.read_csv(csv_dataframe, low_memory=False)

    def load_mssql(self, server, database, username, password, table):
        cnxn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        self._df = pd.read_sql('select * from ' + table, cnxn)
        cnxn.close()

    ''' Columns check methods '''

    def columns_set(self, columns=None):

        if columns == None:
            self._df_columns = [
                'driver_minage',  # int>=18
                'driver_minexp',  # int>=0
                'driver_maxkbm',  # float>0

                'client_type',  # [company,person]
                'client_name',  # 'Иванов Иван Иванович'
                'client_date_birth',  # date
                'client_gender',  # [male,female]

                'vehicle_power',  # int>0
                [
                    'vehicle_issue_year',  # int>0
                    'vehicle_age',  # int>=0
                ],
                'vehicle_type',  # int>=0

                'p_date_start',  # date
                'p_is_taxi',  # [0,1]
                'p_is_driver_unlimit',  # [0,1]
                'kladr',  # str

                'p_claims_sum_infl',  # float>=0
                'p_claims_count_adj',  # float>=0
            ]
        else:
            self._df_columns = columns

    def columns_check(self):

        if not hasattr(self, '_df_columns'):
            self.columns_set()

        _columns_check = {}

        for column in self._df_columns:

            if type(column) == str:

                if column in list(self._df.columns):
                    _columns_check[column] = True
                else:
                    _columns_check[column] = False

            elif type(column) == list:

                j = 0
                c = ''
                for i in range(len(column)):
                    c = c + column[i] + ' _or_ '
                    if column[i] in list(self._df.columns):
                        j += 1
                if j > 0:
                    _columns_check[c] = True
                else:
                    _columns_check[c] = False

        return _columns_check

    ''' Columns match methods '''

    def columns_match(self, match_from_to):
        self._df.rename(columns=match_from_to, inplace=True)

    ''' Person data methods '''

    # Client type
    _client_type_dict = {
        'person': float(0),
        'company': float(1),
        '0': float(0),
        '1': float(1),
        0: float(0),
        1: float(1)
    }

    def transform_client_type(self):
        self._df['client_type'] = self._df['client_type'].map(self._client_type_dict)

    # Gender
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
        self._df['gender_m'], self._df['gender_f'] = zip(
            *self._df[['client_type', 'client_name', 'client_gender']].apply(self._gender, axis=1).to_frame()[0])

    # Age from Birthdate
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
        self._df['driver_minage'] = self._df[['client_date_birth', 'p_date_start']].apply(self._age_get, axis=1)

    # Age
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
        self._df['driver_minage'] = self._df['driver_minage'].apply(self._age, args=(age_max,))

    # Age M/F
    @staticmethod
    def _age_gender(age_gender):
        _age = age_gender[0]
        _gender = age_gender[1]
        if _gender == 0:
            _age = 18
        return _age

    def transform_age_gender(self):
        self._df['driver_minage_m'] = self._df[['driver_minage', 'gender_m']].apply(_age_gender, axis=1)
        self._df['driver_minage_f'] = self._df[['driver_minage', 'gender_f']].apply(_age_gender, axis=1)

    # Experience
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
        self._df['driver_minexp'] = self._df['driver_minexp'].apply(self._exp, args=(exp_max,))

    # Age - Experience < 18
    def transform_age_exp_18(self):
        n = len(self._df.loc[(self._df['driver_minage'] - self._df['driver_minexp']) < 18])
        self._df['driver_minexp'].loc[(self._df['driver_minage'] - self._df['driver_minexp']) < 18] = self._df[
                                                                                                          'driver_minage'] - 18
        return n

        # Check name

    @staticmethod
    def _name_get(client_name):
        _tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            _name = _tokenize_re.findall(str(client_name))[1].upper()
            return _name
        except Exception:
            return 'ERROR'

    def transform_name_check(self, names_list):
        self._df['client_name_check'] = 1 * self._df['client_name'].apply(self._name_get).isin(names_list)

    ''' Vehicle data methods '''

    # Power
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
        self._df['vehicle_power'] = self._df['vehicle_power'].apply(self._power,
                                                                    args=(power_min, power_max, power_group,))

    # Age from Issue year
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
        self._df['vehicle_age'] = self._df[['vehicle_issue_year', 'p_date_start']].apply(self._veh_age_get, axis=1)

    # Age
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
        self._df['vehicle_age'] = self._df['vehicle_age'].apply(self._veh_age, args=(veh_age_max,))

    # Vehicle types sort freq
    def transform_veh_type_sort_freq(self):

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

    # Vehicle types sort avg claim
    def transform_veh_type_sort_ac(self):

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

    ''' Region data methods '''

    # Region from KLADR
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
        self._df['region_num'] = self._df['kladr'].apply(self._region_get)

    # Regions with small data
    def region_useless(self, size_min=1000):
        if 'region_num' not in list(self._df.columns):
            self.region_get()
        _region_size = pd.DataFrame(self._df.groupby('region_num').size().reset_index(name='region_size'))
        return list(_region_size['region_num'].loc[_region_size['region_size'] < size_min])

    # Regions with small data group
    def transform_region_useless_group(self, size_min=1000):
        if 'region_num' not in list(self._df.columns):
            self.region_get()
        self._df.loc[self._df['region_num'].isin(self.region_useless(size_min)), 'region_num'] = 0

    # Regions sort freq
    def transform_region_sort_freq(self):

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

    # Regions sort avg claim
    def transform_region_sort_ac(self):

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

    ''' Other data methods '''

    def polynomizer(self, column, n=2):
        if column in list(self._df.columns):
            for i in range(2, n + 1):
                self._df[column + '_' + str(i)] = self._df[column] ** i
            return 'Ok'
        else:
            return 'Error: column not in columns'

    ''' General methods '''

    def info(self):
        return self._df.info()

    def head(self, n=5):
        return self._df.head(n)

    def len(self):
        return len(self._df)