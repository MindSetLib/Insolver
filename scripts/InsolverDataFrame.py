import pandas as pd
import re
import datetime
import pyodbc


class InsolverDataFrame:

    def __init__(self):
        self._is_frame = True

    ''' Load data methods '''

    def load_csv(self, csv_dataframe):
        self.df = pd.read_csv(csv_dataframe, low_memory=False)

    def load_mssql(self, server, database, username, password, table):
        cnxn = pyodbc.connect(
            'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        self.df = pd.read_sql('select * from ' + table, cnxn)
        cnxn.close()

    ''' Columns matching methods '''

    def columns_matching(self, dict_match=None):
        self.df.rename(columns=dict_match, inplace=True)

    ''' Person data methods '''

    # Client type
    dict_client_type = {
        'person': float(0),
        'company': float(1),
        '0': float(0),
        '1': float(1),
        0: float(0),
        1: float(1)
    }

    def transform_client_type(self):
        self.df['client_type'] = self.df['client_type'].map(self.dict_client_type)

    # Gender
    @staticmethod
    def _gender(client_type_name_gender):

        client_type = client_type_name_gender[0]
        client_name = client_type_name_gender[1]
        client_gender = client_type_name_gender[2]

        if client_type == 'company':  # juridic
            gender_m = 0
            gender_f = 0

        elif client_type == '1':  # juridic
            gender_m = 0
            gender_f = 0

        elif client_type == 1:  # juridic
            gender_m = 0
            gender_f = 0

        elif client_gender == 'male':
            gender_m = 1
            gender_f = 0

        elif client_gender == 'female':
            gender_m = 0
            gender_f = 1

        else:
            try:
                if len(client_name) < 2:
                    gender_m = 0
                    gender_f = 0
                elif client_name[-2:].upper() == 'ИЧ':
                    gender_m = 1
                    gender_f = 0
                elif client_name[-4:].upper() == 'ОГЛЫ':
                    gender_m = 1
                    gender_f = 0
                elif client_name[-2:].upper() == 'НА':
                    gender_m = 0
                    gender_f = 1
                elif client_name[-4:].upper() == 'КЫЗЫ':
                    gender_m = 0
                    gender_f = 1
                else:
                    gender_m = 0
                    gender_f = 0
            except:
                gender_m = 0
                gender_f = 0

        return [gender_m, gender_f]

    def transform_gender(self):
        self.df['gender_m'], self.df['gender_f'] = zip(
            *self.df[['client_type', 'client_name', 'client_gender']].apply(self._gender, axis=1).to_frame()[0])

    # Age from Birthdate
    @staticmethod
    def _age_get(datebirth_datestart):
        client_date_birth = datebirth_datestart[0]
        p_date_start = datebirth_datestart[1]
        age = None
        if client_date_birth > datetime.datetime.now():
            age = None
        elif client_date_birth.year < datetime.datetime.now().year - 120:
            age = None
        elif client_date_birth > p_date_start:
            age = None
        else:
            age = (p_date_start - client_date_birth).days // 365.25
        return age

    def age_get(self):
        self.df['driver_minage'] = self.df[['client_date_birth', 'p_date_start']].apply(self._age_get, axis=1)

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
        self.df['driver_minage'] = self.df['driver_minage'].apply(self._age, args=(age_max,))

    # Age M/F
    @staticmethod
    def _age_gender(age_gender):
        age = age_gender[0]
        gender = age_gender[1]
        if gender == 0:
            age = 18
        return age

    def transform_age_gender(self):
        self.df['driver_minage_m'] = self.df[['driver_minage', 'gender_m']].apply(_age_gender, axis=1)
        self.df['driver_minage_f'] = self.df[['driver_minage', 'gender_f']].apply(_age_gender, axis=1)

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
        self.df['driver_minexp'] = self.df['driver_minexp'].apply(self._exp, args=(exp_max,))

    # Age - Experience < 18
    def transform_age_exp_18(self):
        n = len(self.df.loc[(self.df['driver_minage'] - self.df['driver_minexp']) < 18])
        self.df['driver_minexp'].loc[(self.df['driver_minage'] - self.df['driver_minexp']) < 18] = self.df[
                                                                                                       'driver_minage'] - 18
        return n

        # Check name

    @staticmethod
    def _name_get(client_name):
        tokenize_re = re.compile(r'[\w\-]+', re.I)
        try:
            name = tokenize_re.findall(str(client_name))[1].upper()
            return name
        except Exception:
            return 'ERROR'

    def transform_name_check(self, names_list):
        self.df['client_name_check'] = 1 * self.df['client_name'].apply(self._name_get).isin(names_list)

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
        self.df['vehicle_power'] = self.df['vehicle_power'].apply(self._power,
                                                                  args=(power_min, power_max, power_group,))

    # Age from Issue year
    @staticmethod
    def _veh_age_get(issueyear_datestart):
        vehicle_issue_year = issueyear_datestart[0]
        p_date_start = issueyear_datestart[1]
        veh_age = None
        if vehicle_issue_year > datetime.datetime.now().year:
            veh_age = None
        elif vehicle_issue_year < datetime.datetime.now().year - 70:
            veh_age = None
        elif vehicle_issue_year > p_date_start.year:
            veh_age = None
        else:
            veh_age = p_date_start.year - vehicle_issue_year
        return veh_age

    def veh_age_get(self):
        self.df['vehicle_age'] = self.df[['vehicle_issue_year', 'p_date_start']].apply(self._veh_age_get, axis=1)

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
        self.df['vehicle_age'] = self.df['vehicle_age'].apply(self._veh_age, args=(veh_age_max,))

    # Vehicle types sort freq
    def transform_veh_type_sort_freq(self):

        self.df['count'] = 1

        spr_veh_type_freq = self.df.groupby(['vehicle_type']).sum()[['p_claims_count_adj', 'count']]

        spr_veh_type_freq['freq'] = spr_veh_type_freq['p_claims_count_adj'] / spr_veh_type_freq['count']

        keys = []
        values = []
        for i in enumerate(spr_veh_type_freq.sort_values('freq', ascending=False).index.values):
            keys.append(i[1])
            values.append(float(i[0]))

        dict_veh_type_freq = dict(zip(keys, values))

        self.df['vehicle_type_freq'] = self.df['vehicle_type'].map(dict_veh_type_freq)

        return dict_veh_type_freq

    # Vehicle types sort avg claim
    def transform_veh_type_sort_ac(self):

        spr_veh_type_ac = self.df.groupby(['vehicle_type']).sum()[['p_claims_sum_infl', 'p_claims_count_adj']]

        spr_veh_type_ac['avg_claim'] = spr_veh_type_ac['p_claims_sum_infl'] / spr_veh_type_ac['p_claims_count_adj']

        keys = []
        values = []
        for i in enumerate(spr_veh_type_ac.sort_values('avg_claim', ascending=False).index.values):
            keys.append(i[1])
            values.append(float(i[0]))

        dict_veh_type_ac = dict(zip(keys, values))

        self.df['veh_type_ac'] = self.df['vehicle_type'].map(dict_veh_type_ac)

        return dict_veh_type_ac

    ''' Region data methods '''

    # Region from KLADR
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

    def region_get(self):
        self.df['region_num'] = self.df['kladr'].apply(self._region_get)

    # Regions with small data
    def region_useless(self, size_min=1000):
        if 'region_num' not in self.df.columns:
            self.region_get()
        region_size = pd.DataFrame(self.df.groupby('region_num').size().reset_index(name='region_size'))
        return list(region_size['region_num'].loc[region_size['region_size'] < size_min])

    # Regions with small data group
    def transform_region_useless_group(self, size_min=1000):
        if 'region_num' not in self.df.columns:
            self.region_get()
        self.df.loc[self.df['region_num'].isin(self.region_useless(size_min)), 'region_num'] = 0

    # Regions sort freq
    def transform_region_sort_freq(self):

        self.df['count'] = 1

        spr_region_freq = self.df.groupby(['region_num']).sum()[['p_claims_count_adj', 'count']]

        spr_region_freq['freq'] = spr_region_freq['p_claims_count_adj'] / spr_region_freq['count']

        keys = []
        values = []
        for i in enumerate(spr_region_freq.sort_values('freq', ascending=False).index.values):
            keys.append(i[1])
            values.append(float(i[0]))

        dict_region_freq = dict(zip(keys, values))

        self.df['region_freq'] = self.df['region_num'].map(dict_region_freq)

        return dict_region_freq

    # Regions sort avg claim
    def transform_region_sort_ac(self):

        spr_region_ac = self.df.groupby(['region_num']).sum()[['p_claims_sum_infl', 'p_claims_count_adj']]

        spr_region_ac['avg_claim'] = spr_region_ac['p_claims_sum_infl'] / spr_region_ac['p_claims_count_adj']

        keys = []
        values = []
        for i in enumerate(spr_region_ac.sort_values('avg_claim', ascending=False).index.values):
            keys.append(i[1])
            values.append(float(i[0]))

        dict_region_ac = dict(zip(keys, values))

        self.df['region_ac'] = self.df['region_num'].map(dict_region_ac)

        return dict_region_ac

    ''' Other data methods '''

    def polynomizer(self, column, n=2):
        if column in list(self.df.columns):
            for i in range(2, n + 1):
                self.df[column + '_' + str(i)] = self.df[column] ** i
            return 'Ok'
        else:
            return 'Error: column not in columns'

    ''' General methods '''

    def info(self):
        return self.df.info()

    def head(self, n=5):
        return self.df.head(n)

    def len(self):
        return len(self.df)