import datetime

from numpy import timedelta64
from pandas import isnull, merge


class TransformExp:
    """Transforms values of drivers' minimum experiences in years with values over 'exp_max' grouped.

    Parameters:
        column_driver_minexp (str): Column name in InsolverDataFrame containing drivers' minimum experiences in years,
            column type is integer.
        exp_max (int): Maximum value of drivers' experience in years, bigger values will be grouped, 52 by default.
    """

    def __init__(self, column_driver_minexp, exp_max=52, priority=1):
        self.priority = priority
        self.column_driver_minexp = column_driver_minexp
        self.exp_max = exp_max

    @staticmethod
    def _exp(exp, exp_max):
        if isnull(exp):
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

    Parameters:
        column_driver_minage (str): Column name in InsolverDataFrame containing drivers' minimum ages in years,
            column type is integer.
        column_driver_minexp (str): Column name in InsolverDataFrame containing drivers' minimum experiences in years,
            column type is integer.
        diff_min (int): Minimum allowed difference between age and experience in years.
    """

    def __init__(self, column_driver_minage, column_driver_minexp, diff_min=18, priority=2):
        self.priority = priority
        self.column_driver_minage = column_driver_minage
        self.column_driver_minexp = column_driver_minexp
        self.diff_min = diff_min

    def __call__(self, df):
        self.num_errors = len(df.loc[(df[self.column_driver_minage] - df[self.column_driver_minexp]) < self.diff_min])
        df[self.column_driver_minexp].loc[
            (df[self.column_driver_minage] - df[self.column_driver_minexp]) < self.diff_min
        ] = (df[self.column_driver_minage] - self.diff_min)
        return df


class TransformVehPower:
    """Transforms values of vehicles' powers.
    Values under 'power_min' and over 'power_max' will be grouped.
    Values between 'power_min' and 'power_max' will be grouped with step 'power_step'.

    Parameters:
        column_veh_power (str): Column name in InsolverDataFrame containing vehicles' powers,
            column type is float.
        power_min (float): Minimum value of vehicles' power, lower values will be grouped, 10 by default.
        power_max (float): Maximum value of vehicles' power, bigger values will be grouped, 500 by default.
        power_step (int): Values of vehicles' power will be divided by this parameter, rounded to integers,
            10 by default.
    """

    def __init__(self, column_veh_power, power_min=10, power_max=500, power_step=10, priority=1):
        self.priority = priority
        self.column_veh_power = column_veh_power
        self.power_min = power_min
        self.power_max = power_max
        self.power_step = power_step

    @staticmethod
    def _power(power, power_min, power_max, power_step):
        if isnull(power):
            power = None
        elif power < power_min:
            power = power_min
        elif power > power_max:
            power = power_max
        else:
            power = int(round(power / power_step, 0))
        return power

    def __call__(self, df):
        df[self.column_veh_power] = df[self.column_veh_power].apply(
            self._power,
            args=(
                self.power_min,
                self.power_max,
                self.power_step,
            ),
        )
        return df


class TransformVehAgeGetFromIssueYear:
    """Gets vehicles' ages in years from issue years and policies' start dates.

    Parameters:
        column_veh_issue_year (str): Column name in InsolverDataFrame containing vehicles' issue years,
            column type is integer.
        column_date_start (str): Column name in InsolverDataFrame containing policies' start dates, column type is date.
        column_veh_age (str): Column name in InsolverDataFrame for vehicles' ages in years, column type is integer.
    """

    def __init__(self, column_veh_issue_year, column_date_start, column_veh_age, priority=0):
        self.priority = priority
        self.column_veh_issue_year = column_veh_issue_year
        self.column_date_start = column_date_start
        self.column_veh_age = column_veh_age

    @staticmethod
    def _veh_age_get(issueyear_datestart):
        veh_issue_year = issueyear_datestart[0]
        date_start = issueyear_datestart[1]
        if isnull(veh_issue_year):
            veh_age = None
        elif isnull(date_start):
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
        df[self.column_veh_age] = df[[self.column_veh_issue_year, self.column_date_start]].apply(
            self._veh_age_get, axis=1
        )
        return df


class TransformVehAge:
    """Transforms values of vehicles' ages in years. Values over 'veh_age_max' will be grouped.

    Parameters:
        column_veh_age (str): Column name in InsolverDataFrame containing vehicles' ages in years,
            column type is integer.
        veh_age_max (int): Maximum value of vehicles' age in years, bigger values will be grouped, 25 by default.
    """

    def __init__(self, column_veh_age, veh_age_max=25, priority=1):
        self.priority = priority
        self.column_veh_age = column_veh_age
        self.veh_age_max = veh_age_max

    @staticmethod
    def _veh_age(age, age_max):
        if isnull(age):
            age = None
        elif age < 0:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def __call__(self, df):
        df[self.column_veh_age] = df[self.column_veh_age].apply(self._veh_age, args=(self.veh_age_max,))
        return df


class TransformRegionGetFromKladr:
    """Gets regions' numbers from KLADRs.

    Parameters:
        column_kladr (str): Column name in InsolverDataFrame containing KLADRs, column type is string.
        column_region_num (str): Column name in InsolverDataFrame for regions' numbers, column type is integer.
    """

    def __init__(self, column_kladr, column_region_num, priority=0):
        self.priority = priority
        self.column_kladr = column_kladr
        self.column_region_num = column_region_num

    @staticmethod
    def _region_get(kladr):
        if isnull(kladr):
            region_num = None
        else:
            region_num = kladr[0:2]

        try:
            region_num = int(region_num)
        except ValueError:
            region_num = None

        return region_num

    def __call__(self, df):
        df[self.column_region_num] = df[self.column_kladr].apply(self._region_get)
        return df


class TransformCarFleetSize:
    """Calculates fleet sizes for policyholders.

    Parameters:
        column_id (str): Column name in InsolverDataFrame containing policyholders' IDs.
        column_date_start (str): Column name in InsolverDataFrame containing policies' start dates, column type is date.
        column_fleet_size (str): Column name in InsolverDataFrame for fleet sizes, column type is int.
    """

    def __init__(self, column_id, column_date_start, column_fleet_size, priority=3):
        self.priority = priority
        self.column_id = column_id
        self.column_date_start = column_date_start
        self.column_fleet_size = column_fleet_size

    def __call__(self, df):
        cp = merge(
            df[[self.column_id, self.column_date_start]],
            df[[self.column_id, self.column_date_start]],
            on=self.column_id,
            how='left',
        )
        cp = cp[
            (cp[f'{self.column_date_start}_y'] > cp[f'{self.column_date_start}_x'] - timedelta64(1, 'Y'))
            & (cp[f'{self.column_date_start}_y'] <= cp[f'{self.column_date_start}_y'])
        ]
        cp = cp.groupby(self.column_id).size().to_dict()
        df[self.column_fleet_size] = df[self.column_id].map(cp)
        return df
