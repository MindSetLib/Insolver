import pandas as pd
import numpy as np
import json
import pyodbc

from scripts.InsolverMain import InsolverMain


class InsolverDataFrame(InsolverMain):

    def __init__(
            self,
            data=None,
            driver='{SQL Server}',
            server=None,
            database=None,
            username=None,
            password=None,
            table=None
    ):

        self._is_frame = False

        if type(data) == pd.DataFrame:
            self._df = self._load_pd(data)
            if hasattr(self, '_df'):
                if self._df is not None:
                    self._is_frame = True

        elif type(data) == str and data.endswith('.csv'):
            self._df = self._load_csv(data)
            if hasattr(self, '_df'):
                if self._df is not None:
                    self._is_frame = True

        elif server is not None and database is not None and table is not None:
            self._df = self._load_mssql(driver, server, database, username, password, table)
            if hasattr(self, '_df'):
                if self._df is not None:
                    self._is_frame = True

    # ---------------------------------------------------
    # Load data methods
    # ---------------------------------------------------

    @staticmethod
    def _load_pd(pd_dataframe):
        """
        Loads data from Pandas Dataframe.

        :param pd_dataframe: Pandas Dataframe.
        :returns: Pandas Dataframe.
        """
        return pd_dataframe

    @staticmethod
    def _load_csv(csv_dataframe):
        """
        Loads data from .csv file.

        :param csv_dataframe: Path to .csv file.
        :returns: Pandas Dataframe.
        """
        return pd.read_csv(csv_dataframe, low_memory=False)

    @staticmethod
    def _load_mssql(driver, server, database, username, password, table):
        """
        Loads data from Microsoft SQL Server.

        :param driver: SQL Server Driver.
        :param server: Server name.
        :param database: Database name.
        :param username: Username.
        :param password: Password.
        :param table: Table name.
        :returns: Pandas Dataframe.
        """
        cnxn = pyodbc.connect(
            'DRIVER=' + driver + ';SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        try:
            df = pd.read_sql('select * from ' + table, cnxn)
        except Exception:
            df = None
        finally:
            cnxn.close()
        return df

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
        return self._df[columns].copy()

    # ---------------------------------------------------
    # Columns check methods
    # ---------------------------------------------------

    def columns_set(self, columns=None):
        """
        Sets the list of useful columns.

        :param columns: Dict or JSON-formatted string with columns description.
        :returns: None.
        """
        if columns == None:
            self._df_columns = {
                'json': '_df_columns',
                'columns': [
                    {'name': 'driver_minage', 'type': 'number'},
                    {'name': 'driver_minexp', 'type': 'number'},
                    {'name': 'driver_maxkbm', 'type': 'number'},

                    {'name': 'client_type', 'type': 'str', 'values': ['company', 'person']},
                    {'name': 'client_name', 'type': 'str'},  # 'Иванов Иван Иванович'
                    {'name': 'client_date_birth', 'type': 'datetime'},
                    {'name': 'client_gender', 'type': 'str', 'values': ['male', 'female']},

                    {'name': 'vehicle_power', 'type': 'number'},
                    {'name':
                         ['vehicle_issue_year', 'vehicle_age'],
                     'type':
                         ['number', 'number']
                     },
                    {'name': 'vehicle_type', 'type': 'number'},

                    {'name': 'p_date_start', 'type': 'datetime'},
                    {'name': 'p_is_taxi', 'type': 'number', 'values': [0, 1]},
                    {'name': 'p_is_driver_unlimit', 'type': 'number', 'values': [0, 1]},
                    {'name': 'kladr', 'type': 'str'},

                    {'name': 'p_claims_sum_infl', 'type': 'number'},
                    {'name': 'p_claims_count_adj', 'type': 'number'},
                ]
            }

        else:
            if type(columns) == dict:
                self._df_columns = columns
            elif type(columns) == str:
                self._df_columns = json.loads(columns)

    def columns_check(self):
        """
        Checks if the columns in dataframe are equal to the list of usefull columns.

        :returns: JSON with columns' check info.
        """
        if not hasattr(self, '_df_columns'):
            self.columns_set()

        _columns_check = json.loads('{"json": "_columns_check"}')

        for n in range(len(self._df_columns['columns'])):

            _col = ''
            _col_exists = False
            _col_type = False
            _col_values = True

            if type(self._df_columns['columns'][n]['name']) == str:

                _col = self._df_columns['columns'][n]['name']

                # exists
                if self._df_columns['columns'][n]['name'] in list(self._df.columns):
                    _col_exists = True

                    # type
                    if self._df_columns['columns'][n]['type'] == 'number':
                        if self._df_columns['columns'][n]['name'] in list(
                                self._df.select_dtypes(include=['int32', 'int64', 'float64', 'int64']).columns):
                            _col_type = True
                    elif self._df_columns['columns'][n]['type'] == 'str':
                        if self._df_columns['columns'][n]['name'] in list(
                                self._df.select_dtypes(include=['object']).columns):
                            _col_type = True
                    elif self._df_columns['columns'][n]['type'] == 'datetime':
                        if self._df_columns['columns'][n]['name'] in list(
                                self._df.select_dtypes(include=['datetime64']).columns):
                            _col_type = True

                # values
                if 'values' in self._df_columns['columns'][n].keys():
                    if _col_type == True and not self._df_columns['columns'][n]['values'] == None:
                        for u in self._df[self._df_columns['columns'][n]['name']].unique():
                            if u not in self._df_columns['columns'][n]['values']:
                                _col_values = False
                                break

            # if only one column from the list could exists
            elif type(self._df_columns['columns'][n]['name']) == list:

                _col_exists_ = 0
                _col_type_ = 0
                _col_values_ = 0

                for i in range(len(self._df_columns['columns'][n]['name'])):

                    _col = _col + self._df_columns['columns'][n]['name'][i] + ' _or_ '

                    # exists
                    if self._df_columns['columns'][n]['name'][i] in list(self._df.columns):
                        _col_exists_ += 1

                        # type
                        _col_type_add = 0
                        if self._df_columns['columns'][n]['type'][i] == 'number':
                            if self._df_columns['columns'][n]['name'][i] in list(
                                    self._df.select_dtypes(include=['int32', 'int64', 'float64', 'int64']).columns):
                                _col_type_ += 1
                                _col_type_add = 1
                        elif self._df_columns['columns'][n]['type'][i] == 'str':
                            if self._df_columns['columns'][n]['name'][i] in list(
                                    self._df.select_dtypes(include=['object']).columns):
                                _col_type_ += 1
                                _col_type_add = 1
                        elif self._df_columns['columns'][n]['type'][i] == 'datetime':
                            if self._df_columns['columns'][n]['name'][i] in list(
                                    self._df.select_dtypes(include=['datetime64']).columns):
                                _col_type_ += 1
                                _col_type_add = 1

                        # values
                        if 'values' in self._df_columns['columns'][n].keys():
                            if _col_type_add == 1 and not self._df_columns['columns'][n]['values'][i] == None:
                                for i in range(len(self._df_columns['columns'][n]['name'])):
                                    for u in self._df[self._df_columns['columns'][n]['name'][i]].unique():
                                        if u not in self._df_columns['columns'][n]['values'][i]:
                                            _col_values_ += 1
                                            break

                if _col_exists_ > 0:
                    _col_exists = True
                if _col_type_ > 0:
                    _col_type = True
                if _col_values_ > 0:
                    _col_values = True

            if _col_type == False:
                _col_values = False

            _columns_check.update({_col: {'exists': _col_exists, 'type': _col_type, 'values': _col_values}})

        return _columns_check

    # ---------------------------------------------------
    # Columns match methods
    # ---------------------------------------------------

    def columns_match(self, match_from_to):
        """
        Matches columns in dataframe.

        :param match_from_to: Matching dict.
        :returns: None.
        """
        self._df.rename(columns=match_from_to, inplace=True)

    # ---------------------------------------------------
    # General methods
    # ---------------------------------------------------

    def info(self):
        return self._df.info()

    def head(self, n=5):
        return self._df.head(n)

    def len(self):
        return len(self._df)