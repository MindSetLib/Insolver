import pandas as pd
import json
import pyodbc

from scripts.InsolverMain import InsolverMain


class InsolverDataFrame(InsolverMain):
    """
    Primary dataframe class for Insolver. Is similar to pandas' DataFrame.
    """
    def __init__(
            self,
            data=None,
            sep=',',
            encoding=None,
            driver='{SQL Server}',
            server=None,
            database=None,
            username=None,
            password=None,
            table=None
    ):

        self._is_frame = False

        if isinstance(data, pd.DataFrame):
            self.load_pd(data)

        elif isinstance(data, str):
            if data.endswith('.csv'):
                self.load_csv(data, sep, encoding)

        elif server is not None and database is not None and table is not None:
            self.load_mssql(driver, server, database, username, password, table)

    # ---------------------------------------------------
    # Load data methods
    # ---------------------------------------------------

    def load_pd(self, pd_dataframe):
        """
        Loads data from Pandas' Dataframe.

        :param pd_dataframe: Pandas' Dataframe.
        :returns: is_frame.
        """
        self._df = pd_dataframe
        if self._df is not None:
            self._is_frame = True
        return f'is_frame={self._is_frame}'

    def load_csv(self, csv_dataframe, sep, encoding):
        """
        Loads data from .csv file, uses Pandas' 'read_csv'.

        :param csv_dataframe: Path to .csv file.
        :returns: is_frame.
        """
        self._df = pd.read_csv(csv_dataframe, sep=sep, encoding=encoding, low_memory=False)
        if self._df is not None:
            self._is_frame = True
        return f'is_frame={self._is_frame}'

    def load_mssql(self, driver, server, database, username, password, table):
        """
        Loads data from Microsoft SQL Server.

        :param driver: SQL Server Driver.
        :param server: Server name.
        :param database: Database name.
        :param username: Username.
        :param password: Password.
        :param table: Table name.
        :returns: is_frame.
        """
        cnxn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
        try:
            self._df = pd.read_sql('select * from ' + table, cnxn)
        except Exception:
            self._df = None
        finally:
            cnxn.close()
        if self._df is not None:
            self._is_frame = True
        return f'is_frame={self._is_frame}'

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

    _df_columns_default = {
        'json': '_df_columns',
        'columns': [
            {'name':
                 ['driver_minage', 'client_date_birth'],
             'type':
                 ['number', 'datetime']
             },
            {'name':
                 ['driver_minexp', 'client_date_drive_start'],
             'type':
                 ['number', 'datetime']
             },
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
            {'name':
                 ['p_date_end', 'p_exposure'],
             'type':
                 ['datetime', 'number']
             },
            {'name': 'p_is_taxi', 'type': 'number', 'values': [0, 1]},
            {'name': 'p_is_driver_unlimit', 'type': 'number', 'values': [0, 1]},
            {'name': 'kladr', 'type': 'str'},

            {'name': 'p_claims_sum_infl', 'type': 'number'},
            {'name': 'p_claims_count', 'type': 'number'},
            {'name': 'p_claims_count_adj', 'type': 'number'},
        ]
    }

    def columns_set(self, columns=None):
        """
        Sets the list of useful columns.

        :param columns: Dict or JSON-formatted string with columns description.
        :returns: None.
        """
        if columns is not None:
            if isinstance(columns, dict):
                self._df_columns = columns
            elif isinstance(columns, str):
                self._df_columns = json.loads(columns)

    def columns_check(self):
        """
        Checks if the columns in dataframe are equal to the list of usefull columns.

        :returns: JSON with columns' check info.
        """
        if not hasattr(self, '_df_columns'):
            self.columns_set()

        _columns_check = json.loads('{"json": "_columns_check"}')

        for column in self._df_columns['columns']:

            _col = ''
            _col_exists = False
            _col_type = False
            _col_values = True

            if isinstance(column['name'], str):

                _col = column['name']

                # exists
                if column['name'] in list(self._df.columns):
                    _col_exists = True

                    # type
                    if column['type'] == 'number':
                        if column['name'] in list(
                                self._df.select_dtypes(include=['int32', 'int64', 'float64', 'int64']).columns):
                            _col_type = True
                    elif column['type'] == 'str':
                        if column['name'] in list(
                                self._df.select_dtypes(include=['object']).columns):
                            _col_type = True
                    elif column['type'] == 'datetime':
                        if column['name'] in list(
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
            elif isinstance(column['name'], list):

                _col_exists_ = 0
                _col_type_ = 0
                _col_values_ = 0

                for i in range(len(column['name'])):

                    _col = _col + column['name'][i] + ' _or_ '

                    # exists
                    if column['name'][i] in list(self._df.columns):
                        _col_exists_ += 1

                        # type
                        _col_type_add = 0
                        if column['type'][i] == 'number':
                            if column['name'][i] in list(
                                    self._df.select_dtypes(include=['int32', 'int64', 'float64', 'int64']).columns):
                                _col_type_ += 1
                                _col_type_add = 1
                        elif column['type'][i] == 'str':
                            if column['name'][i] in list(
                                    self._df.select_dtypes(include=['object']).columns):
                                _col_type_ += 1
                                _col_type_add = 1
                        elif column['type'][i] == 'datetime':
                            if column['name'][i] in list(
                                    self._df.select_dtypes(include=['datetime64']).columns):
                                _col_type_ += 1
                                _col_type_add = 1

                        # values
                        if 'values' in column.keys():
                            if _col_type_add == 1 and not column['values'][i] is None:
                                for i in range(len(column['name'])):
                                    for u in self._df[column['name'][i]].unique():
                                        if u not in column['values'][i]:
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

    def __len__(self):
        return len(self._df)

    def columns(self):
        return self._df.columns
