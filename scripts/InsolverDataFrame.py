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
        self._df = pd.DataFrame
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
