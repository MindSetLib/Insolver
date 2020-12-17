import platform

import pyodbc
from pandas import read_sql


class PostgresConnection:
    """Class for setting the connection to PostreSQL database.

    Attributes:
        database (str): Database name for Database connection.
        uid (str): User ID for Database connection.
        pwd (str): Password for Database connection.
        server (str): Server name for Database connection.
        port (str): Port number for Database connection.
        driver (:obj:`str`, optional): Driver name for Database connection (default=None).
        decimal_separator (:obj:`str`, optional): Separator of decimal numbers in Database (default=',').
    """
    def __init__(self, database, uid, pwd, server, port, driver=None, decimal_separator=','):
        self.database = database
        self.uid = uid
        self.pwd = pwd
        self.server = server
        self.port = port
        self.decimal = decimal_separator
        if driver is None:
            if platform.system() == 'Windows':
                self.driver = '{PostgreSQL ANSI(x64)}'
            else:
                self.driver = '{PostgreSQL Unicode}'

        self.conn_str = (f'DRIVER={self.driver};DATABASE={self.database};UID={self.uid};'
                         f'PWD={self.pwd};SERVER={self.server};PORT={self.port};')

    def execute_query(self, query):
        """Method executing SQL queries, extracting the results to the pd.DataFrame object.

        Args:
            query (str): String containing text of the SQL query.

        Returns:
            pd.DataFrame: DataFrame containing SQL query results if it was successful.
        """
        pyodbc.setDecimalSeparator(self.decimal)
        conn = pyodbc.connect(self.conn_str)
        df = read_sql(query, conn)
        conn.close()
        return df
