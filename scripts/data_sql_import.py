import platform
import pyodbc
import pandas as pd


class PostgresConnection(object):
    def __init__(self, database, uid, pwd, server, port, driver=None, decimal_separator=','):
        self.database = database
        self.uid = uid
        self.pwd = pwd
        self.server = server
        self.port = port
        self.decimal = decimal_separator
        if driver is None:
            if platform.system() == "Windows":
                self.driver = "{PostgreSQL ANSI(x64)}"
            elif platform.system() == "Linux":
                self.driver = "{PostgreSQL Unicode}"
            else:
                print('If this does not work, please, specify the correct driver manually')
                self.driver = "{PostgreSQL Unicode}"

        self.conn_str = (f'DRIVER={self.driver};DATABASE={self.database};UID={self.uid};'
                         f'PWD={self.pwd};SERVER={self.server};PORT={self.port};')

    def execute_query(self, query):
        """
        Method executing SQL queries, extracting the results to the pd.DataFrame object.

        :param query: str, string containing text of the SQL query
        :return: pd.DataFrame, containing SQL query results if it was successful.
        """
        pyodbc.setDecimalSeparator(self.decimal)
        conn = pyodbc.connect(self.conn_str)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
