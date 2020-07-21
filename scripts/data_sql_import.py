import platform
import pandas as pd
import pyodbc


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
                print('If this does not work, specify the correct driver manually')
                self.driver = "{PostgreSQL Unicode}"

        self.conn_str = '''DRIVER={};DATABASE={};UID={};
                           PWD={};SERVER={};PORT={};'''.format(self.driver, self.database, self.uid,
                                                               self.pwd, self.server, self.port)

    def execute_query(self, query):
        pyodbc.setDecimalSeparator(self.decimal)
        conn = pyodbc.connect(self.conn_str)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
