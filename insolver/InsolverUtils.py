import platform

import pyodbc
from pandas import read_sql

import insolver


def load_class(module_list, transform_name):
    for module in module_list:
        try:
            transform_class = getattr(module, transform_name)
            return transform_class
        except AttributeError:
            pass


def init_transforms(transforms):
    """Function for creation transformations objects from the dictionary.

    Args:
        transforms: dictionary with classes and their init parameters

    Returns:
        list: List of transformations objects.
    """
    transforms_list = []
    module_list = [insolver.InsolverTransforms]

    try:
        import user_transforms
        module_list.append(user_transforms)

    except ModuleNotFoundError:
        pass

    for transform_name in transforms:
        try:
            del transforms[transform_name]['priority']
        except KeyError:
            pass

        transform_class = load_class(module_list, transform_name)
        if transform_class:
            transforms_list.append(transform_class(**transforms[transform_name]))

    return transforms_list


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
