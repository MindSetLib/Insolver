import warnings
import platform
import pyodbc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import DMatrix
from lightgbm import Dataset


def gb_eval_dev_poisson(yhat, y, weight=None):
    """Function for Poisson Deviance evaluation.

    Args:
        yhat: np.ndarray object with predictions.
        y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable.
        weight: Weights for weighted metric.

    Returns:
        (str, float), tuple with metrics name and its value, if y is xgboost.DMatrix or lightgbm.Dataset;
        float, otherwise.
    """
    that = yhat + 1
    if isinstance(y, (DMatrix, Dataset)):
        t = y.get_label() + 1
        if isinstance(y, DMatrix):
            return 'dev_poisson', 2 * np.sum(t * np.log(t / that) - (t - that))
        if isinstance(y, Dataset):
            return 'dev_poisson', 2 * np.sum(t * np.log(t / that) - (t - that)), False
    else:
        t = y + 1
        if weight:
            return 2 * np.sum(weight*(t * np.log(t / that) - (t - that)))
        else:
            return 2 * np.sum(t * np.log(t / that) - (t - that))


def gb_eval_dev_gamma(yhat, y, weight=None):
    """Function for Gamma Deviance evaluation.

    Args:
        yhat: np.ndarray object with predictions.
        y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable.
        weight: Weights for weighted metric.

    Returns:
        (str, float), tuple with metrics name and its value.
    """
    if isinstance(y, (DMatrix, Dataset)):
        t = y.get_label()
        if isinstance(y, DMatrix):
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat)
        if isinstance(y, Dataset):
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat), False
    else:
        if weight:
            return 2 * np.sum(weight*(-np.log(y/yhat) + (y-yhat)/yhat))
        else:
            return 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)


def train_val_test_split(x, y, val_size, test_size, random_state=0, shuffle=True, stratify=None):
    """Function for splitting dataset into train/validation/test partitions.

    Args:
        x (array_like): Array containing predictors.
        y (array_like): Array containing target variable.
        val_size (float): The proportion of the dataset to include in validation partition.
        test_size (float): The proportion of the dataset to include in test partition.
        random_state (:obj:`int`, optional): Random state, passed to train_test_split() from scikit-learn. (default=0).
        shuffle (:obj:`bool`, optional): Passed to train_test_split() from scikit-learn. (default=True).
        stratify (:obj:`array_like`, optional): Passed to train_test_split() from scikit-learn. (default=None).

    Returns:
        tuple: (x_train, x_valid, x_test, y_train, y_valid, y_test).

        A tuple of partitions of the initial dataset.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, shuffle=shuffle,
                                                        test_size=test_size, stratify=stratify)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=random_state, shuffle=shuffle,
                                                          test_size=val_size/(1-test_size), stratify=stratify)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def train_test_column_split(x, y, df_column):
    """Function for splitting dataset into train/test partitions w.r.t. a column (pd.Series).

    Args:
        x (pd.DataFrame): DataFrame containing predictors.
        y (pd.DataFrame): DataFrame containing target variable.
        df_column (pd.Series): Series for train/test split, assuming it is contained in x.

    Returns:
        tuple: (x_train, x_test, y_train, y_test).

        A tuple of partitions of the initial dataset.
    """
    x1, y1, col_name = x.copy(), y.copy(), df_column.name
    y1[col_name] = df_column
    return (x1[x1[col_name] == 'train'].drop(col_name, axis=1), x1[x1[col_name] == 'test'].drop(col_name, axis=1),
            y1[y1[col_name] == 'train'].drop(col_name, axis=1), y1[y1[col_name] == 'test'].drop(col_name, axis=1))


class PostgresConnection(object):
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
            elif platform.system() == 'Linux':
                self.driver = '{PostgreSQL Unicode}'
            else:
                warnings.warn('If this does not work, please, specify the correct driver manually.')
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
        df = pd.read_sql(query, conn)
        conn.close()
        return df


class PredictionMetrics(object):
    # TODO: Support of train/val/test splits.
    """Class for calculating metrics using predictions of the models.

    Attributes:
        df_predictions (pd.DataFrame): DataFrame containing all predictions that are used to metrics calculation.
        df_targets (pd.DataFrame): DataFrame containing all targets corresponding to df_predictions.
        target_list (list): List of target names from df_targets for every prediction in df_predictions.
    """
    def __init__(self, df_predictions, df_targets, target_list):
        self.predictions = df_predictions
        self.targets = df_targets
        self.target_list = target_list

    def make_summary(self, function):
        metric_name = function.__name__
        metric = pd.DataFrame()
        for i in range(len(self.predictions.columns)):
            pred_name = self.predictions.columns[i]
            metric[f'{pred_name}'] = [function(self.predictions[pred_name],
                                               self.targets[self.target_list[i]]),
                                      self.targets[self.target_list[i]].mean(),
                                      self.predictions[pred_name].mean()]
        metric.index = [metric_name, 'Mean target', 'Mean prediction']
        return metric.T
