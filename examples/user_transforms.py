import pandas as pd
from insolver.transforms import TransformExp


# User changes for the existing transform
class TransformExp(TransformExp):
    @staticmethod
    def exp_new(exp, exp_max):
        if pd.isnull(exp):
            exp = None
        elif exp < 0:
            exp = None
        else:
            exp = exp * 30 // 365
        if exp > exp_max:
            exp = exp_max
        return exp


# New user transform
class TransformSocioCateg:
    def __init__(self, column_socio_categ):
        self.priority = 0
        self.column_socio_categ = column_socio_categ

    def __call__(self, df):
        df[self.column_socio_categ] = df[self.column_socio_categ].str.slice(0, 4)
        return df


class TransformToNumeric:
    """Example of user-defined transformations. Transform values to numeric.

    Parameters:
        column_names (list): List of columns for transformations
        downcast (str): parameter from pd.to_numeric, default: 'float'
    """
    def __init__(self, column_names, downcast='float'):
        self.column_names = column_names
        self.downcast = downcast

    def __call__(self, df):
        for column in self.column_names:
            df[column] = pd.to_numeric(df[column], downcast=self.downcast)
        return df
