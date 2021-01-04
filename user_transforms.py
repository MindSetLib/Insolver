import pandas as pd


class TransformToNumeric:
    """Example of user-defined transformations. Transform values to numeric.

    Attributes:
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
