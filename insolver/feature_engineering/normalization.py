import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, Normalizer


class Normalization:
    """
    Normalization.

    Parameters:
        method (str): Normalization method. Values `standard`, `minmax`, `robust`, `normalizer`, `yeo-johnson`,
         `box-cox`, `log` are supported. If None, values in column_method will only be transformed.
        column_names (str, list): Name or names of columns to transform. If None, all columns will be transformed.
        column_method (dict): Dictionary of columns to preprocess using specified method for each column
         ({column name: method}), columns in column_method can't be duplicated in column_names.

    Attributes:
        new_df (pandas.DataFrame): A new dataframe as a copy of the original data with a transformed columns.

    """

    def __init__(self, method='standard', column_names=None, column_method=None):
        self.method = method
        self.column_names = column_names
        self.column_method = column_method
        self.new_df = pd.DataFrame()
        self.old_df = pd.DataFrame()

    def transform(self, data):
        """
        Main Normalization method.
        It creates new `pandas.DataFrame` as a copy of the original data and transformes either the selected or all
        columns.

        Parameters:
            data (pandas.Dataframe, optional): Original data. If not `pandas.DataFrame`, new dataframe will be created.

        Raises:
            NotImplementedError: If method is not supported.
            Exception: If columns in column_method are duplicated in column_names.
        """
        self._init_methods_dict()

        if self.method:
            if self.method not in self.methods_dict.keys():
                raise NotImplementedError(f'Method {self.method} is not supported.')

        if isinstance(data, pd.DataFrame):
            self.new_df = data.copy()
        else:
            self.new_df = pd.DataFrame(data)

        self.old_df = self.new_df.copy()

        # if self.column_method is a dict, check columns in self.column_names and transform
        if isinstance(self.column_method, dict):
            for column in self.column_method:
                if self.column_names:
                    if column in self.column_names:
                        raise Exception('Columns in column_method cannot be duplicated in column_names')

                # reshape to prevent "ValueError: Expected 2D array, got 1D array instead"
                old_column = self.new_df[column].to_numpy().reshape(-1, 1)
                self.new_df[column] = self._transform_data(old_column, self.column_method[column])

        # if self.column_names is a str, change only this column
        if isinstance(self.column_names, str) and self.method:
            old_column = self.new_df[self.column_names].to_numpy().reshape(-1, 1)
            self.new_df[self.column_names] = self._transform_data(old_column, self.method)

        # if self.column_names is a list, change only columns from the list
        elif isinstance(self.column_names, list) and self.method:
            for column in self.column_names:
                old_column = self.new_df[column].to_numpy().reshape(-1, 1)
                self.new_df[column] = self._transform_data(old_column, self.method)

        else:
            # if self.column_method and self.method transform all columns except ones in self.column_method
            if self.column_method and self.method:
                for column in self.new_df.columns:
                    if column not in self.column_method:
                        old_column = self.new_df[column].to_numpy().reshape(-1, 1)
                        self.new_df[column] = self._transform_data(old_column, self.method)

            # else transform all if self.method
            elif self.method:
                self.new_df = self._transform_data(self.new_df, self.method)

        return self.new_df

    def plot_transformed(self, column, **kwargs):
        """
        Plot old and new transformed column.

        Note:
            This method can be called only after method `transform` has been called.

        Parameters:
            column (str): Column name.
            **kwargs: Arguments for the `seaborn.displot` function.

        Raises:
            Exception: If method is called before transform() method.
        """
        if (len(self.old_df) == 0) and (len(self.new_df) == 0):
            raise Exception('Dataframes were not created yet. Call transform() method.')

        # create new dataframe from old and new columns to plot with the hue
        old_df = pd.DataFrame(self.old_df[column])
        old_df['Column'] = 'old'
        new_df = pd.DataFrame(self.new_df[column])
        new_df['Column'] = 'new'
        self.df_to_plot = pd.concat([old_df, new_df], axis=0, ignore_index=True)

        sns.displot(data=self.df_to_plot, x=column, hue='Column', **kwargs)
        plt.title("Distribution plot", size=15, weight='bold')

    def _transform_data(self, new_data, method):
        """
        Transformation method. It transforms given DataFrame or column with the selected method.

        Parameters:
            new_data : New DataFrame or column to transform.
            method : The method to use in the transformation.
        """
        if method not in self.methods_dict.keys():
            raise NotImplementedError(f'Method {method} is not supported.')

        if method == 'log':
            new_data = np.log(new_data + 1)

        else:
            if isinstance(new_data, pd.DataFrame):
                estimator = self.methods_dict[method]
                new_data = pd.DataFrame(estimator.fit_transform(new_data))
                new_data.set_axis(self.new_df.columns, axis=1, inplace=True)

            else:
                estimator = self.methods_dict[method]
                new_data = estimator.fit_transform(new_data)

        return new_data

    def _init_methods_dict(self):
        """
        Methods dictionary initialization.

        """
        self.methods_dict = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'normalizer': Normalizer(),
            'yeo-johnson': PowerTransformer(method='yeo-johnson'),
            'box-cox': PowerTransformer(method='box-cox'),
            'log': 'log',
        }
