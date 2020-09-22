import pandas as pd
import numpy as np

from .InsolverMain import InsolverMain
from .InsolverUtils import train_val_test_split


class InsolverDataFrame(InsolverMain):
    """
    Primary dataframe class for Insolver.

    :param df: Pandas' DataFrame.
    """

    def __init__(self, df):
        self._df = pd.DataFrame
        self._is_frame = False
        self.categorical_columns = None
        self.numerical_columns = None
        self.binary_columns = None
        self.nonbinary_columns = None
        if isinstance(df, pd.DataFrame):
            self._df = df
            if self._df is not None:
                self._is_frame = True
        else:
            raise NotImplementedError("'df' should be the Pandas' DataFrame.")

    def get_data(self, columns=None):
        """
        Gets data as Pandas' DataFrame.

        :param columns: Columns to get.
        :returns: Pandas' DataFrame.
        """
        if self._is_frame is None:
            return None
        if columns is None:
            columns = self._df.columns
        return self._df[columns].copy()

    def save_data_to_csv(self, path, sep=',', columns=None, *args, **kwargs):
        """
        Saves data to .csv file. Uses Pandas' 'to_csv'.

        :param path: Path to the output file.
        :param sep: Field delimiter for the output file, ',' by default..
        :param columns: Columns to get.
        :returns: Path.
        """
        if self._is_frame is None:
            return None
        if columns is None:
            columns = self._df.columns
        self._df[columns].to_csv(path_or_buf=path, sep=sep, *args, **kwargs)
        return path

    def get_meta_info(self):
        """
        Gets JSON with Insolver meta information.

        :return: JSON.
        """
        if self._is_frame is False:
            meta_json = {
                'type': 'No data loaded'
            }
        else:
            meta_json = {
                'type': 'InsolverDataFrame',
                'len': len(self._df),
                'columns': [],
            }
            for column in self._df.columns:
                meta_json['columns'].append({'name': column, 'dtype': self._df[column].dtypes, 'use': 'unknown'})
        return meta_json

    def split_frame(self, val_size, test_size, random_state=0, shuffle=True, stratify=None):
        return train_val_test_split(self._df, val_size=val_size, test_size=test_size, random_state=random_state,
                                    shuffle=shuffle, stratify=stratify)

    # ---------------------------------------------------
    # Data cleaning methods
    # ---------------------------------------------------

    def fillna(self, *args, **kwargs):
        return self._df.fillna(*args, **kwargs)

    def dropna(self, *args, **kwargs):
        return self._df.dropna(*args, **kwargs)

    def find_num_cat_features(self):
        self.categorical_columns = [c for c in self._df.columns if self._df[c].dtype.name == 'object']
        self.numerical_columns = [c for c in self._df.columns if self._df[c].dtype.name != 'object']

    def find_binary_features(self):
        self.data_describe = self._df.describe(include=[object])
        self.binary_columns = [c for c in self.categorical_columns if self.data_describe[c]['unique'] == 2]
        self.nonbinary_columns = [c for c in self.categorical_columns if self.data_describe[c]['unique'] > 2]

    def fillna_binary_features(self):
        for c in self.binary_columns[1:]:
            top = self.data_describe[c]['top']
            top_items = self._df[c] == top
            self._df.loc[top_items, c] = 0
            self._df.loc[np.logical_not(top_items), c] = 1

    def fillna_not_binary_features(self):
        pd.get_dummies(self._df[self.nonbinary_columns])

    def fillnan_category(self, col_name):
        """Replace nan values with most occured category"""
        most_frequent_category = self._df[col_name].mode()[0]
        self._df[col_name].fillna(most_frequent_category, inplace=True)

    def fillnan_category_imp(self, col_name):
        """
        Replace NAN categories with most occurred values,
        and add a new feature to introduce some weight/importance
        to non-imputed and imputed observations.
        :param col_name:
        :return:
        """
        # 1. add new column and replace if category is null then 1 else 0
        self._df[col_name + "_Imputed"] = np.where(self._df[col_name].isnull(), 1, 0)
        mode_category = self._df[col_name].mode()[0]
        ##2.1 Replace NAN values with most occured category in actual vairable
        self._df[col_name].fillna(mode_category, inplace=True)

    def fillnan_category_with_unknown(self, col_name):
        self._df[col_name] = np.where(self._df[col_name].isnull(), "Unknown", self._df[col_name])

    # ---------------------------------------------------
    # General methods
    # ---------------------------------------------------

    def columns_match(self, match_from_to, *args, **kwargs):
        """
        Matches columns in InsolverDataFrame. Uses Pandas' 'rename'.

        :param match_from_to: Matching dict.
        :returns: None.
        """
        self._df.rename(columns=match_from_to, inplace=True, *args, **kwargs)

    def info(self):
        return self._df.info()

    def head(self, n=5):
        return self._df.head(n)

    def __len__(self):
        return len(self._df)

    def columns(self):
        return self._df.columns
