from pandas import DataFrame

from .InsolverMain import InsolverMain
from .model_tools import train_val_test_split


class InsolverDataFrame(InsolverMain):
    """Primary DataFrame class for Insolver.

    Args:
        df (:obj:`pd.DataFrame`): pandas DataFrame.
    """

    def __init__(self, df):
        self._df = DataFrame
        self._is_frame = False
        if isinstance(df, DataFrame):
            self._df = df
            if self._df is not None:
                self._is_frame = True
        else:
            raise NotImplementedError("'df' should be the Pandas' DataFrame.")

    def get_data(self, columns=None):
        """Gets data as pandas DataFrame.

        Args:
            columns: Columns to get.

        Returns:
            :obj:`pd.DataFrame`.
        """
        if self._is_frame is None:
            return None
        if columns is None:
            columns = self._df.columns
        return self._df[columns].copy()

    def save_data_to_csv(self, path, sep=',', columns=None, *args, **kwargs):
        """Saves data to .csv file. Uses pandas 'to_csv'.

        Args:
            path (str): Path to the output file.
            sep (str): Field delimiter for the output file, ',' by default.
            columns: Columns to get.
            *args:
            **kwargs:

        Returns:
            str: Path to the output file.
        """
        if self._is_frame is None:
            return None
        if columns is None:
            columns = self._df.columns
        self._df[columns].to_csv(path_or_buf=path, sep=sep, *args, **kwargs)
        return path

    def get_meta_info(self):
        """Gets JSON with Insolver meta information.

        Returns:
            dict: Meta information JSON.
        """
        if self._is_frame is False:
            meta_json = {'type': 'No data loaded'}
        else:
            meta_json = {
                'type': 'InsolverDataFrame',
                'len': len(self._df),
                'columns': []
            }
            for column in self._df.columns:
                meta_json['columns'].append({'name': column, 'dtype': self._df[column].dtypes, 'use': 'unknown'})
        return meta_json

    def split_frame(self, val_size, test_size, random_state=0, shuffle=True, stratify=None):
        return train_val_test_split(self._df, val_size=val_size, test_size=test_size, random_state=random_state,
                                    shuffle=shuffle, stratify=stratify)

    # ---------------------------------------------------
    # General methods
    # ---------------------------------------------------

    def columns_match(self, match_from_to, *args, **kwargs):
        """Matches columns in InsolverDataFrame. Uses Pandas' 'rename'.

        Args:
            match_from_to (dict): Matching dict.
            *args: Other arguments.
            **kwargs: Other keyword arguments.
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
