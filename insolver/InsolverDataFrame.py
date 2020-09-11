import pandas as pd

from .InsolverMain import InsolverMain


class InsolverDataFrame(InsolverMain):
    """
    Primary dataframe class for Insolver.

    :param df: Pandas' DataFrame.
    """

    def __init__(self, df):
        self._df = pd.DataFrame
        self._is_frame = False
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
            for c in self._df.columns:
                meta_json['columns'].append({'name': c, 'dtype': self._df[c].dtypes, 'use': 'unknown'})
        return meta_json

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
