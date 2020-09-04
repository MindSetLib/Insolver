import pandas as pd

from scripts.InsolverMain import InsolverMain


class InsolverDataFrame(InsolverMain):
    """
    Primary dataframe class for Insolver.

    :param data: Pandas' DataFrame.
    """

    def __init__(self, df):
        self._df = pd.DataFrame
        self._is_frame = False
        if isinstance(df, pd.DataFrame):
            self._df = df
            if self._df is not None:
                self._is_frame = True

    def get_data(self, columns=None):
        """
        Gets data as InsolverDataFrame.

        :param columns: Columns of InsolverDataFrame to get.
        :returns: InsolverDataFrame.
        """
        if self._is_frame is None:
            return None
        if columns is None:
            columns = self._df.columns
        return self._df[columns].copy()

    def get_meta_info(self):
        """
        Gets JSON with Insolver meta information.

        :return: JSON.
        """
        if self._is_frame is False:
            _meta_json = {
                'type': 'No data loaded'
            }
        else:
            _meta_json = {
                'type': 'InsolverDataFrame',
                'columns': self._df.columns,
                'len': len(self._df),
            }
        return _meta_json

    # ---------------------------------------------------
    # Columns match methods
    # ---------------------------------------------------

    def columns_match(self, match_from_to):
        """
        Matches columns in InsolverDataFrame.

        :param match_from_to: Matching dict.
        :returns: None.
        """
        self._df.rename(columns=match_from_to, inplace=True)

    # ---------------------------------------------------
    # General methods
    # ---------------------------------------------------

    def info(self):
        return self._df.info()

    def head(self, n=5):
        return self._df.head(n)

    def __len__(self):
        return len(self._df)

    def columns(self):
        return self._df.columns
