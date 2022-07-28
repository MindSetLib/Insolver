from numpy import where
from pandas import to_datetime, Timedelta


class DatetimeTransforms:
    """Get selected feature from date variable.

    Parameters:
        column_names (list): List of columns to convert, columns in column_names can't be duplicated in column_feature.
        column_types (dict): Dictionary of columns and types to return.
        dayfirst (bool): Parameter from pandas.to_datetime(), specify a date parse order if arg is str or its
         list-likes.
        yearfirst (bool): Parameter from pandas.to_datetime(), specify a date parse order if arg is str or its
         list-likes.
        feature (str): Type of feature to get from date variable: unix (by default), date, time, month, quarter, year,
         day, day_of_the_week, weekend.
        column_feature (dict): List of columns to preprocess using specified feature for each column in the dictionary,
         columns in column_feature can't be duplicated in column_names.

    """

    def __init__(
        self,
        column_names,
        column_types=None,
        dayfirst=False,
        yearfirst=False,
        feature='unix',
        column_feature=None,
        priority=0,
    ):
        self.priority = priority
        self.feature = feature
        self.column_names = column_names
        self.column_types = column_types
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self._feature_types = ['unix', 'date', 'time', 'month', 'quarter', 'year', 'day', 'day_of_the_week', 'weekend']
        self.column_feature = column_feature

    def _get_date_feature(self, df):
        self.feature_dict = {
            'unix': lambda col: (col - to_datetime("1970-01-01")) // Timedelta('1s'),
            'date': lambda col: col.dt.date,
            'time': lambda col: col.dt.time,
            'month': lambda col: col.dt.month,
            'quarter': lambda col: col.dt.quarter,
            'year': lambda col: col.dt.year,
            'day': lambda col: col.dt.day,
            'day_of_the_week': lambda col: col.dt.dayofweek,
            'weekend': lambda col: where(col.dt.day_name().isin(['Sunday', 'Saturday']), 1, 0),
        }

        if self.column_feature:
            for column in self.column_feature.keys():
                if column in self.column_names:
                    raise Exception(
                        f'Columns in column_feature{list(self.column_feature.keys())}'
                        f'cannot be duplicated in column_names{self.column_names}'
                    )

                else:
                    _col_feature = self.column_feature[column]
                    type_of_column = self.column_types[column] if column in self.column_types.keys() else None
                    if type_of_column:
                        df[f'{column}_{_col_feature}'] = self.feature_dict[_col_feature](
                            to_datetime(df[column], dayfirst=self.dayfirst, yearfirst=self.yearfirst)
                        ).astype(type_of_column)
                    else:
                        df[f'{column}_{_col_feature}'] = self.feature_dict[_col_feature](
                            to_datetime(df[column], dayfirst=self.dayfirst, yearfirst=self.yearfirst)
                        )

        if self.feature in self._feature_types:
            if self.column_types:
                for column in self.column_names:
                    type_of_column = self.column_types[column] if column in self.column_types.keys() else None
                    if type_of_column:
                        df[f'{column}_{self.feature}'] = self.feature_dict[self.feature](
                            to_datetime(df[column], dayfirst=self.dayfirst, yearfirst=self.yearfirst)
                        ).astype(type_of_column)
                    else:
                        df[f'{column}_{self.feature}'] = self.feature_dict[self.feature](
                            to_datetime(df[column], dayfirst=self.dayfirst, yearfirst=self.yearfirst)
                        )
            else:
                for column in self.column_names:
                    df[f'{column}_{self.feature}'] = self.feature_dict[self.feature](
                        to_datetime(df[column], dayfirst=self.dayfirst, yearfirst=self.yearfirst)
                    )

        else:
            raise NotImplementedError(f'Method parameter supports values in {self._feature_types}.')

    def __call__(self, df):
        self._get_date_feature(df)
        return df
