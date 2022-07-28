from numpy import where


class AutoFillNATransforms:
    """Auto Fill NA values.

    Parameters:
        numerical_columns (list): List of numerical columns
        categorical_columns (list): List of categorical columns
        numerical_method (str): Fill numerical NA values using this specified method: 'median' (by default), 'mean',
          'mode' or 'remove'
        categorical_method (str): Fill categorical NA values using this specified method: 'frequent' (by default),
          'new_category', 'imputed_column' or 'remove'
        numerical_constants (dict): Dictionary of constants for each numerical column
        categorical_constants (dict): Dictionary of constants for each categorical column
    """

    def __init__(
        self,
        numerical_columns=None,
        categorical_columns=None,
        numerical_method='median',
        categorical_method='frequent',
        numerical_constants=None,
        categorical_constants=None,
        priority=0,
    ):
        self.priority = priority
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.numerical_constants = numerical_constants
        self.categorical_constants = categorical_constants
        self._num_methods = ['median', 'mean', 'mode', 'remove']
        self._cat_methods = ['frequent', 'new_category', 'imputed_column', 'remove']
        self.numerical_method = numerical_method
        self.categorical_method = categorical_method

    def _find_num_cat_features(self, df):
        if not self.categorical_columns:
            self.categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']
        if not self.numerical_columns:
            self.numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']

    def _fillna_numerical(self, df):
        """Replace numerical NaN values using specified method"""

        if not self.numerical_columns:
            return

        if self.numerical_method == 'remove':
            df.dropna(subset=self.numerical_columns, inplace=True)
            return

        if self.numerical_constants:
            for column in self.numerical_constants.keys():
                df[column].fillna(self.numerical_constants[column], inplace=True)

        if self.numerical_method in self._num_methods:
            self._num_methods_dict = {
                'median': lambda col: df[col].median(),
                'mean': lambda col: df[col].mean(),
                'mode': lambda col: df[col].mode()[0],
            }

            self.values = {}
            for column in self.numerical_columns:
                if df[column].isnull().all():
                    self.values[column] = 1
                else:
                    self.values[column] = self._num_methods_dict[self.numerical_method](column)

                df[column].fillna(self.values[column], inplace=True)
        else:
            raise NotImplementedError(f'Method parameter supports values in {self._num_methods}.')

    def _fillnan_categorical(self, df):
        """Replace categorical NaN values using specified method"""

        if not self.categorical_columns:
            return

        if self.categorical_method == 'remove':
            df.dropna(subset=self.categorical_columns, inplace=True)
            return

        if self.categorical_constants:
            for column in self.categorical_constants.keys():
                df[column].fillna(self.categorical_constants[column], inplace=True)

        if self.categorical_method in self._cat_methods:
            if self.categorical_method == 'new_category':
                for column in self.categorical_columns:
                    df[column].fillna('Unknown', inplace=True)
                return

            if self.categorical_method == 'imputed_column':
                for column in self.categorical_columns:
                    df[f"{column}_Imputed"] = where(df[column].isnull(), 1, 0)

            self.freq_categories = {}
            for column in self.categorical_columns:
                if df[column].mode().values.size > 0:
                    self.freq_categories[column] = df[column].mode()[0]
                else:
                    self.freq_categories[column] = 1

                df[column].fillna(self.freq_categories[column], inplace=True)

        else:
            raise NotImplementedError(f'Method parameter supports values in {self._cat_methods}.')

    def __call__(self, df):
        self._find_num_cat_features(df)
        self._fillna_numerical(df)
        self._fillnan_categorical(df)
        return df
