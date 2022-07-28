from pandas import DataFrame, to_numeric, concat, get_dummies
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TransformToNumeric:
    """Transforms parameter's values to numeric types, uses Pandas' 'to_numeric'.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter to transform.
        downcast: Target numeric dtype, equal to Pandas' 'downcast' in the 'to_numeric' function, 'integer' by default.
    """

    def __init__(self, column_param, downcast='integer', priority=0):
        self.priority = priority
        self.column_param = column_param
        self.downcast = downcast

    def __call__(self, df):
        df[self.column_param] = to_numeric(df[self.column_param], downcast=self.downcast)
        return df


class TransformMapValues:
    """Transforms parameter's values according to the dictionary.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter to map.
        dictionary (dict): The dictionary for mapping.
    """

    def __init__(self, column_param, dictionary, priority=1):
        self.priority = priority
        self.column_param = column_param
        self.dictionary = dictionary

    def __call__(self, df):
        df[self.column_param] = df[self.column_param].map(self.dictionary)
        return df


class TransformPolynomizer:
    """Gets polynomials of parameter's values.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter to polynomize.
        n (int): Polynomial degree.
    """

    def __init__(self, column_param, n=2, priority=3):
        self.priority = priority
        self.column_param = column_param
        self.n = n

    def __call__(self, df):
        for i in range(2, self.n + 1):
            a = self.column_param + '_' + str(i)
            while a in list(df.columns):
                a = a + '_'
            df[a] = df[self.column_param] ** i
        return df


class TransformGetDummies:
    """Gets dummy columns of the parameter, uses Pandas' 'get_dummies'.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter to transform.
        drop_first (bool): Whether to get k-1 dummies out of k categorical levels by removing the first level,
            False by default.
        inference (bool): Sign if the transformation is used for inference, False by default.
        dummy_columns (list): List of the dummy columns, for inference only.
    """

    def __init__(self, column_param, drop_first=False, inference=False, dummy_columns=None, priority=3):
        self.priority = priority
        self.column_param = column_param
        self.drop_first = drop_first
        self.inference = inference
        if inference and dummy_columns is not None:
            self.dummy_columns = dummy_columns
        else:
            self.dummy_columns = []

    def __call__(self, df):
        if self.dummy_columns == list():
            df_dummy = get_dummies(df[[self.column_param]], prefix_sep='_', drop_first=self.drop_first)
            self.dummy_columns = list([col.replace(' ', '_') for col in df_dummy.columns])
            df_dummy.columns = self.dummy_columns
            df = concat([df, df_dummy], axis=1)
        else:
            for column in self.dummy_columns:
                df[column] = ((self.column_param + '_' + df[self.column_param]) == column).astype('int8')
        return df


class EncoderTransforms:
    """Label Encoder

    Parameters:
        column_names (list): columns for label encoding
        le_classes (dict): dictionary with label encoding classes for each column

    """

    def __init__(self, column_names, le_classes=None, priority=3):
        self.priority = priority
        self.column_names = column_names
        self.le_classes = le_classes

    @staticmethod
    def _encode_column(column):
        le = LabelEncoder()
        le.fit(column)
        le_classes = le.classes_.tolist()
        column = le.transform(column)
        return column, le_classes

    def __call__(self, df):
        self.le_classes = {}
        for column_name in self.column_names:
            df[column_name], self.le_classes[column_name] = self._encode_column(df[column_name])
        return df


class OneHotEncoderTransforms:
    """OneHotEncoder Transformations

    Parameters:
        column_names (list): columns for one hot encoding
        encoder_dict (dict): dictionary with encoder_params for each column
    """

    def __init__(self, column_names, encoder_dict=None, priority=3):
        self.priority = priority
        self.column_names = column_names
        self.encoder_dict = encoder_dict

    @staticmethod
    def _encode_column(df, column_name):
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(df[[column_name]])
        encoder_params = encoder.categories_
        encoder_params = [x.tolist() for x in encoder_params]
        column_encoded = DataFrame(encoder.transform(df[[column_name]]))
        column_encoded.columns = encoder.get_feature_names_out([column_name])
        for column in column_encoded.columns:
            df[column] = column_encoded[column]
        return encoder_params

    def __call__(self, df):
        self.encoder_dict = {}
        for column in self.column_names:
            encoder_params = self._encode_column(df, column)
            self.encoder_dict[column] = encoder_params
            df.drop([column], axis=1, inplace=True)
        return df
