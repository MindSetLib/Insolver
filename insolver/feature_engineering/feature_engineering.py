import warnings
import pandas as pd
from insolver.transforms import EncoderTransforms, OneHotEncoderTransforms, AutoFillNATransforms
from insolver.feature_engineering import DimensionalityReduction, Sampling, Smoothing, Normalization, FeatureSelection


class DataPreprocessing:
    """
    Feature Engineering.
    This class allows you to automatically preprocess data.
    By default, it applies AutoFillNA, OneHotEncoder and normalization transformations to data.
    Any feature engineering method used in this class can be disabled using assigned parameters.
    You can also use dimensionality reduction, sampling, smoothing, feature selection and change their parameters
    available in this class using the assigned parameters.

    Parameters:
        numerical_columns (list): List of numerical columns.
        categorical_columns (list): List of categorical columns.
        transform_categorical (str, bool, None), default=True: The name of the categorical transform method,
         values `one_hot_encoder`, `encoder` are supported. If True `one_hot_encoder` will be used. If False/None
         categorical won't be transformed.
        transform_categorical_drop (list): List of categorical columns to not transform.
        fillna (bool, None), default=True: Auto fill NA bool: if True Auto fill NA will be applied, if False/None it
         won't be applied.
        fillna_numerical (str), default='median':  Auto fill NA numerical method name, values `median`, `mean`, `mode`,
         `remove` are supported.
        fillna_categorical (str), default='frequent': Auto fill NA categorical method name, values `frequent`,
         `new_category`, `imputed_column`, `remove` are supported.
        normalization (str, bool, None), default=True: Normalization method name, values `standard`, `minmax`,
         `robust`, `normalizer`, `yeo-johnson`, `box-cox`, `log` are supported. If True 'standard' will be used. If
         False/None normalization won't be applied.
        normalization_drop (list): List of columns to not normalize.
        feature_selection (str, bool, None), default=None: Feature selection method name. If True `random_forest` will
         be used. If False/None feature selection won't be applied.
        feat_select_task (str), default=None: Feature selection task, values `reg`, `class`, `multiclass`,
         `multiclass_multioutput` are supported.
        feat_select_threshold (str, int), default='mean': Feature selection threshold, values `mean`, `median` are
         supported or the threshold can be numeric.
        dim_red (str, bool, None), default=None: Dimensionality reduction method name, values `pca`, `svd`, `lda`,
         `t_sne`, `isomap`, `lle`, `fa`, `nmf` are supported. If True `pca` will be used. If False/None dimensionality
         reduction won't be applied.
        dim_red_n_components (int, None), default=None: Dimensionality reduction n_components parameter value. If None
         n_components will be calculated by the model or will be set to the default value = 2.
        dim_red_n_neighbors (int, None), default=None: Dimensionality reduction n_neighbors (or perplexity in the
         `t_sne`).
         parameter value. If None it will be set to the default value = 5 (for the `t_sne` = 30).
        sampling (str, bool, None), default=None: Sampling method name, values `simple`, `systematic`, `cluster`,
         `stratified` are supported. If True `simple` will be used. If False/None sampling won't be applied.
        sampling_n (int, None), default=None: Sampling n value. If None it will be set to the default value depending on
         the method.
        sampling_n_clusters (int), default=10: Sampling n_clusters value.
        smoothing (str, bool, None), default=None: Smoothing method name, values `moving_average`, `lowess`,
         `s_g_filter`, `fft` are supported. If True `moving_average` will be used. If False/None smoothing won't be
         applied.
        smoothing_column (str): Name of the column to smooth.

    """

    def __init__(
        self,
        numerical_columns=None,
        categorical_columns=None,
        transform_categorical=True,
        transform_categorical_drop=None,
        fillna=True,
        fillna_numerical='median',
        fillna_categorical='frequent',
        normalization=True,
        normalization_drop=None,
        feature_selection=None,
        feat_select_task=None,
        feat_select_threshold='mean',
        dim_red=None,
        dim_red_n_components=None,
        dim_red_n_neighbors=None,
        sampling=None,
        sampling_n=None,
        sampling_n_clusters=10,
        smoothing=None,
        smoothing_column=None,
    ):
        # columns initialization and transformation attributes
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.transform_categorical = transform_categorical
        self.transform_categorical_drop = [] if transform_categorical_drop is None else transform_categorical_drop

        # auto fill NA values attributes
        self.fillna = fillna
        self.fillna_numerical = fillna_numerical
        self.fillna_categorical = fillna_categorical

        # normalization attributes
        self.normalization = normalization
        self.normalization_drop = [] if normalization_drop is None else normalization_drop

        # feature selection attributes
        self.feature_selection = feature_selection
        self.feat_select_task = feat_select_task
        self.feat_select_threshold = feat_select_threshold

        # dimensionality reduction attributes
        self.dim_red = dim_red
        self.dim_red_n_components = dim_red_n_components
        self.dim_red_n_neighbors = dim_red_n_neighbors

        # sampling attributes
        self.sampling = sampling
        self.sampling_n = sampling_n
        self.sampling_n_clusters = sampling_n_clusters

        # smoothing attributes
        self.smoothing = smoothing
        self.smoothing_column = smoothing_column

    def preprocess(self, df, target=None, drop_target=True):
        """
        When you call the `preprocess` method, it finds numerical and categorical columns,
        transformes categorical columns using the `one_hot_encoder` method, fills in the NA values using the'median'
        method.
        for numerical columns and the 'frequent' method for categorical columns, normalizes data using the `standard`
        method.

        Parameters:
            df (pandas.Dataframe): The dataframe.
            target (str): Target column name.
            drop_target (bool), default=True: If True, target column won't be transformed.

        Raises:
            NotImplementedError: If target parameter must be str or list.
        """
        # create a copy of the DataFrame if inplace=False
        df = df.copy()

        if drop_target:
            # if target is str add to lists
            if isinstance(target, str):
                self.normalization_drop.append(target)
                self.transform_categorical_drop.append(target)

            # if target is list iterate and add to lists
            elif isinstance(target, list):
                for t in target:
                    self.normalization_drop.append(t)
                    self.transform_categorical_drop.append(t)

            # else if not str and not list raise error
            else:
                raise NotImplementedError('Target parameter must be str or list.')

        self._init_features(df, target)
        if self.fillna:
            self._fillna(df)
        if self.transform_categorical:
            df = self._transform_categorical(df)
        if self.smoothing:
            df = self._smoothing(df, target)
        if self.normalization:
            df = self._normalization(df)
        if self.feature_selection:
            df = self._feature_selection(df, target)
        if self.dim_red:
            df = self._dim_reduction(df, target)
        if self.sampling:
            df = self._sampling(df)

        return df

    def _init_features(self, df, target=None):
        """
        Numerical and categorical columnsn itialization.

        Parameters:
            df (pandas.Dataframe): The dataframe.
            target (str): Target column name.
        """
        # check if numerical_columns are in the DataFrame
        if self.numerical_columns and set(self.numerical_columns).difference(df.columns):
            xor_cols = set(self.numerical_columns).difference(df.columns)
            raise AttributeError(f'Columns {xor_cols} are not in the DataFrame.')

        # check if categorical_columns are in the DataFrame
        if self.categorical_columns and set(self.categorical_columns).difference(df.columns):
            xor_cols = set(self.categorical_columns).difference(df.columns)
            raise AttributeError(f'Columns {xor_cols} are not in the DataFrame.')

        # if numerical_columns and categorical_columns are not initialized, create them
        if not self.numerical_columns and not self.categorical_columns:
            self.categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']
            self.numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']

        # if not categorical_columns, create categorical_columns
        elif not self.categorical_columns:
            self.categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']

        # if not numerical_columns, create numerical_columns
        elif not self.numerical_columns:
            self.numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']

        # warn about columns that are not included (check with target)
        elif target and set(df.columns) - set(self.categorical_columns + self.numerical_columns) - set(target):
            left_cols = set(df.columns) - set(self.categorical_columns + self.numerical_columns) - set(target)
            warnings.warn(f'Columns {left_cols} are not included.')

        # warn about columns that are not included (check without target)
        elif not target and set(df.columns) - set(self.categorical_columns + self.numerical_columns):
            left_cols = set(df.columns) - set(self.categorical_columns + self.numerical_columns)
            warnings.warn(f'Columns {left_cols} are not included.')

    def _transform_categorical(self, df):
        """
        Categorical Encoding.

        Parameters:
            df (pandas.Dataframe): The dataframe.
        """
        # create categorical transformations dict
        cat_dict = {
            'one_hot_encoder': OneHotEncoderTransforms,
            'encoder': EncoderTransforms,
        }

        # set default value if parameter is initialized as True
        if self.transform_categorical is True:
            self.transform_categorical = 'one_hot_encoder'

        # if transform_categorical_drop, create a copy of data without these columns and copy them to concat later
        if self.transform_categorical_drop:
            df_to_transform = df.drop(self.transform_categorical_drop, axis=1)
            columns_to_concat = df[self.transform_categorical_drop]

            # transform categorical values with the selected method
            transform_method = cat_dict[self.transform_categorical]
            columns_to_transform = set(self.categorical_columns) - set(self.transform_categorical_drop)
            transform_method(column_names=list(columns_to_transform)).__call__(df=df_to_transform)

            return pd.concat([df_to_transform, columns_to_concat], axis=1)

        # else transform all categorical values with the selected method
        else:
            transform_method = cat_dict[self.transform_categorical]
            return transform_method(column_names=self.categorical_columns).__call__(df=df)

    def _fillna(self, df):
        """
        AutoFill NA.

        Parameters:
            df (pandas.Dataframe): The dataframe.
        """
        # fill NA values with the AutoFillNATransforms class
        AutoFillNATransforms(
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            numerical_method=self.fillna_numerical,
            categorical_method=self.fillna_categorical,
        ).__call__(df=df)

    def _normalization(self, df):
        """
        Normalization.

        Parameters:
            df (pandas.Dataframe): The dataframe.
        """
        # set default value if parameter is initialized as True
        if self.normalization is True:
            self.normalization = 'standard'

        # if normalization_drop, create a copy of data without these columns and copy them to concat later
        if self.normalization_drop:
            df_to_norm = df.drop(self.normalization_drop, axis=1)
            columns_to_concat = df[self.normalization_drop]
            df_to_norm = Normalization(method=self.normalization).transform(df_to_norm)
            return pd.concat([df_to_norm, columns_to_concat], axis=1)

        # else normalize and return transformed all values
        else:
            return Normalization(method=self.normalization).transform(df)

    def _dim_reduction(self, df, target):
        """
        Dimensionality Reduction.

        Parameters:
            df (pandas.Dataframe): The dataframe.
            target (str): Target column name.

        Raises:
            NotImplementedError: If Y is Multi target and dim_red == 'lda'.
        """
        # set default value if parameter is initialized as True
        if self.dim_red is True:
            self.dim_red = 'pca'

        if isinstance(target, str):
            # set X as the given DataFrame if the target is None else set X as the given DataFrame without target
            X = df if target is None else df.drop([target], axis=1)
            # set y as the target if it's initialized else set y None
            y = df[target] if target else None
        else:
            X = df if target is None else df.drop(target, axis=1)
            y = df[target] if target else None

        # if the selected method is decomposition, then set X and n_components in the transform method
        if self.dim_red in ['pca', 'svd', 'fa', 'nmf']:
            X = DimensionalityReduction(method=self.dim_red).transform(X=X, n_components=self.dim_red_n_components)

        # if the selected method is LDA, then set X, y and n_components in the transform method
        elif self.dim_red == 'lda':
            # raise error if y is list
            if isinstance(target, list):
                raise NotImplementedError('Multi target is not supported for LDA.')

            X = DimensionalityReduction(method=self.dim_red).transform(X=X, y=y, n_components=self.dim_red_n_components)

        # if the selected method is TSNE, then set X, n_components and perplexity in the transform method
        elif self.dim_red == 't_sne':
            # set n_components and perplexity to default values if not initialized
            n_components = self.dim_red_n_components if self.dim_red_n_components else 2
            perplexity = self.dim_red_n_neighbors if self.dim_red_n_neighbors else 30
            X = DimensionalityReduction(method=self.dim_red).transform(
                X=X, n_components=n_components, perplexity=perplexity
            )

        # if the selected method is manifolds, then set X, n_components and n_neighbors in the transform method
        else:
            # set n_components and n_neighbors to default values if not initialized
            n_components = self.dim_red_n_components if self.dim_red_n_components else 2
            n_neighbors = self.dim_red_n_neighbors if self.dim_red_n_neighbors else 5
            X = DimensionalityReduction(method=self.dim_red).transform(
                X=X, n_components=n_components, n_neighbors=n_neighbors
            )

        return pd.concat([X, y], axis=1)

    def _sampling(self, df):
        """
        Sampling.

        Parameters:
            df (pandas.Dataframe): The dataframe.
        """
        # create dict of default sampling_n values
        n_dict = {
            'simple': int(len(df) / 2),
            'systematic': 2,
            'cluster': int(self.sampling_n_clusters / 2),
            'stratified': int(len(df) / self.sampling_n_clusters / 2),
        }

        # set default value if parameter is initialized as True
        if self.sampling is True:
            self.sampling = 'simple'

        # set sampling_n values if sampling_n is None
        n = self.sampling_n if self.sampling_n else n_dict[self.sampling]
        return Sampling(method=self.sampling, n=n, n_clusters=self.sampling_n_clusters).sample_dataset(df)

    def _smoothing(self, df, target=None):
        """
        Smoothing.

        Parameters:
            df (pandas.Dataframe): The dataframe.
            target (str): Target column name.

        Raises:
            AttributeError: If parameter `smoothing_column` is not initialized.
        """
        # set default value if parameter is initialized as True
        if self.smoothing is True:
            self.smoothing = 'moving_average'

        # check if parameter `smoothing_column` is not initialized
        if not self.smoothing_column:
            raise AttributeError('Parameter `smoothing_column` must be initialized for the smoothing.')

        # smooth and return transformed values
        return Smoothing(method=self.smoothing, x_column=self.smoothing_column, y_column=target).transform(df)

    def _feature_selection(self, df, target=None):
        """
        Feature Selection.

        Parameters:
            df (pandas.Dataframe): The dataframe.
            target (str): Target column name.

        Raises:
            NotImplementedError: If Y is Multi target.
            AttributeError: If parameter `feat_select_task` is not initialized.
        """
        # raise error if target is list
        if isinstance(target, list):
            raise NotImplementedError('Multi target is not supported in feature selection.')

        # set default value if parameter is initialized as True
        if self.feature_selection is True:
            self.feature_selection = 'random_forest'

        # check if parameter `task` is not initialized
        if not self.feat_select_task:
            raise AttributeError('Parameter `feat_select_task` must be initialized for feature selection.')

        _fs = FeatureSelection(y_column=target, task=self.feat_select_task, method=self.feature_selection)
        _fs.create_model(df)
        return _fs.create_new_dataset(self.feat_select_threshold)
