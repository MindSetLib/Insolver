import os

from numpy import arange
from pandas import DataFrame, Series, concat
from h2o import no_progress, cluster, init, load_model, save_model
from h2o.frame import H2OFrame
from h2o.estimators import H2OEstimator
from h2o.grid.grid_search import H2OGridSearch


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def to_h2oframe(df):
    """Function converts pandas.DataFrame to h2o.H2OFrame ensuring there is no bug duplicating rows in results.

    Args:
        df (pandas.DataFrame):  Dataset to convert to h2o.H2OFrame

    Returns:
        DataFrame converted to h2o.H2OFrame.
    """
    df_h2o = df.copy()
    df_h2o['__insolver_temp_row_id'] = arange(len(df_h2o))
    df_h2o = H2OFrame(df_h2o)
    df_h2o = df_h2o.drop_duplicates(columns=['__insolver_temp_row_id'], keep='first')
    df_h2o = df_h2o.drop('__insolver_temp_row_id', axis=1)
    return df_h2o


class InsolverH2OExtension:
    @staticmethod
    def _h2o_init(h2o_init_params):
        no_progress()
        if cluster() is None:
            init(**(h2o_init_params if h2o_init_params is not None else {}))

    def _h2o_load(self, load_path, h2o_init_params):
        self._h2o_init(h2o_init_params)
        self.model = load_model(load_path)

    def _h2o_save(self, path, name, **kwargs):
        model_path = save_model(model=self.model, path=path, **kwargs)
        os.rename(model_path, os.path.join(os.path.dirname(model_path), f'{name}.h2o'))

    @staticmethod
    def _x_y_to_h2o_frame(X, y, sample_weight, params, X_valid, y_valid, sample_weight_valid):
        if isinstance(X, (DataFrame, Series)) & isinstance(y, (DataFrame, Series)):
            features = X.columns.tolist() if isinstance(X, DataFrame) else X.name
            target = y.columns.tolist() if isinstance(y, DataFrame) else y.name
            if (sample_weight is not None) & isinstance(sample_weight, (DataFrame, Series)):
                params['offset_column'] = (
                    sample_weight.columns.tolist() if isinstance(sample_weight, DataFrame) else sample_weight.name
                )
                X = concat([X, sample_weight], axis=1)
            train_set = to_h2oframe(concat([X, y], axis=1))
        else:
            raise TypeError('X, y are supposed to be pandas DataFrame or Series')

        if (X_valid is not None) & (y_valid is not None):
            if isinstance(X_valid, (DataFrame, Series)) & isinstance(y_valid, (DataFrame, Series)):
                if (
                    (sample_weight_valid is not None)
                    & isinstance(sample_weight_valid, (DataFrame, Series))
                    & (sample_weight is not None)
                ):
                    X_valid = concat([X_valid, sample_weight_valid], axis=1)
                valid_set = to_h2oframe(concat([X_valid, y_valid], axis=1))
                params['validation_frame'] = valid_set
            else:
                raise TypeError('X_valid, y_valid are supposed to be pandas DataFrame or Series')
        return features, target, train_set, params

    def optimize_hyperparam(
        self,
        hyper_params,
        X,
        y,
        sample_weight=None,
        X_valid=None,
        y_valid=None,
        sample_weight_valid=None,
        h2o_train_params=None,
        **kwargs,
    ):
        """Hyperparameter optimization & fitting model in H2O.

        Args:
            hyper_params:
            X (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            sample_weight (pd.DataFrame, pd.Series, optional): Training sample weights.
            X_valid (pd.DataFrame, pd.Series, optional): Validation data (only h2o supported).
            y_valid (pd.DataFrame, pd.Series, optional): Validation target values (only h2o supported).
            sample_weight_valid (pd.DataFrame, pd.Series, optional): Validation sample weights.
            h2o_train_params (dict, optional): Parameters passed to `H2OGridSearch.train()`.
            **kwargs: Other parameters passed to H2OGridSearch.

        Returns:
            dict: {`hyperparameter_name`: `optimal_choice`}, Dictionary containing optimal hyperparameter choice.
        """
        if (self.backend == 'h2o') & isinstance(self.model, H2OEstimator):
            params = dict() if h2o_train_params is None else h2o_train_params
            features, target, train_set, params = self._x_y_to_h2o_frame(
                X, y, sample_weight, params, X_valid, y_valid, sample_weight_valid
            )
            model_grid = H2OGridSearch(model=self.model, hyper_params=hyper_params, **kwargs)
            model_grid.train(y=target, x=features, training_frame=train_set, **params)
            sorted_grid = model_grid.get_grid(sort_by='residual_deviance', decreasing=False)
            self.best_params = sorted_grid.sorted_metric_table().loc[0, :'model_ids'].drop('model_ids').to_dict()
            self.best_params = {
                key: self.best_params[key].replace('[', '').replace(']', '')
                for key in self.best_params.keys()
                if key != ''
            }
            self.best_params = {
                key: float(self.best_params[key]) if is_number(self.best_params[key]) else self.best_params[key]
                for key in self.best_params.keys()
            }
            self.model = sorted_grid.models[0]
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')
        return self.best_params
