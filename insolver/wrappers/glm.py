from functools import partial

from pandas import DataFrame, Series, concat
from numpy import sum, sqrt, repeat

from h2o.frame import H2OFrame
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base import InsolverBaseWrapper
from .extensions import InsolverH2OExtension, InsolverCVHPExtension, InsolverPDPExtension


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class InsolverGLMWrapper(InsolverBaseWrapper, InsolverH2OExtension, InsolverCVHPExtension, InsolverPDPExtension):
    """Insolver wrapper for Generalized Linear Models.

    Attributes:
        backend (str): Framework for building GLM, currently 'h2o' and 'sklearn' are supported.
        family (:obj:`str`, :obj:`float`, :obj:`int`, optional): Distribution for GLM. Supports any family from h2o as
        str. For sklearn supported `str` families are ['gaussian', 'normal', 'poisson', 'gamma', 'inverse_gaussian'],
        also may be defined as `int` or `float` as a power for Tweedie GLM. By default, Gaussian GLM is fitted.
        link (:obj:`str`, optional): Link function for GLM. If `None`, sets to default value for both h2o and sklearn.
        standardize (:obj:`bool`, optional): Whether to standardize data before fitting the model. Enabled by default.
        h2o_init_params (:obj:`dict`, optional): Parameters passed to `h2o.init()`, when `backend` == 'h2o'.
        load_path (:obj:`str`, optional): Path to GLM model to load from disk.
        **kwargs: Parameters for GLM estimators (for H2OGeneralizedLinearEstimator or TweedieRegressor) except
        `family` (`power` for TweedieRegressor) and `link`.
        """
    def __init__(self, backend, family=None, link=None, standardize=True, h2o_init_params=None,
                 load_path=None, **kwargs):
        super(InsolverGLMWrapper, self).__init__(backend)
        self.algo, self._backends = 'glm', ['h2o', 'sklearn']
        self._back_load_dict = {'sklearn': self._pickle_load, 'h2o': partial(self._h2o_load,
                                                                             h2o_init_params=h2o_init_params)}
        self._back_save_dict = {'sklearn': self._pickle_save, 'h2o': self._h2o_save}

        if backend not in self._backends:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

        self.params, self.standardize = None, standardize
        if load_path is not None:
            self.load_model(load_path)
        else:
            family = family if family is not None else 'gaussian'
            link = link if link is not None else 'family_default' if backend == 'h2o' else 'auto'
            if backend == 'h2o':
                self._h2o_init(h2o_init_params)
                self.model = H2OGeneralizedLinearEstimator(family=family, link=link, standardize=self.standardize,
                                                           **kwargs)
            elif backend == 'sklearn':
                if isinstance(family, str):
                    family_power = {'gaussian': 0, 'normal': 0, 'poisson': 1, 'gamma': 2, 'inverse_gaussian': 3}
                    if family in family_power.keys():
                        family = family_power[family]
                    else:
                        raise NotImplementedError('Distribution is not supported with sklearn backend.')
                elif isinstance(family, (float, int)) and (0 < family < 1):
                    raise ValueError('No distribution exists for Tweedie power in range (0, 1).')
                kwargs.update({'power': family, 'link': link})
                self.params = kwargs

                def __params_pipe(**glm_pars):
                    glm_pars.update(self.params)
                    return Pipeline([('scaler', StandardScaler(with_mean=self.standardize, with_std=self.standardize)),
                                     ('glm', TweedieRegressor(**glm_pars))])

                self.model, self.object = __params_pipe(**self.params), __params_pipe

    def fit(self, X, y, sample_weight=None, X_valid=None, y_valid=None, sample_weight_valid=None, **kwargs):
        """Fit a Generalized Linear Model.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            sample_weight (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Training sample weights.
            X_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation data (only h2o supported).
            y_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation target values (only h2o supported).
            sample_weight_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation sample weights.
            **kwargs: Other parameters passed to H2OGeneralizedLinearEstimator.
        """
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            if isinstance(X, (DataFrame, Series)):
                self.model.feature_name_ = X.columns.tolist() if isinstance(X, DataFrame) else [X.name]
            self.model.fit(X, y, glm__sample_weight=sample_weight)
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            features, target, train_set, params = self._x_y_to_h2o_frame(X, y, sample_weight, {**kwargs}, X_valid,
                                                                         y_valid, sample_weight_valid)
            self.model.train(y=target, x=features, training_frame=train_set, **params)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def predict(self, X, sample_weight=None, **kwargs):
        """Predict using GLM with feature matrix X.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Samples.
            sample_weight (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Test sample weights.
            **kwargs: Other parameters passed to H2OGeneralizedLinearEstimator.predict().

        Returns:
            array: Returns predicted values.
        """
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            predictions = self.model.predict(X if not hasattr(self.model, 'feature_name_')
                                             else X[self.model.feature_name_])
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            if self.model.parms['offset_column']['actual_value'] is not None and sample_weight is None:
                offset_name = self.model.parms['offset_column']['actual_value']['column_name']
                sample_weight = Series(repeat(0, len(X)), name=offset_name, index=X.index)
            if sample_weight is not None:
                X = concat([X, sample_weight], axis=1)
            h2o_predict = X if isinstance(X, H2OFrame) else H2OFrame(X)
            predictions = self.model.predict(h2o_predict, **kwargs).as_data_frame().values.reshape(-1)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')
        return predictions

    def coef_norm(self):
        """Output GLM coefficients for standardized data.

        Returns:
            dict: {:obj:`str`: :obj:`float`}m Dictionary containing GLM coefficients for standardized data.
        """
        if self.standardize:
            if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
                if self.model.feature_name_ is None:
                    self.model.feature_name_ = [f'Variable_{i}' for i
                                                in range(len(list(self.model.named_steps['glm'].coef_)))]
                coefs = zip(['Intercept'] + self.model.feature_name_,
                            [self.model.named_steps['glm'].intercept_] + list(self.model.named_steps['glm'].coef_))
                coefs = {x[0]: x[1] for x in coefs}
            elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
                coefs = self.model.coef_norm()
            else:
                raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

        else:
            Exception('Normalized coefficients unavailable since model fitted on non-standardized data.')
            coefs = dict()
        return coefs

    def coef(self):
        """Output GLM coefficients for non-standardized data. Also calculated when GLM fitted on standardized data.

        Returns:
            dict: {:obj:`str`: :obj:`float`}m Dictionary containing GLM coefficients for non-standardized data.
        """
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            if self.model.feature_name_ is None:
                self.model.feature_name_ = [f'Variable_{i}' for i
                                            in range(len(list(self.model.named_steps['glm'].coef_)))]
            if self.standardize:
                intercept = self.model.named_steps['glm'].intercept_ - sum(self.model.named_steps['glm'].coef_ *
                                                                           self.model.named_steps['scaler'].mean_ /
                                                                           sqrt(self.model.named_steps['scaler'].var_))
                coefs = self.model.named_steps['glm'].coef_ / sqrt(self.model.named_steps['scaler'].var_)
                coefs = zip(['Intercept'] + self.model.feature_name_, [intercept] + list(coefs))
            else:
                coefs = zip(['Intercept'] + self.model.feature_name_,
                            [self.model.named_steps['glm'].intercept_] + list(self.model.named_steps['glm'].coef_))
            coefs = {x[0]: x[1] for x in coefs}
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            coefs = self.model.coef()
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')
        return coefs

    def optimize_hyperparam(self, hyper_params, X, y, sample_weight=None, X_valid=None, y_valid=None,
                            sample_weight_valid=None, h2o_train_params=None, **kwargs):
        """Hyperparameter optimization & fitting GLM.

        Args:
            hyper_params:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            sample_weight (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Training sample weights.
            X_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation data (only h2o supported).
            y_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation target values (only h2o supported).
            sample_weight_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation sample weights.
            h2o_train_params (:obj:`dict`, optional): Parameters passed to `H2OGridSearch.train()`.
            **kwargs: Other parameters passed to H2OGridSearch.

        Returns:
            dict: {`hyperparameter_name`: `optimal_choice`}, Dictionary containing optimal hyperparameter choice.
        """
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            Exception('optimize_hyperparam is available only for `h2o` backend. Use hyperopt_cv otherwise.')
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            params = dict() if h2o_train_params is None else h2o_train_params
            features, target, train_set, params = self._x_y_to_h2o_frame(X, y, sample_weight, params, X_valid, y_valid,
                                                                         sample_weight_valid)
            model_grid = H2OGridSearch(model=self.model, hyper_params=hyper_params, **kwargs)
            model_grid.train(y=target, x=features, training_frame=train_set, **params)
            sorted_grid = model_grid.get_grid(sort_by='residual_deviance', decreasing=False)
            self.best_params = sorted_grid.sorted_metric_table().loc[0, :'model_ids'].drop('model_ids').to_dict()
            self.best_params = {key: self.best_params[key].replace('[', '').replace(']', '')
                                for key in self.best_params.keys() if key != ''}
            self.best_params = {key: float(self.best_params[key]) if is_number(self.best_params[key])
                                else self.best_params[key] for key in self.best_params.keys()}
            self.model = sorted_grid.models[0]
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')
        return self.best_params
