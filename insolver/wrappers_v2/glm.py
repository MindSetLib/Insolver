import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from os import PathLike
from typing import Optional, Dict, Any, Union, List, Tuple, Callable

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor, LogisticRegression, ElasticNet

from h2o.frame import H2OFrame
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from numpy import repeat, ndarray, insert, sum as npsum, sqrt, exp, true_divide, hstack, ones
from pandas import DataFrame, Series, concat

from ..utils import warn_insolver
from .base import InsolverBaseWrapper, InsolverWrapperWarning
from .utils import save_pickle, save_dill, save_h2o
from .utils.h2o_utils import x_y_to_h2o_frame, h2o_start, h2o_stop, to_h2oframe, load_h2o


class InsolverGLMWrapper(InsolverBaseWrapper):
    algo = 'glm'
    _backends = ["h2o", "sklearn"]
    _tasks = ["class", "reg"]
    _backend_saving_methods = {'sklearn': {'pickle': save_pickle, 'dill': save_dill}, 'h2o': {'h2o': save_h2o}}

    """Insolver wrapper for Generalized Linear Models.

    Parameters:
        backend (str): Framework for building GLM, currently 'h2o' and 'sklearn' are supported.
        task (str): Task that GLM should solve: Classification or Regression. Values 'reg' and 'class' are supported.
        family (str, float, int, optional): Distribution for GLM. Supports any family from h2o as
          str. For sklearn supported `str` families are ['gaussian', 'normal', 'poisson', 'gamma', 'inverse_gaussian'],
          also may be defined as `int` or `float` as a power for Tweedie GLM. By default, Gaussian GLM is fitted.
        link (str, optional): Link function for GLM. If `None`, sets to default value for both h2o and sklearn.
        h2o_init_params (dict, optional): Parameters passed to `h2o.init()`, when `backend` == 'h2o'.
        **kwargs: Parameters for GLM estimators (for H2OGeneralizedLinearEstimator or TweedieRegressor) except
          `family` (`power` for TweedieRegressor) and `link`.

        """

    def __init__(
        self,
        backend: Optional[Literal['sklearn', 'h2o']],
        task: Optional[Literal['class', 'reg']] = 'reg',
        family: Optional[str] = None,
        link: Optional[str] = None,
        h2o_server_params: Optional[Dict] = None,
        **kwargs: Any,
    ):
        self._get_init_args(vars())

        # Checks on supported backends and tasks
        if backend not in self._backends:
            raise ValueError(f'Invalid "{backend}" backend argument. Supported backends: {self._backends}.')
        if task not in self._tasks:
            raise ValueError(f'Invalid "{task}" task argument. Supported tasks: {self._tasks}.')

        self.backend = backend
        self.task = task
        self.family = family
        self.link = link
        self.h2o_server_params = h2o_server_params
        self.kwargs = kwargs
        self.model = self.init_model()
        self.__dict__.update(self.metadata)

    def _init_glm_sklearn(self, **params: Any) -> BaseEstimator:
        model = BaseEstimator()  # Just to mitigate referenced before assignment warning

        # Checks on supported families vs tasks
        if self.family not in [None, 'poisson', 'gamma', 'tweedie', 'normal', 'gaussian', 'inverse_gaussian', 'logit']:
            ValueError(f'Distribution family "{self.family}" is not supported with sklearn backend.')
        else:
            if (self.family in ['logit']) and (self.task == 'reg'):
                ValueError(f'Distribution family "{self.family}" does not match the task "{self.task}".')
            if (self.family not in [None, 'logit']) and (self.task == 'class'):
                ValueError(f'Distribution family "{self.family}" does not match the task "{self.task}".')
            if self.family is None:
                self.family = 'gaussian' if self.task == 'reg' else 'logit'

        # Checks on supported families vs links
        if self.family in ['gamma', 'poisson']:
            self.link = 'log' if self.link is None else self.link
            if self.link != 'log':
                warn_insolver(
                    f'Link function "{self.link}" not supported for "{self.family}",using default "log" link',
                    InsolverWrapperWarning,
                )
        if self.family in ['tweedie', 'inverse_gaussian']:
            self.link = 'log' if self.link is None else self.link
            if self.link not in ['log', 'identity']:
                warn_insolver(
                    f'Link function "{self.link}" not supported for "{self.family}",using default "log" link',
                    InsolverWrapperWarning,
                )
        if self.family in ['normal', 'gaussian']:
            self.link = 'identity' if self.link is None else self.link
            if self.link != 'identity':
                warn_insolver(
                    f'Link function "{self.link}" not supported for "{self.family}",using default "identity" link',
                    InsolverWrapperWarning,
                )
        if self.family in ['normal', 'gaussian']:
            self.link = 'identity' if self.link is None else self.link
            if self.link != 'identity':
                warn_insolver(
                    f'Link function "{self.link}" not supported for "{self.family}",using default "identity" link',
                    InsolverWrapperWarning,
                )
        if self.family == 'logit':
            self.link = 'logit' if self.link is None else self.link
            if self.link != 'logit':
                warn_insolver(
                    f'Link function "{self.link}" not supported for "{self.family}",using default "logit" link',
                    InsolverWrapperWarning,
                )

        # Estimator initialization
        if self.family == 'poisson':
            # alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0
            model = PoissonRegressor(**params)
        if self.family == 'gamma':
            # alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0
            model = GammaRegressor(**params)
        if self.family == 'tweedie':
            # power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001,
            # warm_start=False, verbose=0
            model = TweedieRegressor(**params)
        if self.family == 'inverse_gaussian':
            # alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0
            model = TweedieRegressor(power=3, **params)
        if self.family in ['normal', 'gaussian']:
            # alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize='deprecated', precompute=False,
            # max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None,
            # selection='cyclic'
            model = ElasticNet(**params)
        if self.family == 'logit':
            # penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
            # random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False,
            # n_jobs=None, l1_ratio=None
            model = LogisticRegression(**params)

        if self.family in ['poisson', 'gamma', 'tweedie', 'inverse_gaussian']:
            model = Pipeline([('scaler', StandardScaler(with_mean=True, with_std=True)), ('glm', model)])
            self.metadata.update({'is_standardized': True})
        else:
            self.metadata.update({'is_standardized': False})

        return model

    def _init_glm_h2o(self, **params: Any) -> H2OGeneralizedLinearEstimator:
        model = H2OGeneralizedLinearEstimator(family=self.family, link=self.link, **params)
        return model

    def init_model(self) -> Any:
        model = None
        if self.backend == 'sklearn':
            params = self.metadata['init_params']['kwargs']
            # params.update(params.pop('kwargs'))
            model = self._init_glm_sklearn(**params)
        if self.backend == 'h2o':
            params = {
                key: self.metadata['init_params'][key]
                for key in self.metadata['init_params']
                if key not in ['family', 'link', 'backend', 'task', 'h2o_server_params', 'kwargs']
            }
            params.update(self.metadata['init_params']['kwargs'])
            model = self._init_glm_h2o(**params)
        self._update_metadata()
        return model

    def fit(
        self,
        x: Union[DataFrame, Series],
        y: Union[DataFrame, Series],
        sample_weight: Union[None, DataFrame, Series] = None,
        x_valid: Union[None, DataFrame, Series] = None,
        y_valid: Union[None, DataFrame, Series] = None,
        sample_weight_valid: Union[None, DataFrame, Series] = None,
        report: Union[None, List, Tuple, Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Fit a Generalized Linear Model.

        Args:
            x (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            sample_weight (pd.DataFrame, pd.Series, optional): Training sample weights.
            x_valid (pd.DataFrame, pd.Series, optional): Validation data (only h2o supported).
            y_valid (pd.DataFrame, pd.Series, optional): Validation target values (only h2o supported).
            sample_weight_valid (pd.DataFrame, pd.Series, optional): Validation sample weights.
            report (list, tuple, optional): A list of metrics to report after model fitting, optional.
            **kwargs: Other parameters passed to H2OGeneralizedLinearEstimator.
        """
        for arg in [x, y, sample_weight, x_valid, y_valid, sample_weight_valid]:
            if (arg is not None) and (not isinstance(arg, (DataFrame, Series))):
                argname = [k for k, v in locals().items() if v == arg][0]
                raise TypeError(
                    f'Invalid type {type(arg)} for "{argname}". It must be either pd.DataFrame or pd.Series.'
                )

        for y_var in [y, y_valid]:
            if isinstance(y_var, DataFrame) and y_var.shape[1] > 1:
                argname = [k for k, v in locals().items() if v == y_var][0]
                raise ValueError(f'Argument "{argname}" must be a one-dimensional DataFrame.')

        features = list(x.columns) if isinstance(x, DataFrame) else [x.name]
        target = list(y.columns) if isinstance(y, DataFrame) else y.name
        self.metadata.update({'feature_names': features, 'target': target})
        prediction = None

        if self.backend == 'sklearn':
            if any(arg is not None for arg in [x_valid, y_valid, sample_weight_valid]):
                warn_insolver(
                    'Arguments x_valid, y_valid, sample_weight_valid are not supported by sklearn backend',
                    InsolverWrapperWarning,
                )
            if self.metadata['is_standardized']:
                self.model.fit(x, y, glm__sample_weight=sample_weight)
            else:
                self.model.fit(x, y, sample_weight=sample_weight)
            self.metadata.update({'is_fitted': True})
            self.metadata.update({'coefs': self.coef()})
            if isinstance(report, (list, tuple)) or callable(report):
                prediction = self.model.predict(x)
        if self.backend == 'h2o':
            h2o_start()
            train_set, params = x_y_to_h2o_frame(x, y, sample_weight, {**kwargs}, x_valid, y_valid, sample_weight_valid)
            self.model.train(y=target, x=features, training_frame=train_set, **params)
            self.metadata.update({'is_fitted': True})
            self.metadata.update({'coefs': self.coef()})
            if isinstance(report, (list, tuple)) or callable(report):
                prediction = self.model.predict(train_set).as_data_frame().values.reshape(-1)
            self._model_cached = self.save_model()
            h2o_stop()

        if prediction is not None:
            if not callable(report) and (report is not None):
                print(
                    DataFrame([[x.__name__, x(y, prediction)] for x in report], columns=['Metrics', 'Value']).set_index(
                        'Metrics'
                    )
                )
            if callable(report) and (report is not None):
                print(
                    DataFrame([[report.__name__, report(y, prediction)]], columns=['Metrics', 'Value']).set_index(
                        'Metrics'
                    )
                )

    def predict(
        self, x: Union[DataFrame, Series], sample_weight: Union[None, DataFrame, Series] = None, **kwargs: Any
    ) -> Optional[ndarray]:
        """Predict using GLM with feature matrix X.

        Args:
            x (pd.DataFrame, pd.Series): Samples.
            sample_weight (pd.DataFrame, pd.Series, optional): Test sample weights.
            **kwargs: Other parameters passed to H2OGeneralizedLinearEstimator.predict().

        Returns:
            array: Returns predicted values.
        """
        if not self.metadata['is_fitted']:
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        if not isinstance(x, (DataFrame, Series)):
            raise TypeError(f'Invalid type {type(x)} for "x". It must be either pd.DataFrame or pd.Series.')

        predictions = None
        if self.backend == 'sklearn':
            predictions = self.model.predict(x[self.metadata['feature_names']] if isinstance(x, DataFrame) else x)

        if self.backend == 'h2o':
            if self._model_cached is not None:
                load_h2o(self._model_cached, self.h2o_server_params, terminate=False)
            if self.model.parms['offset_column']['actual_value'] is not None and sample_weight is None:
                offset_name = self.model.parms['offset_column']['actual_value']['column_name']
                sample_weight = Series(repeat(1, len(x)), name=offset_name, index=x.index)
            if sample_weight is not None:
                x = concat([x, sample_weight], axis=1)
            h2o_predict = x if isinstance(x, H2OFrame) else to_h2oframe(x)
            predictions = self.model.predict(h2o_predict, **kwargs).as_data_frame().values.reshape(-1)
            h2o_stop()
        return predictions

    def predict_coef(self, x: Union[DataFrame, Series]) -> Optional[ndarray]:
        """Predict using only GLM coefficients (without model itself) with feature matrix X.

        Args:
            x (pd.DataFrame, pd.Series): Samples.

        Returns:
            array: Returns predicted values.
        """
        if (not self.metadata['is_fitted']) or ('coefs' not in self.metadata.keys()):
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        if not isinstance(x, (DataFrame, Series)):
            raise TypeError(f'Invalid type {type(x)} for "x". It must be either pd.DataFrame or pd.Series.')

        def link_identity(lin_pred: ndarray) -> ndarray:
            return lin_pred

        def link_log(lin_pred: ndarray) -> ndarray:
            return exp(lin_pred)

        def link_inverse(lin_pred: ndarray) -> ndarray:
            return true_divide(1, lin_pred)

        def link_logit(lin_pred: ndarray) -> ndarray:
            return true_divide(exp(-lin_pred), 1 + exp(-lin_pred))

        # def link_ologit(lin_pred):
        #     pass
        #
        # def link_tweedie(lin_pred):
        #     pass

        link_map = {'identity': link_identity, 'log': link_log, 'inverse': link_inverse, 'logit': link_logit}

        coefs = self.metadata['coefs']

        if isinstance(x, DataFrame):
            difference = set(coefs).difference(set(x.columns))
        elif isinstance(x, Series):
            difference = set(coefs).difference({x.name})
        else:
            difference = {'Intercept'}
        difference.discard('Intercept')
        if difference != set():
            raise KeyError(f'Input data missing columns: {difference}')

        coefs = Series(coefs)
        x_ = x[coefs.index.drop('Intercept')] if isinstance(x, DataFrame) else x
        x_ = hstack((ones((x_.shape[0], 1)), x_.values))
        linear_prediction = x_.dot(coefs.values)
        if self.metadata['link'] in ['ologit', 'tweedie']:
            raise NotImplementedError(f"Link function `{self.metadata['link']}` is not implemented.")
        else:
            return link_map[self.metadata['link']](linear_prediction).reshape(-1)

    def coef_norm(self) -> Optional[Dict[str, float]]:
        """Output GLM coefficients for standardized data.

        Returns:
            dict: {`str`: `float`} Dictionary containing GLM coefficients for standardized data.
        """
        if not self.metadata['is_fitted']:
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        coefs = None
        if self.backend == 'sklearn':
            if self.metadata['is_standardized']:
                if self.metadata['feature_names'] is None:
                    features_ = [f'Feature_{i}' for i in range(len(self.model.named_steps['glm'].coef_))]
                    self.metadata['feature_names'] = features_
                else:
                    features_ = self.metadata['feature_names']

                _zip = zip(
                    ['Intercept'] + features_,
                    insert(self.model.named_steps['glm'].coef_, 0, self.model.named_steps['glm'].intercept_),
                )
                coefs = {x: y for x, y in _zip}
            else:
                raise NotImplementedError(f'Current method does not support {self.family} family.')
        if self.backend == 'h2o':
            coefs = self.model.coef_norm()
        return coefs

    def coef(self) -> Optional[Dict[str, float]]:
        """Output GLM coefficients for non-standardized data. Also calculated when GLM fitted on standardized data.

        Returns:
            dict: {`str`: `float`} Dictionary containing GLM coefficients for non-standardized data.
        """
        if not self.metadata['is_fitted']:
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        coefs = None
        if self.backend == 'sklearn':
            if self.metadata['feature_names'] is None:
                if self.metadata['is_standardized']:
                    features_ = [f'Feature_{i}' for i in range(len(self.model.named_steps['glm'].coef_))]
                else:
                    features_ = [f'Feature_{i}' for i in range(len(self.model.coef_))]
                self.metadata['feature_names'] = features_
            else:
                features_ = self.metadata['feature_names']

            if self.metadata['is_standardized']:
                _int = self.model.named_steps['glm'].intercept_
                _coef = self.model.named_steps['glm'].coef_
                _mean = self.model.named_steps['scaler'].mean_
                _var = self.model.named_steps['scaler'].var_
                intercept = _int - npsum(_coef * _mean / sqrt(_var))
                coefs_ = _coef / sqrt(_var)
            else:
                intercept = self.model.intercept_
                coefs_ = self.model.coef_

            _zip = zip(['Intercept'] + features_, insert(coefs_, 0, intercept))
            coefs = {x: y for x, y in _zip}
        if self.backend == 'h2o':
            coefs = self.model.coef()
        return coefs

    def coef_to_csv(self, path_or_buf: Union[None, str, 'PathLike[str]'] = None, **kwargs: Any) -> None:
        """Write GLM coefficients to a comma-separated values (csv) file.

        Args:
            path_or_buf : str or file handle, default None
                File path or object, if None is provided the result is returned as
                a string.  If a non-binary file object is passed, it should be opened
                with `newline=''`, disabling universal newlines. If a binary
                file object is passed, `mode` might need to contain a `'b'`.
            **kwargs: Other parameters passed to Pandas DataFrame.to_csv method.
        Returns:
            None or str
                If path_or_buf is None, returns the resulting csv format as a
                string. Otherwise, returns None.
        """
        result = DataFrame()
        sources_methods = {
            'coefficients for standardized data': self.coef_norm,
            'coefficients for non-standardized data': self.coef,
        }

        for name, method in sources_methods.items():
            try:
                column = method()

                if isinstance(column, dict):
                    result = result.join(Series(column, name=name), how='outer')
            except NotImplementedError:
                pass

        if result.size > 0:
            if path_or_buf is None:
                return result.to_csv(path_or_buf, **kwargs)
            else:
                result.to_csv(path_or_buf, **kwargs)
        else:
            warn_insolver('No coefficients available!', InsolverWrapperWarning)
