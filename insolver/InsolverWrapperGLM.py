import os
import time
import pickle

from h2o import init, no_progress, load_model, save_model, cluster
from h2o.frame import H2OFrame
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas import DataFrame, Series, concat
from numpy import sum, sqrt

from . import __version__


class InsolverGLMWrapper:
    """Insolver wrapper for Generalized Linear Models.

    Attributes:
        backend (str): Framework for building GLM, currently 'h2o' and 'sklearn' are supported.
        family (:obj:`str`, :obj:`float`, :obj:`int`, optional): Distribution for GLM. Supports any family from h2o as
        str. For sklearn supported `str` families are ['gaussian', 'normal', 'poisson', 'gamma', 'inverse_gaussian'],
        also may be defined as `int` or `float` as a power for Tweedie GLM. By default, Gaussian GLM is fitted.
        link (:obj:`str`, optional): Link function for GLM. If `None`, sets to default value for both h2o and sklearn.
        standardize (:obj:`bool`, optional): Whether to standardize data before fitting the model. Enabled by default.
        h2o_init_params (:obj:`dict`, optional): Parameters passed to `h2o.init()`, when `backend` == 'h2o'.
        load (:obj:`str`, optional): Path to GLM model to load from disk.
        **kwargs: Parameters for GLM estimators (for H2OGeneralizedLinearEstimator or TweedieRegressor) except
        `family` (`power` for TweedieRegressor) and `link`.
        """
    def __init__(self, backend, family=None, link=None, standardize=True, h2o_init_params=None,
                 load_path=None, **kwargs):
        if backend in ['h2o', 'sklearn']:
            self.backend = backend
        else:
            raise NotImplementedError("Only ['h2o', 'sklearn'] are supported as backends.")

        if load_path is not None:
            load_path = os.path.normpath(load_path)
            if self.backend == 'h2o':
                no_progress()
                self.model = load_model(load_path)
            else:
                with open(load_path, 'rb') as _model:
                    self.model = pickle.load(_model)
        else:
            family = family if family is not None else 'gaussian'
            link = link if link is not None else 'family_default' if backend == 'h2o' else 'auto'
            if backend == 'h2o':
                no_progress()
                if cluster() is None:
                    if h2o_init_params is not None:
                        init(**h2o_init_params)
                    else:
                        init()
                self.model = H2OGeneralizedLinearEstimator(family=family, link=link, standardize=standardize, **kwargs)
            elif backend == 'sklearn':
                if isinstance(family, str):
                    family_power = {'gaussian': 0, 'normal': 0, 'poisson': 1, 'gamma': 2, 'inverse_gaussian': 3}
                    if family in family_power.keys():
                        family = family_power[family]
                    else:
                        raise NotImplementedError('Distribution is not supported with sklearn backend.')
                self.model = Pipeline([('scaler', StandardScaler(with_mean=standardize, with_std=standardize)),
                                       ('glm', TweedieRegressor(power=family, link=link, **kwargs))])
        self.best_params, self.features, self.offset, self.standardize = None, None, None, standardize

    def __call__(self):
        return self.model

    def __x_y_to_h2o_frame(self, X, y, sample_weight, params, X_valid, y_valid, sample_weight_valid):
        params = params
        features, target, train_set = None, None, None

        if isinstance(X, (DataFrame, Series)) & isinstance(y, (DataFrame, Series)):
            features = X.columns.tolist() if isinstance(X, DataFrame) else X.name
            target = y.columns.tolist() if isinstance(y, DataFrame) else y.name
            if (sample_weight is not None) & isinstance(sample_weight, (DataFrame, Series)):
                params['offset_column'] = (sample_weight.columns.tolist() if isinstance(sample_weight, DataFrame)
                                           else sample_weight.name)
                self.offset = params['offset_column']
                # noinspection PyPep8Naming
                X = concat([X, sample_weight], axis=1)
            train_set = H2OFrame(concat([X, y], axis=1))
        else:
            NotImplementedError('X, y are supposed to be pandas DataFrame or Series')

        if (X_valid is not None) & (y_valid is not None):
            if isinstance(X_valid, (DataFrame, Series)) & isinstance(y_valid, (DataFrame, Series)):
                if ((sample_weight_valid is not None) & isinstance(sample_weight_valid, (DataFrame, Series)) &
                        (sample_weight is not None)):
                    # noinspection PyPep8Naming
                    X_valid = concat([X_valid, sample_weight_valid], axis=1)
                valid_set = H2OFrame(concat([X_valid, y_valid], axis=1))
                params['validation_frame'] = valid_set
            else:
                NotImplementedError('X_valid, y_valid are supposed to be pandas DataFrame or Series')
        return features, target, train_set, params

    def fit(self, X, y, sample_weight=None, X_valid=None, y_valid=None, sample_weight_valid=None, **kwargs):
        """Fit a Generalized Linear Model.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            sample_weight (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Training sample weights.
            X_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation data.
            y_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation target values.
            sample_weight_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation sample weights.
            **kwargs: Other parameters passed to H2OGeneralizedLinearEstimator.
        """
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            if isinstance(X, (DataFrame, Series)):
                self.features = X.columns.tolist() if isinstance(X, DataFrame) else X.name
            self.model.fit(X, y, glm__sample_weight=sample_weight)
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            features, target, train_set, params = self.__x_y_to_h2o_frame(X, y, sample_weight, {**kwargs}, X_valid,
                                                                          y_valid, sample_weight_valid)
            self.model.train(y=target, x=features, training_frame=train_set, **params)
        else:
            NotImplementedError('Error with the backend choice.')

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
            predictions = self.model.predict(X)
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            if (self.offset is not None) and (sample_weight is not None):
                # noinspection PyPep8Naming
                X = concat([X, sample_weight], axis=1)
            h2o_predict = X if isinstance(X, H2OFrame) else H2OFrame(X)
            predictions = self.model.predict(h2o_predict, **kwargs).as_data_frame().values.reshape(-1)
        else:
            NotImplementedError('Error with the backend choice.')
            predictions = None
        return predictions

    def save_model(self, path=None, name=None, **kwargs):
        """Saving the GLM model.

        Args:
            path (:obj:`str`, optional): Path to save the model. Using current working directory by default.
            name (:obj:`str`, optional): Optional, name of the model.
            **kwargs: Parameters passed to h2o.save_model().
        """
        path = os.getcwd() if path is None else os.path.normpath(path)
        def_name = f"insolver_{__version__.replace('.', '-')}_glm_{self.backend}_{round(time.time() * 1000)}"
        name = name if name is not None else def_name
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            with open(os.path.join(path, name), 'wb') as _model:
                pickle.dump(self.model, _model, pickle.HIGHEST_PROTOCOL)
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            model_path = save_model(model=self.model, path=path, **kwargs)
            os.rename(model_path, os.path.join(os.path.dirname(model_path), name))
        else:
            NotImplementedError('Error with the backend choice.')

    def coef_norm(self):
        """Output GLM coefficients for standardized data.

        Returns:
            dict: {:obj:`str`: :obj:`float`}m Dictionary containing GLM coefficients for standardized data.
        """
        if self.standardize:
            if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
                if self.features is None:
                    self.features = [f'Variable_{i}' for i in range(len(list(self.model.named_steps['glm'].coef_)))]
                coefs = zip(['Intercept'] + self.features,
                            [self.model.named_steps['glm'].intercept_] + list(self.model.named_steps['glm'].coef_))
                coefs = {x[0]: x[1] for x in coefs}
            elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
                coefs = self.model.coef_norm()
            else:
                NotImplementedError('Error with the backend choice.')
                coefs = dict()
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
            if self.features is None:
                self.features = [f'Variable_{i}' for i in range(len(list(self.model.named_steps['glm'].coef_)))]
            if self.standardize:
                intercept = self.model.named_steps['glm'].intercept_ - sum(self.model.named_steps['glm'].coef_ *
                                                                           self.model.named_steps['scaler'].mean_ /
                                                                           sqrt(self.model.named_steps['scaler'].var_))
                coefs = self.model.named_steps['glm'].coef_ / sqrt(self.model.named_steps['scaler'].var_)
                coefs = zip(['Intercept'] + self.features, [intercept] + list(coefs))
            else:
                coefs = zip(['Intercept'] + self.features,
                            [self.model.named_steps['glm'].intercept_] + list(self.model.named_steps['glm'].coef_))
            coefs = {x[0]: x[1] for x in coefs}
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            coefs = self.model.coef()
        else:
            NotImplementedError('Error with the backend choice.')
            coefs = dict()
        return coefs

    def optimize_hyperparam(self, hyper_params, X, y, sample_weight=None, X_valid=None, y_valid=None,
                            sample_weight_valid=None, h2o_train_params=None, **kwargs):
        """Hyperparameter optimization & fitting GLM.

        Args:
            hyper_params:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            sample_weight (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Training sample weights.
            X_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation data.
            y_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation target values.
            sample_weight_valid (:obj:`pd.DataFrame`, :obj:`pd.Series`, optional): Validation sample weights.
            h2o_train_params (:obj:`dict`, optional): Parameters passed to `H2OGridSearch.train()`.
            **kwargs: Other parameters passed to H2OGridSearch.

        Returns:
            dict: {`hyperparameter_name`: `optimal_choice`}, Dictionary containing optimal hyperparameter choice.
        """
        if (self.backend == 'sklearn') & isinstance(self.model, Pipeline):
            pass
        #     if isinstance(X, (DataFrame, Series)):
        #         self.features = X.columns.tolist() if isinstance(X, DataFrame) else X.name
        #     self.model.fit(X, y, glm__sample_weight=sample_weight)
        elif (self.backend == 'h2o') & isinstance(self.model, H2OGeneralizedLinearEstimator):
            params = dict() if h2o_train_params is None else h2o_train_params
            features, target, train_set, params = self.__x_y_to_h2o_frame(X, y, sample_weight, params, X_valid, y_valid,
                                                                          sample_weight_valid)
            model_grid = H2OGridSearch(model=self.model, hyper_params=hyper_params, **kwargs)
            model_grid.train(y=target, x=features, training_frame=train_set, **params)
            sorted_grid = model_grid.get_grid(sort_by='residual_deviance', decreasing=False)
            self.best_params = sorted_grid.sorted_metric_table().loc[0, :'model_ids'].drop('model_ids').to_dict()
            self.best_params = {key: self.best_params[key].replace('[', '').replace(']', '')
                                for key in self.best_params.keys() if key != ''}
            self.best_params = {key: float(self.best_params[key]) if self.best_params[key].isdigit()
                                else self.best_params[key] for key in self.best_params.keys()}
            self.model = sorted_grid.models[0]
        else:
            NotImplementedError('Error with the backend choice.')
        return self.best_params
