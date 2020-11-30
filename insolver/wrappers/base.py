import os
import types
import time
import pickle
import functools

from matplotlib.pyplot import show, tight_layout
from numpy import array, mean, broadcast_to
from pandas import DataFrame, Series, concat
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error
from sklearn.inspection import plot_partial_dependence
from pdpbox.pdp import pdp_isolate, pdp_plot

from h2o import no_progress, cluster, init, load_model, save_model
from h2o.frame import H2OFrame

from hyperopt import STATUS_OK, Trials, tpe, fmin, space_eval


class InsolverBaseWrapper:
    def __init__(self, backend):
        self.algo, self.backend, self._backends = None, backend, None
        self._back_load_dict, self._back_save_dict = None, None
        self.object, self.model = None, None
        self.features, self.best_params, self.trials = None, None, None

    def __call__(self):
        return self.model

    def load_model(self, load_path):
        """Loading a model to the wrapper.

        Args:
            load_path (:obj:`str`): Path to the model that will be loaded to wrapper.
        """
        load_path = os.path.normpath(load_path)
        if self.backend in self._back_load_dict.keys():
            self._back_load_dict[self.backend](load_path)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def save_model(self, path=None, name=None, suffix=None, **kwargs):
        """Saving the model contained in wrapper.

        Args:
            path (:obj:`str`, optional): Path to save the model. Using current working directory by default.
            name (:obj:`str`, optional): Optional, name of the model.
            suffix (:obj:`str`, optional): Optional, suffix in the name of the model.
            **kwargs: Other parameters passed to, e.g. h2o.save_model().
        """
        path = os.getcwd() if path is None else os.path.normpath(path)
        def_name = f"insolver_{self.algo}_{self.backend}_{round(time.time() * 1000)}"
        name = name if name is not None else def_name
        name = name if suffix is None else f'{name}_{suffix}'

        if self.backend in self._back_save_dict.keys():
            self._back_save_dict[self.backend](path, name, **kwargs)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def _pickle_load(self, load_path):
        with open(load_path, 'rb') as _model:
            self.model = pickle.load(_model)

    def _pickle_save(self, path, name):
        with open(os.path.join(path, name), 'wb') as _model:
            pickle.dump(self.model, _model, pickle.HIGHEST_PROTOCOL)

    def _hyperopt_obj_cv(self, params, X, y, scoring, cv=None, agg=None, **kwargs):
        """Default hyperopt objective performing K-fold cross-validation.

        Args:
            params (dict): Dictionary of hyperopt parameters.
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            scoring (:obj:`callable`): Metrics passed to cross_val_score calculation.
            cv (:obj:`int, cross-validation generator or an iterable`, optional): Cross-validation strategy from
             sklearn. Performs 5-fold cv by default.
            agg (:obj:`callable`, optional): Function computing the final score out of test cv scores.
            **kwargs: Other parameters passed to sklearn.model_selection.cross_val_score().

        Returns:
            dict: {'status': STATUS_OK, 'loss': `cv_score`}
        """
        agg = mean if agg is None else agg
        cv = KFold(n_splits=5) if cv is None else cv
        params = {key: params[key] if not (isinstance(params[key], float) and params[key].is_integer()) else
                  int(params[key]) for key in params.keys()}
        estimator = self.object(**params)
        njobs = -1 if 'n_jobs' not in kwargs else kwargs.pop('n_jobs')
        score = agg(cross_val_score(estimator, X, y=y, scoring=scoring, cv=cv, n_jobs=njobs, **kwargs))
        return {'status': STATUS_OK, 'loss': score}

    def hyperopt_cv(self, X, y, params, fn=None, algo=None, max_evals=10, timeout=None,
                    fmin_params=None, fn_params=None):
        """Hyperparameter optimization using hyperopt. Using cross-validation to evaluate hyperparameters by default.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            params (dict): Dictionary of hyperparameters passed to hyperopt.
            fn (:obj:`callable`, optional): Objective function to optimize with hyperopt.
            algo (:obj:`callable`, optional): Algorithm for hyperopt. Available choices are: hyperopt.tpe.suggest and
             hyperopt.random.suggest. Using hyperopt.tpe.suggest by default.
            max_evals (:obj:`int`, optional): Number of function evaluations before returning.
            timeout (:obj:`None`, :obj:`int`, optional): Limits search time by parametrized number of seconds.
            If None, then the search process has no time constraint. None by default.
            fmin_params (:obj:`dict`, optional): Dictionary of supplementary arguments for hyperopt.fmin function.
            fn_params (:obj:`dict`, optional):  Dictionary of supplementary arguments for custom fn objective function.

        Returns:
            dict: Dictionary of best choice of hyperparameters. Also best model is fitted.
        """
        if self.backend == 'h2o':
            raise Exception('hyperopt_cv is not supported by `h2o` backend. Use `optimize_hyperparam`')

        trials = Trials()
        algo = tpe.suggest if algo is None else algo
        if fn is None:
            scoring = (None if not (isinstance(fn_params, dict) and ('scoring' in fn_params.keys()))
                       else fn_params.pop('scoring'))
            scoring = make_scorer(mean_squared_error) if scoring is None else scoring
            try:
                check_scoring(self, scoring)
            except ValueError:
                scoring = make_scorer(scoring)
            fn = functools.partial(self._hyperopt_obj_cv, X=X, y=y, scoring=scoring,
                                   **(fn_params if fn_params is not None else {}))
        best = fmin(fn=fn, space=params, trials=trials, algo=algo, max_evals=max_evals, timeout=timeout,
                    **(fmin_params if fmin_params is not None else {}))
        best_params = space_eval(params, best)
        best_params = {key: best_params[key] if not (isinstance(best_params[key], float) and
                                                     best_params[key].is_integer()) else int(best_params[key])
                       for key in best_params.keys()}
        self.best_params, self.trials = best_params, trials
        self.model = self.object(**self.best_params)
        self.model.fit(X, y, **({} if not ((fn_params is not None) and ('fit_params' in fn_params))
                                else fn_params['fit_params']))
        return self.best_params

    def pdp(self, X, features, feature_name, plot_backend='sklearn', **kwargs):
        if self.backend == 'h2o':
            return
        else:
            if plot_backend == 'sklearn':
                if self.backend in ['catboost', 'lightgbm']:
                    self.model.dummy_ = True
                plot_partial_dependence(self.model, X, features=features, **kwargs)
                tight_layout()
                show()
            elif plot_backend == 'pdpbox':
                pdp_plot(pdp_isolate(self.model, X, features, feature_name), feature_name, **kwargs)
                show()
            else:
                raise NotImplementedError(f'Plot backend {plot_backend} is not implemented.')

    def _cross_val(self, X, y, scoring=None, cv=None, **kwargs):
        if self.backend != 'h2o':
            cv = KFold(n_splits=5) if cv is None else cv
            njobs = -1 if 'n_jobs' not in kwargs else kwargs.pop('n_jobs')
            if 'return_estimator' in kwargs:
                kwargs.pop('return_estimator')
            scoring = make_scorer(mean_squared_error) if scoring is None else scoring

            if callable(scoring) or isinstance(scoring, str):
                scorers = scoring
                try:
                    check_scoring(self.model, scorers)
                    scorers = (make_scorer(scorers) if
                               isinstance(scorers, (types.FunctionType, types.BuiltinFunctionType, functools.partial))
                               else scorers)
                except ValueError:
                    scorers = make_scorer(scorers)
            elif isinstance(scoring, (tuple, list)):
                scorers = []
                for scorer in scoring:
                    try:
                        check_scoring(self.model, scorer)
                        scorers.append([scorer.__name__.replace('_', ' '),
                                        (make_scorer(scorer) if
                                         isinstance(scorer, (types.FunctionType, types.BuiltinFunctionType,
                                                             functools.partial)) else scorer)])
                    except ValueError:
                        scorers.append([scorer.__name__.replace('_', ' '), make_scorer(scorer)])
                scorers = {scorer[0]: scorer[1] for scorer in scorers}
            else:
                raise NotImplementedError(f'Scoring of type {type(scoring)} is not supported.')

            cv_results = cross_validate(self.model, X, y=y, scoring=scorers, cv=cv, n_jobs=njobs,
                                        return_estimator=True, **kwargs)
            estimators = cv_results.pop('estimator')
            cv_results = {key.split('test_')[1]: cv_results[key] for key in cv_results if key.startswith('test_')}
            return estimators, cv_results
        else:
            pass  # TODO: CV for h2o. nfolds...


class InsolverH2OWrapper:
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
        os.rename(model_path, os.path.join(os.path.dirname(model_path), name))

    @staticmethod
    def _x_y_to_h2o_frame(X, y, sample_weight, params, X_valid, y_valid, sample_weight_valid):
        if isinstance(X, (DataFrame, Series)) & isinstance(y, (DataFrame, Series)):
            features = X.columns.tolist() if isinstance(X, DataFrame) else X.name
            target = y.columns.tolist() if isinstance(y, DataFrame) else y.name
            if (sample_weight is not None) & isinstance(sample_weight, (DataFrame, Series)):
                params['offset_column'] = (sample_weight.columns.tolist() if isinstance(sample_weight, DataFrame)
                                           else sample_weight.name)
                # noinspection PyPep8Naming
                X = concat([X, sample_weight], axis=1)
            train_set = H2OFrame(concat([X, y], axis=1))
        else:
            raise TypeError('X, y are supposed to be pandas DataFrame or Series')

        if (X_valid is not None) & (y_valid is not None):
            if isinstance(X_valid, (DataFrame, Series)) & isinstance(y_valid, (DataFrame, Series)):
                if ((sample_weight_valid is not None) & isinstance(sample_weight_valid, (DataFrame, Series)) &
                        (sample_weight is not None)):
                    # noinspection PyPep8Naming
                    X_valid = concat([X_valid, sample_weight_valid], axis=1)
                valid_set = H2OFrame(concat([X_valid, y_valid], axis=1))
                params['validation_frame'] = valid_set
            else:
                raise TypeError('X_valid, y_valid are supposed to be pandas DataFrame or Series')
        return features, target, train_set, params


class InsolverTrivialWrapper(InsolverBaseWrapper):
    """Dummy wrapper for returning trivial "predictions" or actual values for metric comparison and statistics.

    Attributes:
        y (:obj:`pd.DataFrame` or :obj:`pd.Series`): Target values.
        column (:obj:`pd.DataFrame` or :obj:`pd.Series`, optional): Column to perform groupby.
        agg (:obj:`callable`, optional): Aggregation function.
        **kwargs: Other arguments.
    """
    def __init__(self, y=None, column=None, agg=None, x_train=None, y_train=None, **kwargs):
        super(InsolverTrivialWrapper, self).__init__(backend='trivial')
        self._backends, self.x_train, self.y_train = ['trivial'], None, None
        self._back_load_dict, self._back_save_dict = {'trivial': self._pickle_load}, {'trivial': self._pickle_save}

        if isinstance(column, (Series, DataFrame)) or column is None:
            self.column = column
        else:
            raise TypeError(f'Column of type {type(self.column)} is not supported.')
        self.y, self.agg, self.kwargs = y, agg, kwargs

        if (self.column is None) and (self.agg is None):
            self.algo = 'actual'
        elif self.column is None:
            self.algo = self.agg.__name__.replace('_', ' ')
        else:
            self.agg = mean if self.agg is None else self.agg
            name = self.column.name if isinstance(self.column, Series) else self.column.columns[0]
            self.algo = f"{self.agg.__name__} target: {name}"

    def fit(self, X, y):
        self.x_train, self.y_train = X, y

    def predict(self, X):
        """Making dummy predictions.

        Args:
            X (:obj:`pd.DataFrame` or :obj:`pd.Series`): Data.

        Returns:
            array: Trivial model "prediction".
        """
        if (self.column is None) and (self.agg is None):
            output = self.y
        elif self.column is None:
            output = broadcast_to(self.agg(self.y), self.y.shape)
        else:
            _df = concat([self.y, self.column], axis=1)
            output = _df.groupby(_df.columns[1]).transform(self.agg)
        if len(X) == len(output):
            return array(output)
        else:
            raise ValueError(f'Dimension mismatch: input [{len(X)}]; output [{len(output)}]')
