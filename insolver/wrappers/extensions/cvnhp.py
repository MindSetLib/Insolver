import types
import functools

from numpy import mean
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error
from hyperopt import STATUS_OK, Trials, tpe, fmin, space_eval


class InsolverCVHPExtension:
    def _hyperopt_obj_cv(self, params, X, y, scoring, cv=None, agg=None, maximize=False, **kwargs):
        """Default hyperopt objective performing K-fold cross-validation.

        Args:
            params (dict): Dictionary of hyperopt parameters.
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            scoring (:obj:`callable`): Metrics passed to cross_val_score calculation.
            cv (:obj:`int, cross-validation generator or an iterable`, optional): Cross-validation strategy from
             sklearn. Performs 5-fold cv by default.
            agg (:obj:`callable`, optional): Function computing the final score out of test cv scores.
            maximize (:obj:`bool`, optional): Indicator whether to maximize or minimize objective.
             Minimizing by default.
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
        error_score = 'raise' if 'error_score' not in kwargs else kwargs.pop('error_score')
        score = agg(cross_val_score(estimator, X, y=y, scoring=scoring, cv=cv, n_jobs=njobs,
                                    error_score=error_score, **kwargs))
        score = -score if maximize else score
        return {'status': STATUS_OK, 'loss': score}

    def hyperopt_cv(self, X, y, params, fn=None, algo=None, max_evals=10, timeout=None,
                    fmin_params=None, fn_params=None, p_last=True):
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
            p_last (:obj:`str`, optional): If model object is a sklearn.Pipeline then apply fit parameters to the last
             step. True by default.

        Returns:
            dict: Dictionary of best choice of hyperparameters. Also best model is fitted.
        """
        if self.backend == 'h2o':
            raise Exception('hyperopt_cv is not supported by `h2o` backend. Use `optimize_hyperparam`')

        trials = Trials()
        algo = tpe.suggest if algo is None else algo
        if isinstance(self.model, Pipeline) and ((fn_params is not None) and ('fit_params' in fn_params)) and p_last:
            fn_params['fit_params'] = {f'{self.model.steps[-1][0]}__{key}': fn_params['fit_params'].get(key)
                                       for key in fn_params['fit_params'].keys()}
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
        if not hasattr(self.model, 'feature_name_'):
            self.model.feature_name_ = X.columns.tolist() if isinstance(X, DataFrame) else [X.name]
        return self.best_params

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
                    scorers = {scorers.__name__.replace('_', ' '): (make_scorer(scorers) if
                               isinstance(scorers, (types.FunctionType, types.BuiltinFunctionType, functools.partial))
                               else scorers)}
                except ValueError:
                    scorers = {scorers.__name__.replace('_', ' '): make_scorer(scorers)}
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
            raise NotImplementedError('_cross_val method is not implemented for backend=`h2o`')
