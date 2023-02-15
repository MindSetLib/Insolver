import functools
from typing import Dict, Union, Tuple, Any, Callable, Optional, Iterable

from numpy import mean
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error
from hyperopt import STATUS_OK, Trials, tpe, rand, fmin, space_eval, hp

from ..base import InsolverBaseWrapper


def hyperopt_obj_cv(
    params: Dict,
    x: Union[DataFrame, Series],
    y: Union[DataFrame, Series],
    wrapper: InsolverBaseWrapper,
    scoring: Callable,
    cv: Union[None, Callable, Iterable, int] = None,
    agg: Optional[Callable] = None,
    maximize: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """Default hyperopt objective performing cross-validation.

    Args:
        params (dict): Dictionary of hyperopt parameters.
        x (pd.DataFrame, pd.Series): Training data.
        y (pd.DataFrame, pd.Series): Training target values.
        wrapper (insolver.wrappers_v2.base.InsolverBaseWrapper): Wrapper containing the model.
        scoring (callable): Metrics passed to cross_val_score calculation.
        cv (int, iterable, cross-validation generator, optional): Cross-validation strategy from
         sklearn. Performs 5-fold cv by default.
        agg (callable, optional): Function computing the final score out of test cv scores.
        maximize (bool, optional): Indicator whether to maximize or minimize objective.
         Minimizing by default.
        **kwargs: Other parameters passed to sklearn.model_selection.cross_val_score().

    Returns:
        dict: {'status': STATUS_OK, 'loss': `cv_score`}
    """
    agg = mean if agg is None else agg
    cv = KFold(n_splits=5) if cv is None else cv
    params = {
        key: params[key] if not (isinstance(params[key], float) and params[key].is_integer()) else int(params[key])
        for key in params.keys()
    }
    estimator = wrapper.init_model(params)
    error_score = "raise" if "error_score" not in kwargs else kwargs.pop("error_score")

    score = agg(
        cross_val_score(
            estimator,
            x,
            y=y,
            scoring=scoring,
            cv=cv,
            error_score=error_score,
            **kwargs,
        )
    )
    score = -score if maximize else score
    return {"status": STATUS_OK, "loss": score}


def hyperopt_cv_proc(
    wrapper: InsolverBaseWrapper,
    x: Union[DataFrame, Series],
    y: Union[DataFrame, Series],
    params: Dict,
    fn: Optional[Callable] = None,
    algo: Union[None, rand.suggest, tpe.suggest] = None,
    max_evals: int = 10,
    timeout: Optional[int] = None,
    fmin_params: Dict[str, Any] = None,
    fn_params: Dict[str, Any] = None,
) -> Tuple[Dict, Trials]:
    """Hyperparameter optimization using hyperopt. Using cross-validation to evaluate hyperparameters by default.

    Args:
        wrapper (insolver.wrappers_v2.base.InsolverBaseWrapper): Wrapper containing the model.
        x (pd.DataFrame, pd.Series): Training data.
        y (pd.DataFrame, pd.Series): Training target values.
        params (dict): Dictionary of hyperparameters passed to hyperopt.
        fn (callable, optional): Objective function to optimize with hyperopt.
        algo (callable, optional): Algorithm for hyperopt. Available choices are: hyperopt.tpe.suggest and
         hyperopt.random.suggest. Using hyperopt.tpe.suggest by default.
        max_evals (int, optional): Number of function evaluations before returning.
        timeout (None, int, optional): Limits search time by parametrized number of seconds.
         If None, then the search process has no time constraint. None by default.
        fmin_params (dict, optional): Dictionary of supplementary arguments for hyperopt.fmin function.
        fn_params (dict, optional):  Dictionary of supplementary arguments for custom fn objective function.

    Returns:
        dict: Dictionary of the best choice of hyperparameters. Also, best model is fitted.
    """
    trials = Trials()
    algo = tpe.suggest if algo is None else algo

    if fn is None:
        scoring = (
            None if not (isinstance(fn_params, dict) and ("scoring" in fn_params.keys())) else fn_params.pop("scoring")
        )
        scoring = make_scorer(mean_squared_error) if scoring is None else scoring
        try:
            check_scoring(wrapper, scoring)
        except ValueError:
            scoring = make_scorer(scoring)

        fn = functools.partial(
            hyperopt_obj_cv,
            x=x,
            y=y,
            wrapper=wrapper,
            scoring=scoring,
            **(fn_params if fn_params is not None else {}),
        )

    best = fmin(
        fn=fn,
        space=params,
        trials=trials,
        algo=algo,
        max_evals=max_evals,
        timeout=timeout,
        **(fmin_params if fmin_params is not None else {}),
    )

    best_params = space_eval(params, best)
    best_params = {
        key: (
            best_params[key]
            if not (isinstance(best_params[key], float) and best_params[key].is_integer())
            else int(best_params[key])
        )
        for key in best_params.keys()
    }

    return best_params, trials


AUTO_SPACE_CONFIG = {
    "xgboost": {
        "max_depth": hp.choice("max_depth", [5, 8, 10, 12, 15]),
        "min_child_weight": hp.uniform("min_child_weight", 0, 50),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "alpha": hp.uniform("alpha", 0, 1),
        "lambda": hp.uniform("lambda", 0, 1),
        "eta": hp.uniform("eta", 0.01, 1),
        "gamma": hp.uniform("gamma", 0.01, 1000),
    },
    "lightgbm": {
        "max_depth": hp.choice("max_depth", [5, 8, 10, 12, 15]),
        "min_child_weight": hp.uniform("min_child_weight", 0, 50),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "alpha": hp.uniform("alpha", 0, 1),
        "num_leaves": hp.quniform("num_leaves", 31, 10000, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 1),
    },
    "catboost": {
        "max_depth": hp.choice("max_depth", [5, 8, 10, 12, 15]),
        "min_child_samples": hp.uniform("min_child_samples", 0, 50),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
        "reg_lambda": hp.uniform("reg_lambda", 2, 30),
        "learning_rate": hp.uniform("learning_rate", 0.01, 1),
    },
}
