# TODO: Hyperopt internal usage, training class or function, docstrings data types.
import pickle
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cgb
from hyperopt import hp, tpe, space_eval, Trials
from hyperopt.fmin import fmin


def gb_eval_dev_poisson(yhat, y, weight=None):
    """Function for Poisson Deviance evaluation.

    Args:
        yhat: np.ndarray object with predictions.
        y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable.
        weight: Weights for weighted metric.

    Returns:
        (str, float), tuple with metrics name and its value, if y is xgboost.DMatrix or lightgbm.Dataset;
        float, otherwise.
    """
    that = yhat + 1
    if isinstance(y, (xgb.DMatrix, lgb.Dataset)):
        t = y.get_label() + 1
        if isinstance(y, xgb.DMatrix):
            return 'dev_poisson', 2 * np.sum(t * np.log(t / that) - (t - that))
        if isinstance(y, lgb.Dataset):
            return 'dev_poisson', 2 * np.sum(t * np.log(t / that) - (t - that)), False
    else:
        t = y + 1
        if weight:
            return 2 * np.sum(weight*(t * np.log(t / that) - (t - that)))
        else:
            return 2 * np.sum(t * np.log(t / that) - (t - that))


def gb_eval_dev_gamma(yhat, y, weight=None):
    """Function for Gamma Deviance evaluation.

    Args:
        yhat: np.ndarray object with predictions.
        y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable.
        weight: Weights for weighted metric.

    Returns:
        (str, float), tuple with metrics name and its value.
    """
    if isinstance(y, (xgb.DMatrix, lgb.Dataset)):
        t = y.get_label()
        if isinstance(y, xgb.DMatrix):
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat)
        if isinstance(y, lgb.Dataset):
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat), False
    else:
        if weight:
            return 2 * np.sum(weight*(-np.log(y/yhat) + (y-yhat)/yhat))
        else:
            return 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)


def save_model(model, params, name, target=None, suffix=None):
    if isinstance(model, xgb.Booster):
        name = f'{name}_xgboost'
    elif isinstance(model, lgb.Booster):
        name = f'{name}_lightgbm'
    elif isinstance(model, cgb.core.CatBoost):
        name = f'{name}_catboost'
    else:
        name = f'{name}_other'

    if suffix:
        name = f'{name}_{suffix}'

    p = params.copy()
    for key in ['data', 'feval']:
        if key in p.keys():
            del p[key]

    model_dict = {'model': model, 'parameters': p}
    if target:
        model_dict['target'] = target
    with open(f'{name}.model', 'wb') as h:
        pickle.dump(model_dict, h, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model_path):
    with open(model_path, 'rb') as h:
        model_dict = pickle.load(h)
    target = model_dict['target'] if 'target' in model_dict.keys() else None
    return model_dict['model'], model_dict['parameters'], target
