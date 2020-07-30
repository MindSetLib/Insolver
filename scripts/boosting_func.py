# TODO: Hyperopt internal usage, training class or function, docstrings data types.
import pickle
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cgb
# from hyperopt import hp, tpe, space_eval
# from hyperopt.fmin import fmin


def gb_eval_dev_poisson(yhat, y, weight=None):
    """Function for Poisson Deviance evaluation.

    Args:
        yhat: np.ndarray object with predictions.
        y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable.
        weight: Weights for weighted metric.

    Returns:
        (str, float), tuple with metrics name and its value.
    """
    that = yhat + 1
    if type(y) in [xgb.DMatrix, lgb.Dataset]:
        t = y.get_label() + 1
        if type(y) == xgb.DMatrix:
            return 'dev_poisson', 2 * np.sum(t * np.log(t / that) - (t - that))
        if type(y) == lgb.Dataset:
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
    if type(y) in [xgb.DMatrix, lgb.Dataset]:
        t = y.get_label()
        if type(y) == xgb.DMatrix:
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat)
        if type(y) == lgb.Dataset:
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat), False
    else:
        if weight:
            return 2 * np.sum(weight*(-np.log(y/yhat) + (y-yhat)/yhat))
        else:
            return 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)


def objective_gb(params):
    """Objective function for hyperopt. Optimizing mean cross-validation error with XGBoost.

    Args:
        params: dict object passed to hyperopt fmin() function.

    Returns:
        float: mean cross-validation error for XGBoost, LightGBM or Catboost utilizing params.
    """
    if 'alg' in params.keys():
        alg = params.pop('alg')
    else:
        alg = None
    for x in ['max_depth', 'num_boost_round', 'max_leaves', 'max_bin']:
        if x in params.keys():
            params[x] = int(params[x])

    n_b_r = params.pop('num_boost_round')
    data = params.pop('data')
    nfold = params.pop('nfold')
    e_s_r = params.pop('early_stopping_rounds')
    stratified = params.pop('stratified')
    shuffle = params.pop('shuffle')
    name = ''
    if alg == 'xgboost':
        dtrain = xgb.DMatrix(data[0], data[1]/data[2]) if 'poisson' in params['objective'].lower() \
            else xgb.DMatrix(data[0], data[1])
        if 'feval' in params.keys():
            feval = params.pop('feval')
            name = feval(dtrain.get_label(), dtrain)[0]
        else:
            feval = None
        maximize = params.pop('maximize')
        cv_result = xgb.cv(params, dtrain, num_boost_round=n_b_r, nfold=nfold, seed=0, maximize=maximize,
                           stratified=stratified, shuffle=shuffle, feval=feval, early_stopping_rounds=e_s_r)
        score = cv_result['test-{}-mean'.format(name)][-1:].values[0]
    elif alg == 'lightgbm':
        dtrain = lgb.Dataset(data[0], data[1] / data[2]) if 'poisson' in params['objective'].lower() \
            else lgb.Dataset(data[0], data[1])
        if 'feval' in params.keys():
            feval = params.pop('feval')
            name = feval(dtrain.get_label(), dtrain)[0]
        else:
            feval = None
            name = params['metric']
        cv_result = lgb.cv(params, dtrain, num_boost_round=n_b_r, nfold=nfold, seed=0, stratified=stratified,
                           shuffle=shuffle, feval=feval, early_stopping_rounds=e_s_r)
        score = cv_result['{}-mean'.format(name)][-1]
    else:
        dtrain = cgb.Pool(data[0], data[1] / data[2]) if 'poisson' in params['objective'].lower() \
            else cgb.Pool(data[0], data[1])
        cv_result = cgb.cv(params=params, dtrain=dtrain, num_boost_round=n_b_r,
                           nfold=nfold, seed=0, stratified=stratified, shuffle=shuffle,
                           early_stopping_rounds=e_s_r)
        name = params['objective']
        score = cv_result['test-{}-mean'.format(name)][-1:].values[0]
    return score


def train_gb_best_params(params, dtrain, evals, early_stopping_rounds, evals_result=None, verbose_eval=None):
    """Function to train XGBoost, LightGBM or Catboost estimator from set of parameters, passed from hyperopt.

    Args:
        params (dict): hyperparameters from hyperopt space_eval() function.
        dtrain: xgb.DMatrix, lgb.Dataset or list (for Catboost), to train model on.
        evals: list of pairs (DMatrix, str). Same from xgb.train().
        early_stopping_rounds (int): Same from xgb.train() or lgb.train() or cgb.train().
        evals_result (dict): Same from xgb.train().
        verbose_eval: bool or int. Same from xgb.train() or lgb.train() or cgb.train().

    Returns:
        xgb.Booster or lgb.Booster or catboost object, trained model
    """
    for label in ['nfold', 'data', 'early_stopping_rounds', 'stratified', 'shuffle']:
        if label in params.keys():
            del params[label]
    alg = params.pop('alg')
    n_b_r = 10 if 'num_boost_round' not in params.keys() else params.pop('num_boost_round')

    if 'feval' in params.keys():
        feval = params.pop('feval')
    else:
        feval = None

    if alg == 'xgboost':
        data = xgb.DMatrix(dtrain[0], dtrain[1]/dtrain[2]) if 'poisson' in params['objective'].lower() \
            else xgb.DMatrix(dtrain[0], dtrain[1])
        evals = [(xgb.DMatrix(x[0][0], x[0][1]/x[0][2]), x[1]) if len(x[0]) == 3 else
                 (xgb.DMatrix(x[0][0], x[0][1]), x[1]) for x in evals]
        maximize = params.pop('maximize')
        return xgb.train(params=params, dtrain=data, num_boost_round=n_b_r, evals=evals, feval=feval,
                         maximize=maximize, early_stopping_rounds=early_stopping_rounds, evals_result=evals_result,
                         verbose_eval=verbose_eval)
    elif alg == 'lightgbm':
        data = lgb.Dataset(dtrain[0], dtrain[1] / dtrain[2]) if 'poisson' in params['objective'].lower() \
            else lgb.Dataset(dtrain[0], dtrain[1])
        valid_names = [x[1] for x in evals]
        valid_sets = [(lgb.Dataset(x[0][0], x[0][1]/x[0][2])) if len(x[0]) == 3 else
                      (lgb.Dataset(x[0][0], x[0][1])) for x in evals]
        return lgb.train(params=params, train_set=data, num_boost_round=n_b_r, valid_sets=valid_sets, feval=feval,
                         valid_names=valid_names, early_stopping_rounds=early_stopping_rounds,
                         evals_result=evals_result, verbose_eval=verbose_eval)
    else:
        data = cgb.Pool(dtrain[0], dtrain[1] / dtrain[2]) if 'poisson' in params['objective'].lower() \
            else cgb.Pool(dtrain[0], dtrain[1])
        valid_sets = [cgb.Pool(x[0][0], x[0][1]/x[0][2]) if len(x[0]) == 3 else
                      cgb.Pool(x[0][0], x[0][1]) for x in evals]
        return cgb.train(params=params, dtrain=data, num_boost_round=n_b_r,
                         evals=valid_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)


def save_model(model, params, name, target=None, suffix=None):
    if type(model) == xgb.Booster:
        name = f'{name}_xgboost'
    elif type(model) == lgb.Booster:
        name = f'{name}_lightgbm'
    elif type(model) == cgb.core.CatBoost:
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
