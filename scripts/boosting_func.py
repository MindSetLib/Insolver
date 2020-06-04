import pickle
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cgb
from sklearn.model_selection import train_test_split
# from hyperopt import hp, tpe, space_eval
# from hyperopt.fmin import fmin


def train_val_test_split(x, y, val_size, test_size, random_state=0, shuffle=True, stratify=None):
    """
    Function for splitting dataset into train/validation/test partitions.
    :param x: np.array object containing predictors
    :param y: np.array object containing target variable
    :param val_size: float, The proportion of the dataset to include in validation partition
    :param test_size: float, The proportion of the dataset to include in test partition
    :param random_state: int, optional (default=0). Passed to train_test_split() from scikit-learn.
    :param shuffle: boolean, optional (default=True). Passed to train_test_split() from scikit-learn.
    :param stratify: array-like or None (default=None). Passed to train_test_split() from scikit-learn.
    :return: tuple, splitting (x_train, x_valid, x_test, y_train, y_valid, y_test)
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state, shuffle=shuffle,
                                                        test_size=test_size, stratify=stratify)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=random_state, shuffle=shuffle,
                                                          test_size=val_size/(1-test_size), stratify=stratify)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def train_test_column_split(x, y, df_column):
    """
    Function for splitting dataset into train/test partitions w.r.t. a column (pd.Series).
    :param x: pd.DataFrame object containing predictors
    :param y: pd.DataFrame object containing target variable
    :param df_column: pd.Series object, for train/test split, assuming it is contained in x
    :return: tuple, splitting (x_train, x_test, y_train, y_test)
    """
    x1, y1, col_name = x.copy(), y.copy(), df_column.name
    y1[col_name] = df_column
    return (x1[x1[col_name] == 'train'].drop(col_name, axis=1), x1[x1[col_name] == 'test'].drop(col_name, axis=1),
            y1[y1[col_name] == 'train'].drop(col_name, axis=1), y1[y1[col_name] == 'test'].drop(col_name, axis=1))


def gb_eval_dev_poisson(yhat, y, weight=None):
    """
    Function for Poisson Deviance evaluation
    :param yhat: np.ndarray object with predictions
    :param y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable
    :param weight: Weights for weighted metric
    :return: (str, float), tuple with metrics name and its value
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
            pdev = 2 * np.sum(weight*(t * np.log(t / that) - (t - that)))
        else:
            pdev = 2 * np.sum(t * np.log(t / that) - (t - that))
        return pdev


def gb_eval_dev_gamma(yhat, y, weight=None):
    """
    Function for Gamma Deviance evaluation
    :param yhat: np.ndarray object with predictions
    :param y: xgb.DMatrix, lgb.Dataset or np.ndarray object with target variable
    :param weight: Weights for weighted metric
    :return: (str, float), tuple with metrics name and its value
    """
    if type(y) in [xgb.DMatrix, lgb.Dataset]:
        t = y.get_label()
        if type(y) == xgb.DMatrix:
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat)
        if type(y) == lgb.Dataset:
            return 'dev_gamma', 2 * np.sum(-np.log(t/yhat) + (t-yhat)/yhat), False
    else:
        if weight:
            gdev = 2 * np.sum(weight*(-np.log(y/yhat) + (y-yhat)/yhat))
        else:
            gdev = 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)
        return gdev


def objective_gb(params):
    """
    Objective function for hyperopt. Optimizing mean cross-validation error with XGBoost.
    :param params: dict object passed to hyperopt fmin() function
    :return: float, mean cross-validation error for XGBoost, LightGBM or Catboost utilizing params
    """
    params['max_depth'] = int(params['max_depth'])
    n_b_r = int(params.pop('num_boost_round'))
    data = params.pop('data')
    nfold = params.pop('nfold')
    e_s_r = params.pop('early_stopping_rounds')
    if type(data) == xgb.DMatrix:
        feval = params.pop('feval')
        name = feval*(data.get_label(), data)[0]
        maximize = params.pop('maximize')
        cv_result = xgb.cv(params, data, num_boost_round=n_b_r, nfold=nfold, seed=0, maximize=maximize,
                           feval=feval, early_stopping_rounds=e_s_r)
        score = cv_result['test-{}-mean'.format(name)][-1:].values[0]
    elif type(data) == lgb.Dataset:
        feval = params.pop('feval')
        name = feval * (data.get_label(), data)[0]
        stratified = params.pop('stratified')
        shuffle = params.pop('shuffle')
        cv_result = lgb.cv(params, data, num_boost_round=n_b_r, nfold=nfold, seed=0, stratified=stratified,
                           shuffle=shuffle, feval=feval, early_stopping_rounds=e_s_r)
        score = cv_result['{}-mean'.format(name)][-1]
    else:
        stratified = params.pop('stratified')
        shuffle = params.pop('shuffle')
        cv_result = cgb.cv(params=params, dtrain=cgb.Pool(data[0], data[1], weight=data[2]), num_boost_round=n_b_r,
                           nfold=nfold, seed=0, stratified=stratified, shuffle=shuffle,
                           early_stopping_rounds=e_s_r)
        name = params['objective']
        score = cv_result['test-{}-mean'.format(name)][-1:].values[0]
    return score


def train_gb_best_params(params, dtrain, evals, early_stopping_rounds, evals_result=None, verbose_eval=None):
    """
    Function to train XGBoost, LightGBM or Catboost estimator from set of parameters, passed from hyperopt.
    :param params: dict, hyperparameters from hyperopt space_eval() function
    :param dtrain: xgb.DMatrix, lgb.Dataset or list (for Catboost), to train model on
    :param evals: list of pairs (DMatrix, str). Same from xgb.train().
    :param early_stopping_rounds: int. Same from xgb.train() or lgb.train() or cgb.train().
    :param evals_result: dict. Same from xgb.train().
    :param verbose_eval: bool or int. Same from xgb.train() or lgb.train() or cgb.train().
    :return: xgb.Booster or lgb.Booster or catboost object, trained model
    """
    for label in ['nfold', 'data', 'early_stopping_rounds', 'stratified', 'shuffle']:
        if label in params.keys():
            del params[label]
    n_b_r = 10 if 'num_boost_round' not in params.keys() else int(params.pop('num_boost_round'))

    if type(dtrain) == xgb.DMatrix:
        maximize = params.pop('maximize')
        feval = params.pop('feval')
        return xgb.train(params=params, dtrain=dtrain, num_boost_round=n_b_r, evals=evals, feval=feval,
                         maximize=maximize, early_stopping_rounds=early_stopping_rounds, evals_result=evals_result,
                         verbose_eval=verbose_eval)
    elif type(dtrain) == lgb.Dataset:
        feval = params.pop('feval')
        valid_sets, valid_names = [x[0] for x in evals], [x[1] for x in evals]
        return lgb.train(params=params, train_set=dtrain, num_boost_round=n_b_r, valid_sets=valid_sets, feval=feval,
                         valid_names=valid_names, early_stopping_rounds=early_stopping_rounds,
                         evals_result=evals_result, verbose_eval=verbose_eval)
    else:
        valid_sets = [cgb.Pool(x[0][0], x[0][1], weight=x[0][2]) if len(x[0]) == 3 else
                      cgb.Pool(x[0][0], x[0][1]) for x in evals]
        return cgb.train(params=params, dtrain=cgb.Pool(dtrain[0], dtrain[1], weight=dtrain[2]), num_boost_round=n_b_r,
                         evals=valid_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)


def save_model(model, params, name, target=None, suffix=None):
    if type(model) == xgb.Booster:
        name = name + '_xgb'
    elif type(model) == lgb.Booster:
        name = name + '_lgb'
    else:
        name = name + '_cgb'

    if suffix:
        name = name + '_' + suffix

    p = params.copy()
    for key in ['data', 'feval']:
        if key in p.keys():
            del p[key]

    model_dict = dict()
    model_dict['model'] = model
    model_dict['parameters'] = p
    if target:
        model_dict['target'] = target
    with open(name + '.model', 'wb') as h:
        pickle.dump(model_dict, h, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model_path):
    with open(model_path, 'rb') as h:
        model_dict = pickle.load(h)
    target = model_dict['target'] if 'target' in model_dict.keys() else None
    return model_dict['model'], model_dict['parameters'], target


# Определим границы, в которых будем искать гиперпараметры
# space_freq = {'data': train_freq,
#               'objective': 'count:poisson',
#               'feval': xgb_eval_dev_poisson,
#               'maximize': False,
#               'nfold': 5,
#               'early_stopping_rounds': 20,
#               'num_boost_round': 300,  # hp.choice('num_boost_round', [50, 300, 500])
#               'max_depth': hp.choice('max_depth', [5, 8, 10, 12, 15]),
#               'min_child_weight': hp.uniform('min_child_weight', 0, 50),
#               'subsample': hp.uniform('subsample', 0.5, 1),
#               'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#               'alpha': hp.uniform('alpha', 0, 1),
#               'lambda': hp.uniform('lambda', 0, 1),
#               'eta': hp.uniform('eta', 0.01, 1),
#               }
#
# space_avgclm = {'data': train_avgclaim,
#                 'objective': 'reg:gamma',
#                 'feval': xgb_eval_dev_gamma,
#                 'maximize': False,
#                 'nfold': 5,
#                 'early_stopping_rounds': 20,
#                 'num_boost_round': 300,  # hp.choice('num_boost_round', [50, 300, 500])
#                 'max_depth': hp.choice('max_depth', [5, 8, 10, 12, 15]),
#                 'min_child_weight': hp.uniform('min_child_weight', 0, 50),
#                 'subsample': hp.uniform('subsample', 0.5, 1),
#                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#                 'alpha': hp.uniform('alpha', 0, 1),
#                 'lambda': hp.uniform('lambda', 0, 1),
#                 'eta': hp.uniform('eta', 0.01, 1),
#                 }

# Оптимизация
# model_freq_best = fmin(fn=objective_xgb, space=space_freq, algo=tpe.suggest, max_evals=50)
# model_avclaim_best = fmin(fn=objective_xgb, space=space_avgclm, algo=tpe.suggest, max_evals=50)

# Оптимальные гиперпараметры
# best_params_freq = space_eval(space_freq, model_freq_best)
# best_params_avclaim = space_eval(space_avgclm, model_avclaim_best)
