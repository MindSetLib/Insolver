import numpy as np
import xgboost as xgb
from hyperopt import hp, tpe, space_eval
from hyperopt.fmin import fmin
from sklearn.model_selection import train_test_split


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


def xgb_eval_dev_poisson(yhat, dtrain):
    """
    Function for Poisson Deviance evaluation

    :param yhat: np.ndarray object with predictions
    :param dtrain: xgb.DMatrix object with target variable
    :return: (str, float), tuple with metrics name and its value
    """
    y = dtrain.get_label()
    return 'dev_poisson', 2 * np.sum(y*np.log(y/yhat) - (y-yhat))


def xgb_eval_dev_gamma(yhat, dtrain):
    """
    Function for Gamma Deviance evaluation

    :param yhat: np.ndarray object with predictions
    :param dtrain: xgb.DMatrix object with target variable
    :return: (str, float), tuple with metrics name and its value
    """
    y = dtrain.get_label()
    return 'dev_gamma', 2 * np.sum(-np.log(y/yhat) + (y-yhat)/yhat)


def objective_xgb(params):
    """
    Objective function for hyperopt. Optimizing mean cross-validation error with XGBoost.

    :param params: dict object passed to hyperopt fmin() function
    :return: float, mean cross-validation error for XGBoost utilizing params
    """
    params['max_depth'] = int(params['max_depth'])
    n_b_r = int(params.pop('num_boost_round'))
    data = params.pop('data')
    feval = params.pop('feval')
    nfold = params.pop('nfold')
    e_s_r = params.pop('early_stopping_rounds')
    maximize = params.pop('maximize')
    cv_result = xgb.cv(params, data, num_boost_round=n_b_r, nfold=nfold, seed=0, maximize=maximize,
                       feval=feval, early_stopping_rounds=e_s_r)
    name, _ = feval(data.get_label(), data)
    score = cv_result['test-{}-mean'.format(name)][-1:].values[0]
    return score


def train_xgb_best_params(params, dtrain, evals, early_stopping_rounds, evals_result=None, verbose_eval=None):
    """
    Function to train XGBoost estimator from set of parameters, passed from hyperopt.

    :param params: dict, hyperparameters from hyperopt space_eval() function
    :param dtrain: xgb.DMatrix object, to train model on
    :param evals: list of pairs (DMatrix, str). Same from xgb.train().
    :param early_stopping_rounds: int. Same from xgb.train().
    :param evals_result: dict. Same from xgb.train().
    :param verbose_eval: bool or int. Same from xgb.train().
    :return: xgb.Booster object, trained model
    """
    for label in ['nfold', 'data', 'early_stopping_rounds']:
        del params[label]
    n_b_r = int(params.pop('num_boost_round'))
    maximize = params.pop('maximize')
    feval = params.pop('feval')
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=n_b_r, evals=evals, feval=feval, maximize=maximize,
                     early_stopping_rounds=early_stopping_rounds, evals_result=evals_result, verbose_eval=verbose_eval)

# Определим границы, в которых будем искать гиперпараметры
space_freq = {'data': train_freq,
              'objective': 'count:poisson',
              'feval': xgb_eval_dev_poisson,
              'maximize': False,
              'nfold': 5,
              'early_stopping_rounds': 20,
              'num_boost_round': 300,  # hp.choice('num_boost_round', [50, 300, 500])
              'max_depth': hp.choice('max_depth', [5, 8, 10, 12, 15]),
              'min_child_weight': hp.uniform('min_child_weight', 0, 50),
              'subsample': hp.uniform('subsample', 0.5, 1),
              'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
              'alpha': hp.uniform('alpha', 0, 1),
              'lambda': hp.uniform('lambda', 0, 1),
              'eta': hp.uniform('eta', 0.01, 1),
              }

space_avgclm = {'data': train_avgclaim,
                'objective': 'reg:gamma',
                'feval': xgb_eval_dev_gamma,
                'maximize': False,
                'nfold': 5,
                'early_stopping_rounds': 20,
                'num_boost_round': 300,  # hp.choice('num_boost_round', [50, 300, 500])
                'max_depth': hp.choice('max_depth', [5, 8, 10, 12, 15]),
                'min_child_weight': hp.uniform('min_child_weight', 0, 50),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'alpha': hp.uniform('alpha', 0, 1),
                'lambda': hp.uniform('lambda', 0, 1),
                'eta': hp.uniform('eta', 0.01, 1),
                }

# Оптимизация
model_freq_best = fmin(fn=objective_xgb, space=space_freq, algo=tpe.suggest, max_evals=50)
model_avclaim_best = fmin(fn=objective_xgb, space=space_avgclm, algo=tpe.suggest, max_evals=50)

# Оптимальные гиперпараметры
best_params_freq = space_eval(space_freq, model_freq_best)
best_params_avclaim = space_eval(space_avgclm, model_avclaim_best)