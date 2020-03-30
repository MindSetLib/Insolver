
import xgboost as xgb
from hyperopt import hp, tpe, space_eval
from hyperopt.fmin import fmin
import numpy as np


# Определим функцию Deviance для распределения Пуассона

def xgb_eval_dev_poisson(yhat, dtrain):
    y = dtrain.get_label()
    return 'dev_poisson', 2 * np.sum( y*np.log(y/yhat) - (y-yhat) )


# Определим функцию Deviance для распределения Гамма

def xgb_eval_dev_gamma(yhat, dtrain):
    y = dtrain.get_label()
    return 'dev_gamma', 2 * np.sum( -np.log(y/yhat) + (y-yhat)/yhat )


# Определим функцию для оптимизации гиперпараметров алгоритмом TPE

def objective_freq( params ):
    parameters = {
        'objective': 'count:poisson',
        'max_depth':  int(params['max_depth']),
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'eta': params['eta'],
        'num_boost_round': int(params['num_boost_round']),
        'alpha': params['alpha'],
        'lambda': params['lambda'],
    }

    cv_result = xgb.cv(parameters, train_freq, nfold=3, seed=0, maximize=False, feval=xgb_eval_dev_poisson, early_stopping_rounds=10)
    score = cv_result['test-dev_poisson-mean'][-1:].values[0]
    return score


# Определим функцию для оптимизации гиперпараметров алгоритмом TPE

def objective_avclaim( params ):
    parameters = {
        'objective': 'reg:gamma',
        'max_depth':  int(params['max_depth']),
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'num_boost_round': int(params['num_boost_round']),
        'eta': params['eta'],
        'alpha': params['alpha'],
        'lambda': params['lambda'],
    }

    cv_result = xgb.cv(parameters, train_avgclaim, nfold=3, seed=0, maximize=False, feval=xgb_eval_dev_gamma, early_stopping_rounds=20)
    score = cv_result['test-dev_gamma-mean'][-1:].values[0]
    return score


# Определим границы, в которых будем искать гиперпараметры

space = {'num_boost_round': hp.choice('num_boost_round', [300]),
         'max_depth': hp.choice('max_depth', [5, 8, 10, 12, 15]),
         'min_child_weight': hp.uniform('min_child_weight', 0, 50),
         'subsample': hp.uniform('subsample', 0.5, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'alpha': hp.uniform('alpha', 0, 1),
         'lambda': hp.uniform('lambda', 0, 1),
         'eta': hp.uniform('eta', 0.01, 1),
        }


# Оптимизация

model_freq_best = fmin(fn=objective_freq, space=space, algo=tpe.suggest, max_evals=50)


# Оптимальные гиперпараметры

best_params_freq = space_eval(space, model_freq_best)
best_params_freq['objective'] = 'count:poisson'


# Оптимизация

model_avclaim_best = fmin(fn=objective_avclaim(), space=space, algo=tpe.suggest, max_evals=50)


# Оптимальные гиперпараметры

best_params_avclaim = space_eval(space, model_avclaim_best)
best_params_avclaim['objective'] = 'count:poisson'