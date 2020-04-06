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
def objective_xgb( params ):
    params['max_depth'] = int(params['max_depth'])
    n_b_r = int(params.pop('num_boost_round'))
    data = params.pop('data')
    feval = params.pop('feval')
    nfold = params.pop('nfold')
    e_s_r = params.pop('early_stopping_rounds')
    maximize = params.pop('maximize')
    cv_result = xgb.cv(params, data, num_boost_round=n_b_r, nfold=nfold, seed=0, maximize=maximize, feval=feval, early_stopping_rounds=e_s_r)
    name, _ = feval(data.get_label(), data)
    score = cv_result['test-{}-mean'.format(name)][-1:].values[0]
    return score

# Определим границы, в которых будем искать гиперпараметры
space_freq = {'data': train_freq,
              'objective': 'count:poisson',
              'feval': xgb_eval_dev_poisson,
              'maximize': False,
              'nfold': 5,
              'early_stopping_rounds': 20,
              'num_boost_round': 300, # hp.choice('num_boost_round', [50, 300, 500])
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
              'num_boost_round': 300, # hp.choice('num_boost_round', [50, 300, 500])
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
best_params_freq = space_eval(space, model_freq_best)
best_params_avclaim = space_eval(space, model_avclaim_best)