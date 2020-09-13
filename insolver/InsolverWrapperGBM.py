import warnings
import pickle
from functools import partial

from xgboost import DMatrix, cv as xcv, train as xtrain, Booster as XBooster, XGBClassifier, XGBRegressor
from lightgbm import Dataset, cv as lcv, train as ltrain, Booster as LBooster, LGBMClassifier, LGBMRegressor
from catboost import Pool, cv as ccv, train as ctrain, CatBoost, CatBoostClassifier, CatBoostRegressor
from hyperopt import tpe, space_eval, Trials, STATUS_OK, STATUS_FAIL, fmin  # , hp


def objective_gb(params, algorithm, cv_params, data_params, X, y):
    if data_params is None:
        data_params = dict()
    for x in ['max_depth', 'num_boost_round', 'max_leaves', 'max_bin', 'num_leaves',
              'min_child_samples', 'min_data_in_leaf']:
        if x in params.keys():
            params[x] = int(params[x])
    if algorithm == 'xgboost':
        dtrain = DMatrix(X, y, **data_params)
        cv_result = xcv(params=params, dtrain=dtrain, **cv_params)
        name = [i for i in cv_result.columns if all([i.startswith('test-'), i.endswith('-mean')])][-1]
        score = cv_result[name][-1:].values[0]
    elif algorithm == 'lightgbm':
        dtrain = Dataset(X, y, **data_params)
        cv_result = lcv(params=params, train_set=dtrain, **cv_params)
        name = [i for i in cv_result.keys() if i.endswith('-mean')][-1]
        score = cv_result[name][-1]
    elif algorithm == 'catboost':
        dtrain = Pool(X, y, **data_params)
        cv_result = ccv(params=params, dtrain=dtrain, **cv_params)
        name = [i for i in cv_result.columns if all([i.startswith('test-'), i.endswith('-mean')])][-1]
        score = cv_result[name][-1:].values[0]
    else:
        warnings.warn('Error occurred in "algorithm" attribute')
        score = 0
    return {'loss': score, 'status': STATUS_OK}


class InsolverGradientBoostingWrapper(object):
    def __init__(self, algorithm, task):
        if algorithm in ['xgboost', 'lightgbm', 'catboost']:
            self.algorithm = algorithm
        else:
            warnings.warn('Specified algorithm parameter is not supported. '
                          'Try to enter one of the following options: ["xgboost", "lightgbm", "catboost"].')
        if task in ['regression', 'classification']:
            self.task = task
        else:
            warnings.warn('Specified task parameter is not supported. '
                          'Try to enter one of the following options: ["regression", "classification"].')
        self.trials, self.best_params, self.core_params, self.data_params = None, None, None, None
        self.model, self.booster = None, None

    cv_parameters_default_xgboost = {'num_boost_round': 10,
                                     'nfold': 3,
                                     'stratified': False,
                                     'folds': None,
                                     'metrics': (),
                                     'obj': None,
                                     'feval': None,
                                     'maximize': False,
                                     'early_stopping_rounds': None,
                                     'fpreproc': None,
                                     'as_pandas': True,
                                     'verbose_eval': None,
                                     'show_stdv': True,
                                     'seed': 0,
                                     'callbacks': None,
                                     'shuffle': True}

    cv_parameters_default_lightgbm = {'num_boost_round': 100,
                                      'folds': None,
                                      'nfold': 5,
                                      'stratified': True,
                                      'shuffle': True,
                                      'metrics': None,
                                      'fobj': None,
                                      'feval': None,
                                      'init_model': None,
                                      'feature_name': 'auto',
                                      'categorical_feature': 'auto',
                                      'early_stopping_rounds': None,
                                      'fpreproc': None,
                                      'verbose_eval': None,
                                      'show_stdv': True,
                                      'seed': 0,
                                      'callbacks': None,
                                      'eval_train_metric': False,
                                      'return_cvbooster': False}

    cv_parameters_default_catboost = {'iterations': None,
                                      'num_boost_round': None,
                                      'fold_count': 3,
                                      'nfold': None,
                                      'inverted': False,
                                      'partition_random_seed': 0,
                                      'seed': None,
                                      'shuffle': True,
                                      'logging_level': None,
                                      'stratified': None,
                                      'as_pandas': True,
                                      'metric_period': None,
                                      'verbose': None,
                                      'verbose_eval': None,
                                      'plot': False,
                                      'early_stopping_rounds': None,
                                      'folds': None,
                                      'type': 'Classical'}

    def hyperopt_cv(self, X, y, params, cv_params, data_params=None, max_evals=10, fn=None, algo=None, timeout=None):
        # TODO: Add default CV params if not specified.
        trials = Trials()
        if data_params is not None:
            self.data_params = data_params
        if algo is None:
            algo = tpe.suggest
        if fn is None:
            fn = partial(objective_gb, algorithm=self.algorithm, X=X, y=y, cv_params=cv_params, data_params=data_params)
        try:
            best = fmin(fn=fn, space=params, trials=trials, algo=algo, max_evals=max_evals, timeout=timeout)
            best_params = space_eval(params, best)
            # TODO: Check whether all int parameters are int after hyperopt optimization with aliases
            for x in ['max_depth', 'num_boost_round', 'max_leaves', 'max_bin', 'num_leaves',
                      'min_child_samples', 'min_data_in_leaf']:
                if x in best_params.keys():
                    best_params[x] = int(best_params[x])
            self.best_params, self.trials = best_params, trials

            core_params = ['num_boost_round', 'obj', 'feval', 'maximize', 'early_stopping_rounds',
                           'verbose_eval', 'callbacks', 'fobj', 'init_model', 'feature_name', 'categorical_feature',
                           'iterations', 'logging_level', 'metric_period', 'verbose', 'plot']
            self.core_params = {i: cv_params[i] for i in cv_params if i in core_params}

        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

    def fit_booster(self, X, y, data_params=None, core_params=None):
        dtrain_params, train_params = dict(), dict()
        if self.data_params is not None:
            dtrain_params.update(self.data_params)
        if data_params is not None:
            dtrain_params.update(data_params)
        if self.core_params is not None:
            train_params.update(self.core_params)
        if core_params is not None:
            train_params.update(core_params)
        if self.best_params is not None:
            params = self.best_params
        else:
            params = {}

        if self.algorithm == 'xgboost':
            dtrain = DMatrix(X, y, **dtrain_params)
            if 'evals' in train_params.keys():
                train_params['evals'] = [(DMatrix(i[0][0], i[0][1]), i[1]) for i in train_params['evals']]
            self.booster = xtrain(params=params, dtrain=dtrain, **train_params)
        elif self.algorithm == 'lightgbm':
            dtrain = Dataset(X, y, **dtrain_params)
            if 'evals' in train_params.keys():
                evals = train_params.pop('evals')
                train_params['valid_names'] = [i[1] for i in evals]
                train_params['valid_sets'] = [(Dataset(i[0][0], i[0][1])) for i in evals]
            self.booster = ltrain(params=params, train_set=dtrain, **train_params)
        elif self.algorithm == 'catboost':
            dtrain = Pool(X, y, **dtrain_params)
            if 'evals' in train_params.keys():
                train_params['evals'] = [Pool(i[0][0], i[0][1]) for i in train_params['evals']]
            self.booster = ctrain(params=params, dtrain=dtrain, **train_params)
        else:
            warnings.warn('Specified algorithm parameter is not supported.')

    def model_init(self, params=None):
        hparams = dict()
        if self.core_params is not None:
            hparams.update(self.core_params)
        if self.best_params is not None:
            hparams.update(self.best_params)
        if params is not None:
            hparams.update(params)

        aliases = {'eta': 'learning_rate', 'boosting': 'boosting_type', 'max_leaves': 'num_leaves',
                   'num_iterations': 'n_estimators', 'num_iteration': 'n_estimators',
                   'n_iter': 'n_estimators', 'num_tree': 'n_estimators', 'num_trees': 'n_estimators',
                   'num_round': 'n_estimators', 'num_rounds': 'n_estimators', 'num_boost_round': 'n_estimators'}

        for x in hparams.keys():
            if x in aliases.keys():
                hparams[aliases[x]] = hparams.pop(x)

        if self.task == 'classification':
            if self.algorithm == 'xgboost':
                self.model = XGBClassifier(**hparams)
            elif self.algorithm == 'lightgbm':
                self.model = LGBMClassifier(**hparams)
            elif self.algorithm == 'catboost':
                self.model = CatBoostClassifier(**hparams)
            else:
                warnings.warn('Specified algorithm parameter is not supported.')
        elif self.task == 'regression':
            if self.algorithm == 'xgboost':
                self.model = XGBRegressor(**hparams)
            elif self.algorithm == 'lightgbm':
                self.model = LGBMRegressor(**hparams)
            elif self.algorithm == 'catboost':
                self.model = CatBoostRegressor(**hparams)
            else:
                warnings.warn('Specified algorithm parameter is not supported.')
        else:
            warnings.warn('Tasks other than "classification" and "regression" are not supported.')

    def fit(self, X, y, **kwargs):
        if self.model is None:
            warnings.warn('Model is not initiated, please use .model_init() method.')
        else:
            self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self.model is None:
            warnings.warn('Please fit or load a model first.')
        else:
            return self.model.predict(X, **kwargs)

    def predict_booster(self, X, **kwargs):
        if self.booster is None:
            warnings.warn('Please fit or load a booster first.')
        else:
            if isinstance(self.booster, XBooster):
                data = X if isinstance(X, DMatrix) else DMatrix(X)
                return self.booster.predict(data, **kwargs)
            elif isinstance(self.booster, LBooster):
                return self.booster.predict(X, **kwargs)
            elif isinstance(self.booster, CatBoost):
                return self.booster.predict(X, **kwargs)

    def save_booster(self, name, target=None, suffix=None):
        if isinstance(self.model, XBooster):
            name = f'{name}_xgboost'
        elif isinstance(self.model, LBooster):
            name = f'{name}_lightgbm'
        elif isinstance(self.model, CatBoost):
            name = f'{name}_catboost'
        else:
            name = f'{name}_other'

        if suffix:
            name = f'{name}_{suffix}'

        p = self.best_params.copy()
        for key in ['data', 'feval']:
            if key in p.keys():
                del p[key]

        model_dict = {'model': self.model, 'parameters': p}
        if target:
            model_dict['target'] = target
        with open(f'{name}.model', 'wb') as h:
            pickle.dump(model_dict, h, protocol=pickle.HIGHEST_PROTOCOL)

    def load_booster(self, model_path):
        with open(model_path, 'rb') as h:
            model_dict = pickle.load(h)
        target = model_dict['target'] if 'target' in model_dict.keys() else None
        self.booster = model_dict['model']
        self.best_params = model_dict['parameters']
        self.target = target
