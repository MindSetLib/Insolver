from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from insolver.wrappers.base import InsolverWrapperBase


class InsolverGBMWrapper(InsolverWrapperBase):
    """Insolver wrapper for Gradient Boosting Machines.

    Attributes:
        backend (str): Framework for building GBM, 'xgboost', 'lightgbm' and 'catboost' are supported.
        task (str): Task that GBM should solve: Classification or Regression. Values 'reg' and 'class' are supported.
        n_estimators (:obj:`int`, optional): Number of boosting rounds. Equals 100 by default.
        objective (:obj:`str` or :obj:`callable`): Objective function for GBM to optimize.
        load_path (:obj:`str`, optional): Path to GBM model to load from disk.
        **kwargs: Parameters for GBM estimators except `n_estimators` and `objective`. Will not be changed in hyperopt.
    """
    def __init__(self, backend, task=None, n_estimators=100, objective=None, load_path=None, **kwargs):
        super(InsolverGBMWrapper, self).__init__(backend)
        self.algo, self._backends = 'gbm', ['xgboost', 'lightgbm', 'catboost']
        self._tasks = ['class', 'reg']
        self._back_load_dict = {'xgboost': self._pickle_load, 'lightgbm': self._pickle_load,
                                'catboost': self._pickle_load}
        self._back_save_dict = {'xgboost': self._pickle_save, 'lightgbm': self._pickle_save,
                                'catboost': self._pickle_save}
        self.n_estimators, self.objective, self.params = n_estimators, objective, None

        if backend not in self._backends:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

        if load_path is not None:
            self.load_model(load_path)
        else:
            if task in self._tasks:
                gbm_init = {
                    'class': {'xgboost': XGBClassifier, 'lightgbm': LGBMClassifier, 'catboost': CatBoostClassifier},
                    'reg': {'xgboost': XGBRegressor, 'lightgbm': LGBMRegressor, 'catboost': CatBoostRegressor}
                }

                objectives = {
                    'regression': {'xgboost': 'reg:squarederror', 'lightgbm': 'regression', 'catboost': 'RMSE'},
                    'binary': {'xgboost': 'binary:logistic', 'lightgbm': 'binary', 'catboost': 'Logloss'},
                    'multiclass': {'xgboost': 'multi:softmax', 'lightgbm': 'multiclass', 'catboost': 'MultiClass'},
                    'poisson': {'xgboost': 'count:poisson', 'lightgbm': 'poisson', 'catboost': 'Poisson'},
                    'gamma': {'xgboost': 'reg:gamma', 'lightgbm': 'gamma',
                              'catboost': 'Tweedie:variance_power=1.9999999'}
                }
                self.objective = (objectives[self.objective][self.backend] if self.objective in objectives.keys()
                                  else self.objective)
                kwargs.update({'objective': self.objective, 'n_estimators': self.n_estimators})
                self.model, self.params = gbm_init[task][self.backend](**(kwargs if kwargs is not None else {})), kwargs

                def __params_gbm(**params):
                    params.update(self.params)
                    return gbm_init[task][self.backend](**params)

                self.object = __params_gbm
            else:
                raise NotImplementedError(f'Task parameter supports values in {self._tasks}.')

    def fit(self, X, y, **kwargs):
        """Fit a Gradient Boosting Machine.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            **kwargs: Other parameters passed to Scikit-learn API .fit().
        """
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """Predict using GBM with feature matrix X.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Samples.
            **kwargs: Other parameters passed to Scikit-learn API .predict().

        Returns:
            array: Returns predicted values.
        """
        return self.model.predict(X, **kwargs)

    # def cross_val(self, x_train, y_train, strategy, metrics):
    #     fold_metrics, shap_coefs = list(), list()
    #     self.fit(x_train, y_train)
    #     if isinstance(metrics, (list, tuple)):
    #         for metric in metrics:
    #             fold_metrics.append(metric(y_train, self.model.predict(x_train)))
    #     else:
    #         fold_metrics.append(metrics(y_train, self.model.predict(x_train)))
    #     explainer = TreeExplainer(self.model)
    #     shap_coefs.append(explainer.expected_value.tolist() +
    #                       explainer.shap_values(x_train).mean(axis=0).tolist())
    #
    #     for train_index, test_index in strategy.split(x_train):
    #         if isinstance(x_train, DataFrame):
    #             x_train_cv, x_test_cv = x_train.iloc[train_index], x_train.iloc[test_index]
    #             y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
    #         else:
    #             x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
    #             y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
    #         self.fit(x_train_cv, y_train_cv)
    #         predict = self.model.predict(x_test_cv)
    #
    #         if isinstance(metrics, (list, tuple)):
    #             for metric in metrics:
    #                 fold_metrics.append(metric(y_test_cv, predict))
    #         else:
    #             fold_metrics.append(metrics(y_test_cv, predict))
    #
    #         explainer = TreeExplainer(self.model)
    #         shap_coefs.append(explainer.expected_value.tolist() +
    #                           explainer.shap_values(x_test_cv).mean(axis=0).tolist())
    #
    #     if isinstance(metrics, (list, tuple)):
    #         output = DataFrame(array(fold_metrics).reshape(-1, len(metrics)).T, index=[x.__name__ for x in metrics],
    #                            columns=['Overall'] + [f'Fold {x}' for x in range(strategy.n_splits)])
    #     else:
    #         output = DataFrame(array([fold_metrics]), index=[metrics.__name__],
    #                            columns=['Overall'] + [f'Fold {x}' for x in range(strategy.n_splits)])
    #     coefs = DataFrame(array(shap_coefs).T, columns=['Overall'] + [f'Fold {x}' for x in range(strategy.n_splits)],
    #                       index=[['Intercept'] + x_train.columns.tolist()])
    #     return output, coefs
    #
    # def explain_shap(self, data):
    #     self.explainer = TreeExplainer(self.model) if self.model is not None else TreeExplainer(self.booster)
    #     if isinstance(data, Series):
    #         data = DataFrame(data).T
    #     self.shap_values = self.explainer.shap_values(data)
    #
    #     if isinstance(self.shap_values, list) and (len(self.shap_values) == 2):
    #         shap_values = self.shap_values[0]
    #         expected_value = self.explainer.expected_value[0].tolist()
    #     else:
    #         shap_values = self.shap_values
    #         expected_value = [self.explainer.expected_value]
    #
    #     summary_plot(shap_values, data, plot_type='bar')
    #     variables = ['Intercept'] + list(data.columns)
    #     mean_shap = expected_value + shap_values.mean(axis=0).tolist()
    #     return {variables[i]: mean_shap[i] for i in range(len(variables))}
