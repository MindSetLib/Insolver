from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from insolver.wrappers.base import InsolverWrapperMain


class InsolverGBMWrapper(InsolverWrapperMain):
    def __init__(self, backend, task=None, load_path=None, objective=None, params=None, **kwargs):
        super(InsolverGBMWrapper, self).__init__(backend)
        self._backends = ['xgboost', 'lightgbm', 'catboost']
        self._tasks = ['class', 'reg']
        self._back_load_dict = {'xgboost': self._pickle_load, 'lightgbm': self._pickle_load,
                                'catboost': self._pickle_load}
        self._back_save_dict = {'xgboost': self._pickle_save, 'lightgbm': self._pickle_save,
                                'catboost': self._pickle_save}

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
                self.object = gbm_init[task][self.backend]
                self.model = (self.object(**params) if params is not None
                              else self.object(**(kwargs if kwargs is not None else {})))
            else:
                raise NotImplementedError(f'Task parameter supports values in {self._tasks}.')

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
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
