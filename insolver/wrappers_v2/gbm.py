import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import Optional, Any, Callable, Union, List, Tuple, Dict

from numpy import ndarray, array, abs as npabs, mean, argsort, float64
from pandas import DataFrame, Series

from xgboost import XGBClassifier, XGBRegressor, XGBModel, DMatrix
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost, Pool, EFstrType

from plotly.graph_objects import Figure, Bar

from .base import InsolverBaseWrapper
from .utils import save_pickle


class InsolverGBMWrapper(InsolverBaseWrapper):
    """Insolver wrapper for Gradient Boosting Machines.

    Parameters:
        backend (str): Framework for building GBM, currently 'xgboost', 'lightgbm' and 'catboost' are supported.
        task (str): Task that GBM should solve: Classification or Regression. Values 'reg' and 'class' are supported.
        n_estimators (int, optional): Number of boosting rounds. Equals 100 by default.
        objective (str, callable): Objective function for GBM to optimize.
        **kwargs: Parameters for GBM estimators except `n_estimators` and `objective`. Will not be changed in hyperopt.

    """

    algo = 'gbm'
    _backends = ['xgboost', 'lightgbm', 'catboost']
    _tasks = ["class", "reg"]
    _backend_saving_methods = {
        'xgboost': {'pickle': save_pickle},
        'lightgbm': {'pickle': save_pickle},
        'catboost': {'pickle': save_pickle},
    }

    def __init__(
        self,
        backend: Optional[Literal['xgboost', 'lightgbm', 'catboost']],
        task: Optional[Literal['class', 'reg']] = 'reg',
        objective: Union[None, str, Callable] = None,
        n_estimators: int = 100,
        **kwargs: Any,
    ):
        self._get_init_args(vars())

        # Checks on supported backends and tasks
        if backend not in self._backends:
            raise ValueError(f'Invalid "{backend}" backend argument. Supported backends: {self._backends}.')
        if task not in self._tasks:
            raise ValueError(f'Invalid "{task}" task argument. Supported tasks: {self._tasks}.')

        self.backend = backend
        self.task = task
        self.objective = objective
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.model = self.init_model()
        self.__dict__.update(self.metadata)

    def _init_gbm_xgboost(self, **params: Any) -> XGBModel:
        model = XGBModel()  # Just to mitigate referenced before assignment warning

        reg_obj = [
            'reg:squarederror',
            'reg:squaredlogerror',
            'reg:logistic',
            'reg:pseudohubererror',
            'count:poisson',
            'survival:cox',
            'survival:aft',
            'reg:gamma',
            'reg:tweedie',
        ]
        class_obj = ['binary:logistic', 'binary:logitraw', 'binary:hinge', 'multi:softmax', 'multi:softprob']
        rank_obj = ['rank:ndcg', 'rank:map']

        alias = {
            'regression': 'reg:squarederror',
            'logit': 'reg:logistic',
            'binary': 'binary:logistic',
            'multiclass': 'multi:softmax',
            'poisson': 'count:poisson',
            'gamma': 'reg:gamma',
        }

        # Checks on supported objectives vs tasks
        if self.objective is None:
            if self.task == 'reg':
                objective: Union[str, Callable] = 'regression'
            else:
                objective = 'binary'
        else:
            objective = self.objective

        if isinstance(objective, str):
            objective = objective if objective not in alias.keys() else alias[objective]
            if objective in rank_obj:
                raise ValueError(f'Ranking objective "{objective}" is not supported.')
            elif objective in reg_obj:
                if self.task != 'reg':
                    raise ValueError(f'Objective "{objective}" does not match the task "{self.task}".')
            elif objective in class_obj:
                if self.task != 'class':
                    raise ValueError(f'Objective "{objective}" does not match the task "{self.task}".')
            else:
                raise ValueError(
                    f'Invalid objective "{objective}" supported objectives '
                    f'{[*list(alias.keys()), *reg_obj, *class_obj]}.'
                )

        if self.task == 'reg':
            model = XGBRegressor(objective=objective, n_estimators=self.n_estimators, **params)
        if self.task == 'class':
            model = XGBClassifier(objective=objective, n_estimators=self.n_estimators, **params)

        return model

    def _init_gbm_lightgbm(self, **params: Any) -> LGBMModel:
        model = LGBMModel()  # Just to mitigate referenced before assignment warning

        reg_obj = [
            'regression',
            'regression_l2',
            'l2',
            'mean_squared_error',
            'mse',
            'l2_root',
            'root_mean_squared_error',
            'rmse',
            'regression_l1',
            'l1',
            'mean_absolute_error',
            'mae',
            'huber',
            'fair',
            'poisson',
            'quantile',
            'mape',
            'mean_absolute_percentage_error',
            'gamma',
            'tweedie',
        ]
        class_obj = [
            'binary',
            'multiclass',
            'softmax',
            'multiclassova',
            'multiclass_ova',
            'ova',
            'ovr',
            'cross_entropy',
            'xentropy',
            'cross_entropy_lambda',
            'xentlambda',
        ]
        rank_obj = ['lambdarank', 'rank_xendcg', 'xendcg', 'xe_ndcg', 'xe_ndcg_mart', 'xendcg_mart', 'rank_xendcg']

        alias = {'logit': 'binary'}

        # Checks on supported objectives vs tasks
        if self.objective is None:
            if self.task == 'reg':
                objective: Union[str, Callable] = 'regression'
            else:
                objective = 'binary'
        else:
            objective = self.objective

        if isinstance(objective, str):
            objective = objective if objective not in alias.keys() else alias[objective]
            if objective in rank_obj:
                raise ValueError(f'Ranking objective "{objective}" is not supported.')
            elif objective in reg_obj:
                if self.task != 'reg':
                    raise ValueError(f'Objective "{objective}" does not match the task "{self.task}".')
            elif objective in class_obj:
                if self.task != 'class':
                    raise ValueError(f'Objective "{objective}" does not match the task "{self.task}".')
            else:
                raise ValueError(
                    f'Invalid objective "{objective}" supported objectives '
                    f'{[*list(alias.keys()), *reg_obj, *class_obj]}.'
                )

        if self.task == 'reg':
            model = LGBMRegressor(objective=objective, n_estimators=self.n_estimators, **params)
        if self.task == 'class':
            model = LGBMClassifier(objective=objective, n_estimators=self.n_estimators, **params)

        return model

    def _init_gbm_catboost(self, **params: Any) -> CatBoost:
        model = CatBoost()  # Just to mitigate referenced before assignment warning

        reg_obj = [
            'MAE',
            'MAPE',
            'Poisson',
            'Quantile',
            'MultiQuantile',
            'RMSE',
            'RMSEWithUncertainty',
            'LogLinQuantile',
            'Lq',
            'Huber',
            'Expectile',
            'Tweedie',
            'LogCosh',
            'FairLoss',
            'NumErrors',
            'SMAPE',
            'R2',
            'MSLE',
            'MedianAbsoluteError',
            'MultiRMSE',
            'MultiRMSEWithMissingValues',
        ]
        class_obj = [
            'Logloss',
            'CrossEntropy',
            'Precision',
            'Recall',
            'F',
            'F1',
            'BalancedAccuracy',
            'BalancedErrorRate',
            'MCC',
            'Accuracy',
            'CtrFactor',
            'AUC',
            'QueryAUC',
            'NormalizedGini',
            'BrierScore',
            'HingeLoss',
            'HammingLoss',
            'ZeroOneLoss',
            'Kappa',
            'WKappa',
            'LogLikelihoodOfPrediction',
            'MultiClass',
            'MultiClassOneVsAll',
            'TotalF1',
            'MultiLogloss',
            'MultiCrossEntropy',
        ]
        rank_obj = [
            'PairLogit',
            'PairLogitPairwise',
            'PairAccuracy',
            'YetiRank',
            'YetiRankPairwise',
            'StochasticFilter',
            'StochasticRank',
            'QueryCrossEntropy',
            'QueryRMSE',
            'QuerySoftMax',
            'PFound',
            'NDCG',
            'DCG',
            'FilteredDCG',
            'AverageGain',
            'PrecisionAt',
            'RecallAt',
            'MAP',
            'ERR',
            'MRR',
        ]

        alias = {
            'regression': 'RMSE',
            'logit': 'Logloss',
            'binary': 'Logloss',
            'multiclass': 'MultiClass',
            'poisson': 'Poisson',
            'gamma': 'Tweedie:variance_power=1.9999999',
        }

        # Checks on supported objectives vs tasks
        if self.objective is None:
            if self.task == 'reg':
                objective: Union[str, Callable] = 'regression'
            else:
                objective = 'binary'
        else:
            objective = self.objective

        if isinstance(objective, str):
            objective = objective if objective not in alias.keys() else alias[objective]
            if objective in rank_obj:
                raise ValueError(f'Ranking objective "{objective}" is not supported.')
            elif (objective in reg_obj) or ('Tweedie:variance_power=' in objective):
                if self.task != 'reg':
                    raise ValueError(f'Objective "{objective}" does not match the task "{self.task}".')
            elif objective in class_obj:
                if self.task != 'class':
                    raise ValueError(f'Objective "{objective}" does not match the task "{self.task}".')
            else:
                raise ValueError(
                    f'Invalid objective "{objective}" supported objectives '
                    f'{[*list(alias.keys()), *reg_obj, *class_obj]}.'
                )

        if self.task == 'reg':
            model = CatBoostRegressor(objective=objective, n_estimators=self.n_estimators, **params)
        if self.task == 'class':
            model = CatBoostClassifier(objective=objective, n_estimators=self.n_estimators, **params)

        return model

    def init_model(self) -> Any:
        model = None
        params = self.metadata['init_params']['kwargs']
        if self.backend == 'xgboost':
            model = self._init_gbm_xgboost(**params)
        if self.backend == 'lightgbm':
            model = self._init_gbm_lightgbm(**params)
        if self.backend == 'catboost':
            model = self._init_gbm_catboost(**params)
        self._update_metadata()
        return model

    def fit(
        self,
        x: Union[DataFrame, Series],
        y: Union[DataFrame, Series],
        report: Union[None, List, Tuple, Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Fit a Gradient Boosting Machine.

        Args:
            x (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            report (list, tuple, optional): A list of metrics to report after model fitting, optional.
            **kwargs: Other parameters passed to Scikit-learn API .fit().
        """
        for arg in [x, y]:
            if (arg is not None) and (not isinstance(arg, (DataFrame, Series))):
                argname = [k for k, v in locals().items() if v == arg][0]
                raise TypeError(
                    f'Invalid type {type(arg)} for "{argname}". It must be either pd.DataFrame or pd.Series.'
                )

        if isinstance(y, DataFrame) and y.shape[1] > 1:
            argname = [k for k, v in locals().items() if v == y][0]
            raise ValueError(f'Argument "{argname}" must be a one-dimensional DataFrame.')

        features = list(x.columns) if isinstance(x, DataFrame) else [x.name]
        target = list(y.columns) if isinstance(y, DataFrame) else y.name
        self.metadata.update({'feature_names': features, 'target': target})

        self.model.fit(x, y, **kwargs)
        self.metadata.update({'is_fitted': True})
        if isinstance(report, (list, tuple)) or callable(report):
            prediction = self.model.predict(x)
            if callable(report):
                report_data = [[report.__name__, report(y, prediction)]]
            else:
                report_data = [[x.__name__, x(y, prediction)] for x in report]
            print(DataFrame(report_data, columns=['Metrics', 'Value']).set_index('Metrics'))

    def predict(self, x: Union[DataFrame, Series], **kwargs: Any) -> Optional[ndarray]:
        """Predict using GBM with feature matrix x.

        Args:
            x (pd.DataFrame, pd.Series): Samples.
            **kwargs: Other parameters passed to Scikit-learn API .predict().

        Returns:
            array: Returns predicted values.
        """
        if not self.metadata['is_fitted']:
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        if not isinstance(x, (DataFrame, Series)):
            raise TypeError(f'Invalid type {type(x)} for "x". It must be either pd.DataFrame or pd.Series.')

        return self.model.predict(x[self.metadata['feature_names']] if isinstance(x, DataFrame) else x, **kwargs)

    def shap(self, x: Union[DataFrame, Series], show: bool = False) -> Dict[str, float64]:
        """Method for shap values calculation and corresponding plot of feature importances.

        Args:
            x (pd.DataFrame, pd.Series): Data for shap values calculation.
            show (boolean, optional): Whether to plot a graph.

        Returns:
            JSON containing shap values.
        """
        if not self.metadata['is_fitted']:
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        if not isinstance(x, (DataFrame, Series)):
            raise TypeError(f'Invalid type {type(x)} for "x". It must be either pd.DataFrame or pd.Series.')

        shap_values: ndarray = ndarray((0,))
        if self.backend == 'lightgbm':
            shap_values = self.model.predict(x, pred_contrib=True)
        if self.backend == 'xgboost':
            shap_values = self.model._Booster.predict(DMatrix(x), pred_contribs=True)
        if self.backend == 'catboost':
            shap_values = self.model.get_feature_importance(Pool(x), type=EFstrType.ShapValues)

        imps = mean(npabs(shap_values), axis=0)[:-1]
        order = argsort(imps, axis=-1)
        sorted_features_names = array(self.metadata['feature_names'])[order]
        imps = imps[order]

        if show:
            fig = Figure(Bar(x=imps, y=sorted_features_names, orientation='h'))
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(title_text="Mean(|SHAP value|) (average impact on model output magnitude)"),
            )
            fig.update_yaxes(automargin=True)
            fig.show()

        return dict(zip(sorted_features_names[::-1], imps[::-1]))
