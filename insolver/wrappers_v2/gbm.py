import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import Optional, Any, Callable, Union, List, Tuple, Dict

from numpy import ndarray, array, abs as npabs, mean, argsort, float64, cumsum, append, diff
from pandas import DataFrame, Series

from xgboost import XGBClassifier, XGBRegressor, XGBModel, DMatrix
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost, Pool, EFstrType

from plotly.graph_objects import Figure, Bar, Waterfall

from .base import InsolverBaseWrapper
from .utils import save_pickle
from .utils.hypertoptcv import hyperopt_cv_proc, tpe, rand, AUTO_SPACE_CONFIG


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
        task: Literal['class', 'reg'] = 'reg',
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
        self.best_params: Optional[Dict[str, Any]] = None
        self.trials = None
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

        def_params = dict(verbose=0, allow_writing_files=False)
        for key, val in def_params.items():
            if key not in params.keys():
                params.update({key: val})

        if self.task == 'reg':
            model = CatBoostRegressor(objective=objective, n_estimators=self.n_estimators, **params)
        if self.task == 'class':
            model = CatBoostClassifier(objective=objective, n_estimators=self.n_estimators, **params)

        return model

    def init_model(self, additional_params: Optional[Dict] = None) -> Any:
        model = None
        params = self.metadata['init_params']['kwargs']
        if additional_params is not None:
            params.update(additional_params)
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

    def _calc_shap_values(self, x: Union[DataFrame, Series]) -> ndarray:
        if not self.metadata['is_fitted']:
            raise ValueError("This instance is not fitted yet. Call '.fit(...)' before using this estimator.")

        if not isinstance(x, (DataFrame, Series)):
            raise TypeError(f'Invalid type {type(x)} for "x". It must be either pd.DataFrame or pd.Series.')

        feature_names = self.metadata['feature_names']
        x = DataFrame(x).T[feature_names] if isinstance(x, Series) else x[feature_names]
        shap_values: ndarray = ndarray((0,))
        if self.backend == 'lightgbm':
            shap_values = self.model.predict(x, pred_contrib=True)
        if self.backend == 'xgboost':
            shap_values = self.model._Booster.predict(DMatrix(x), pred_contribs=True)
        if self.backend == 'catboost':
            shap_values = self.model.get_feature_importance(Pool(x), type=EFstrType.ShapValues)
        return shap_values

    def shap(self, x: Union[DataFrame, Series], show: bool = True) -> Optional[Dict[str, float64]]:
        """Method for SHAP feature importance estimation.

        Args:
            x (pd.DataFrame, pd.Series): Data for SHAP feature importance estimation.
            show (boolean, optional): Whether to plot a graph (default: show=True).

        Returns:
            Dict[str, float64] containing SHAP feature importances.
        """
        # Currently does not support multiclass
        shap_values = self._calc_shap_values(x)
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
            return None
        else:
            return dict(zip(sorted_features_names[::-1], imps[::-1]))

    def shap_explain(
        self, data: Union[DataFrame, Series], show: bool = True, layout_dict: Dict[str, Any] = None
    ) -> Optional[Dict[str, Dict[str, float64]]]:
        """Method for plotting a waterfall with feature contributions or returning a dict with feature contributions.

        Args:
            data (pd.DataFrame, pd.Series): One-dimensional data sample for shap feature contribution calculation.
            show (boolean, optional): Whether to plot a graph or return a json.
            layout_dict (boolean, optional): Dictionary containing the parameters of plotly figure layout.

        Returns:
            None or dict: Waterfall graph or corresponding dict.
        """
        # Currently does not support multiclass. Also, lacks link function. Different results with SHAP.

        if isinstance(data, DataFrame) and (data.shape[0] != 1):
            raise ValueError('Argument "data" must be a one-dimensional data sample.')

        shap_values = self._calc_shap_values(data)[0]
        factors, base = shap_values[:-1], shap_values[-1]
        order = argsort(-npabs(factors))[::-1]
        feature_names = array(self.metadata['feature_names'])[order]
        value = data.values.reshape(-1)[order]

        cumsum_ = cumsum(append(base, factors[order]))
        contribution = diff(cumsum_)

        mask_ = contribution != 0
        feature_names = feature_names[mask_]
        value = value[mask_]
        contribution = contribution[mask_]

        if show:
            fig = Figure(
                Waterfall(
                    orientation='h',
                    measure=['relative'] * contribution.shape[0],
                    base=base,
                    y=[f'{feature_names[i]} = {value[i]}' for i in range(len(feature_names))],
                    x=contribution,
                )
            )
            fig.add_vline(
                x=base, annotation_text='E[f(x)]', annotation_position="top left", line_width=0.2, line_dash="dot"
            )
            fig.add_vline(
                x=cumsum_[-1],
                annotation_text='f(x)',
                annotation_position="bottom right",
                line_width=0.2,
                line_dash="dot",
            )
            fig.update_layout(**(layout_dict if layout_dict is not None else {'margin': dict(l=0, r=0, t=0, b=0)}))
            fig.update_yaxes(automargin=True)
            fig.show()
            return None
        else:
            explain = {f: {'value': v, 'contribution': c} for f, v, c in zip(feature_names, value, contribution)}
            explain.update({'E[f(x)]': {'value': base, 'contribution': base}})
            return explain

    def hyperopt_cv(
        self,
        x: Union[DataFrame, Series],
        y: Union[DataFrame, Series],
        params: Dict[str, Any],
        fn: Callable = None,
        algo: Union[None, rand.suggest, tpe.suggest] = None,
        max_evals: int = 10,
        timeout: Optional[int] = None,
        fmin_params: Dict[str, Any] = None,
        fn_params: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Hyperparameter optimization using hyperopt. Using cross-validation to evaluate hyperparameters by default.

        Args:
            x (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            params (dict): Dictionary of hyperparameters passed to hyperopt.
            fn (callable, optional): Objective function to optimize with hyperopt.
            algo (callable, optional): Algorithm for hyperopt. Available choices are: hyperopt.tpe.suggest and
             hyperopt.random.suggest. Using hyperopt.tpe.suggest by default.
            max_evals (int, optional): Number of function evaluations before returning.
            timeout (None, int, optional): Limits search time by parametrized number of seconds.
             If None, then the search process has no time constraint. None by default.
            fmin_params (dict, optional): Dictionary of supplementary arguments for hyperopt.fmin function.
            fn_params (dict, optional):  Dictionary of supplementary arguments for custom fn objective function.

        Returns:
            dict: Dictionary of the best choice of hyperparameters. Also, best model is fitted.
        """
        self.best_params, self.trials = hyperopt_cv_proc(
            self, x, y, params, fn, algo, max_evals, timeout, fmin_params, fn_params
        )
        self._update_metadata()
        self.model = self.init_model(self.best_params)
        self.fit(
            x, y, **({} if not ((fn_params is not None) and ("fit_params" in fn_params)) else fn_params["fit_params"])
        )
        return self.best_params

    def auto_hyperopt_cv(
        self,
        x: Union[DataFrame, Series],
        y: Union[DataFrame, Series],
        metric: Callable,
        offset: str = None,
        max_evals: int = 15,
        selection: Optional[str] = None,
        selection_thresh: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """Hyperparameter optimization using hyperopt. Using cross-validation to evaluate hyperparameters by default.

        Args:
            x (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            metric: Callable,
            offset (str, optional): Column name of the offset column.
            max_evals (int, optional): Number of function evaluations before returning.
            selection (str, optional): Feature selection method. Currently only 'shap' method is supported.
            selection_thresh (float, optional): Threshold for feature selection for 'shap' method. Default 0.05.

        Returns:
            dict: Dictionary of the best choice of hyperparameters. Also, best model is fitted.
        """
        if self.backend in AUTO_SPACE_CONFIG.keys():
            params: Dict[str, Any] = AUTO_SPACE_CONFIG[self.backend]
            self.best_params = self.hyperopt_cv(
                x,
                y,
                params,
                max_evals=max_evals,
                fn_params={'scoring': metric, 'fit_params': {'sample_weight': offset}},
            )

            if selection == 'shap':
                shaps: Union[Dict, Series, DataFrame] = self.shap(x, show=False)
                shaps = DataFrame.from_dict({'shap': shaps}).abs().sort_values('shap', ascending=False)
                shaps = shaps / shaps.sum()
                columns = shaps[shaps['shap'] >= selection_thresh].index.tolist()
                self.best_params = self.hyperopt_cv(
                    x[columns],
                    y,
                    params,
                    max_evals=max_evals,
                    fn_params={'scoring': metric, 'fit_params': {'sample_weight': offset}},
                )
            return self.best_params
        else:
            return None
