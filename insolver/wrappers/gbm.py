from base64 import b64encode

from numpy import cumsum, diff, exp, true_divide, add, append, nan, concatenate, array
from pandas import DataFrame, Series

from sklearn.metrics import mean_squared_error, SCORERS
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from shap import TreeExplainer, summary_plot

from plotly.graph_objects import Figure, Waterfall
from plotly.io import to_image

from .base import InsolverBaseWrapper
from .extensions import InsolverCVHPExtension, InsolverPDPExtension


class InsolverGBMWrapper(InsolverBaseWrapper, InsolverCVHPExtension, InsolverPDPExtension):
    """Insolver wrapper for Gradient Boosting Machines.

    Attributes:
        backend (str): Framework for building GBM, 'xgboost', 'lightgbm' and 'catboost' are supported.
        task (str): Task that GBM should solve: Classification or Regression. Values 'reg' and 'class' are supported.
        n_estimators (:obj:`int`, optional): Number of boosting rounds. Equals 100 by default.
        objective (:obj:`str` or :obj:`callable`): Objective function for GBM to optimize.
        load_path (:obj:`str`, optional): Path to GBM model to load from disk.
        **kwargs: Parameters for GBM estimators except `n_estimators` and `objective`. Will not be changed in hyperopt.
    """
    def __init__(self, backend, task=None, objective=None, n_estimators=100, load_path=None, **kwargs):
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
        if not hasattr(self.model, 'feature_name_'):
            self.model.feature_name_ = X.columns if isinstance(X, DataFrame) else [X.name]

    def predict(self, X, **kwargs):
        """Predict using GBM with feature matrix X.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Samples.
            **kwargs: Other parameters passed to Scikit-learn API .predict().

        Returns:
            array: Returns predicted values.
        """
        return self.model.predict(X if not hasattr(self.model, 'feature_name_')
                                  else X[self.model.feature_name_], **kwargs)

    def shap(self, X, plot=False, plot_type='bar'):
        """Method for shap values calculation and corresponding plot of feature importances.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Data for shap values calculation.
            plot (:obj:`boolean`, optional): Whether to plot a graph.
            plot_type (:obj:`str`, optional): Type of feature importance graph, takes value in ['dot', 'bar'].

        Returns:
            JSON containing shap values.
        """
        explainer = TreeExplainer(self.model)
        X = DataFrame(X).T if isinstance(X, Series) else X
        shap_values = explainer.shap_values(X)

        shap_values = shap_values[0] if isinstance(shap_values, list) and (len(shap_values) == 2) else shap_values
        expected_value = (explainer.expected_value[0].tolist()
                          if isinstance(shap_values, list) and (len(shap_values) == 2) else [explainer.expected_value])
        variables = ['Intercept'] + list(X.columns)
        mean_shap = expected_value + shap_values.mean(axis=0).tolist()

        if plot:
            summary_plot(shap_values, X, plot_type=plot_type)
        return {variables[i]: mean_shap[i] for i in range(len(variables))}

    def shap_explain(self, data, index=None, link=None, show=True, layout_dict=None):
        """Method for plotting a waterfall graph or return corresponding JSON if show=False.

        Args:
            data (:obj:`pd.DataFrame`, :obj:`pd.Series`): Data for shap values calculation.
            index (:obj:`int`, optional): Index of the observation of interest, if data is pd.DataFrame.
            link (:obj:`callable`, optional): A function for transforming shap values into predictions.
            Unnecessary if self.objective is present and it takes values in ['binary', 'poisson', 'gamma'].
            show (:obj:`boolean`, optional): Whether to plot a graph or return a json.
            layout_dict (:obj:`boolean`, optional): Dictionary containing the parameters of plotly figure layout.

        Returns:
            None or dict: Waterfall graph or corresponding JSON.
        """

        def logit(x):
            return true_divide(1, add(1, exp(-x)))

        explainer = TreeExplainer(self.model)
        if isinstance(self.model, (XGBClassifier, XGBRegressor)):
            feature_names = self.model.get_booster().feature_names
        elif isinstance(self.model, (LGBMClassifier, LGBMRegressor)):
            feature_names = self.model.feature_name_
        elif isinstance(self.model, (CatBoostClassifier, CatBoostRegressor)):
            feature_names = self.model.feature_names_
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

        index = index if (isinstance(data, DataFrame)) and (index is not None) else None
        data = DataFrame(data).T[feature_names] if isinstance(data, Series) else data[feature_names]
        data = data if index is None else data.loc[[index], :]
        shap_values = explainer.shap_values(data)
        cond_bool = isinstance(shap_values, list) and (len(shap_values) == 2)
        shap_values = shap_values[0] if cond_bool else shap_values
        expected_value = explainer.expected_value[0] if cond_bool else explainer.expected_value

        prediction = DataFrame([expected_value] + shap_values.reshape(-1).tolist(), index=['Intercept'] + feature_names,
                               columns=['SHAP Value'])
        prediction['CumSum'] = cumsum(prediction['SHAP Value'])
        prediction['Value'] = append(nan, data.values.reshape(-1))

        if (self.objective is not None) and (link is None):
            link = exp if self.objective in ['poisson', 'gamma'] else logit if self.objective == 'binary' else None
        if link is not None:
            prediction['Link'] = link(prediction['CumSum'])
            prediction['Contribution'] = [link(expected_value)] + list(diff(prediction['Link']))
        else:
            prediction['Contribution'] = [expected_value] + list(diff(prediction['CumSum']))

        fig = Figure(Waterfall(name=f'Prediction {index}',
                               orientation='h',
                               measure=['relative'] * len(prediction),
                               y=[prediction.index[i] if i == 0
                                  else f'{prediction.index[i]}={data.values.reshape(-1)[i-1]}' for i
                                  in range(len(prediction.index))],
                               x=prediction['Contribution']))
        fig.update_layout(**(layout_dict if layout_dict is not None else {}))

        if show:
            fig.show()
        else:
            json_ = prediction[['Value', 'SHAP Value', 'Contribution']].T.to_dict()
            fig_base64 = b64encode(to_image(fig, format='jpeg', engine='kaleido')).decode('ascii')
            json_.update({'id': int(data.index.values), 'predict': prediction['Link'][-1],
                          "ShapValuesPlot": fig_base64})
            return json_

    def cross_val(self, X, y, scoring=None, cv=None, **kwargs):
        """Method for performing cross-validation given the hyperparameters of initialized or fitted model.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training data.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Training target values.
            scoring (:obj:`callable`): Metrics passed to sklearn.model_selection.cross_validate calculation.
            cv (:obj:`int, cross-validation generator or an iterable`, optional): Cross-validation strategy from
             sklearn. Performs 5-fold cv by default.
            **kwargs: Other parameters passed to sklearn.model_selection.cross_validate.

        Returns:
            pd.DataFrame, pd.DataFrame: DataFrame with metrics on folds, DataFrame with shap values on folds.
        """
        scoring = mean_squared_error if scoring is None else scoring
        models, metrics = self._cross_val(X, y, scoring=scoring, cv=cv, **kwargs)
        if callable(scoring):
            scorers = {scoring.__name__.replace('_', ' '): array([scoring(y, self.model.predict(X))])}
        elif isinstance(scoring, (tuple, list)):
            scorers = {scorer.__name__.replace('_', ' '): array([scorer(y, self.model.predict(X))]) for
                       scorer in scoring}
        elif isinstance(scoring, str):
            if scoring in SCORERS:
                scorers = {scoring.replace('_', ' '): array([SCORERS[scoring](self.model, X=X, y=y)])}
            else:
                raise ValueError(f'Scorer {scoring} is not supported.')
        else:
            raise NotImplementedError(f'Scoring of type {scoring} is not supported')
        metrics = DataFrame({key: concatenate((scorers[key], metrics[key])) for key in scorers.keys()}).T
        metrics.columns = [f'Fold {i}' if i != 0 else 'Overall' for i in range(metrics.shape[1])]
        shap_coefs = []
        explainer = TreeExplainer(self.model)

        shap_coefs.append(([explainer.expected_value] if explainer.expected_value is None
                           else explainer.expected_value.tolist()) + explainer.shap_values(X).mean(axis=0).tolist())
        for model in models:
            explainer = TreeExplainer(model)
            shap_coefs.append(([explainer.expected_value] if explainer.expected_value is None
                               else explainer.expected_value.tolist()) + explainer.shap_values(X).mean(axis=0).tolist())
        shapdf = DataFrame(array(shap_coefs).T, columns=['Overall'] + [f'Fold {x}' for x in range(1, len(models) + 1)],
                           index=['Intercept'] + X.columns.tolist())
        return metrics, shapdf
