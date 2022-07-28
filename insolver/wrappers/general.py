from numpy import concatenate, array
from pandas import DataFrame

from sklearn.metrics import mean_squared_error, SCORERS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import InsolverBaseWrapper
from .extensions import InsolverCVHPExtension, InsolverPDPExtension


class InsolverRFWrapper(InsolverBaseWrapper, InsolverCVHPExtension, InsolverPDPExtension):
    """Insolver wrapper for Random Forest.

    Parameters:
        backend (str): Framework for building RF, 'sklearn' is supported.
        task (str): Task that RF should solve: Classification or Regression. Values 'reg' and 'class' are supported.
        n_estimators (int, optional): Number of trees in the forest. Equals 100 by default.
        load_path (str, optional): Path to RF model to load from disk.
        **kwargs: Parameters for RF estimators except `n_estimators`. Will not be changed in hyperopt.
    """

    def __init__(self, backend, task=None, n_estimators=100, load_path=None, **kwargs):
        super(InsolverRFWrapper, self).__init__(backend)
        self.init_args = self._get_init_args(vars())
        self.algo, self._backends = 'rf', ['sklearn']
        self._tasks = ['class', 'reg']
        self._back_load_dict = {'sklearn': self._pickle_load}
        self._back_save_dict = {'sklearn': self._pickle_save}
        self.n_estimators, self.params = n_estimators, None

        if backend not in self._backends:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

        if load_path is not None:
            self.load_model(load_path)
        else:
            if task in self._tasks:
                rf_init = {'class': {'sklearn': RandomForestClassifier}, 'reg': {'sklearn': RandomForestRegressor}}

                kwargs.update({'n_estimators': self.n_estimators})
                self.model, self.params = rf_init[task][self.backend](**(kwargs if kwargs is not None else {})), kwargs

                def __params_rf(**params):
                    params.update(self.params)
                    return rf_init[task][self.backend](**params)

                self.object = __params_rf
            else:
                raise NotImplementedError(f'Task parameter supports values in {self._tasks}.')
        self._update_meta()

    def fit(self, X, y, report=None, **kwargs):
        """Fit a Random Forest.

        Args:
            X (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            report (list, tuple, optional): A list of metrics to report after model fitting, optional.
            **kwargs: Other parameters passed to Scikit-learn API .fit().
        """
        self.model.fit(X, y, **kwargs)
        if not hasattr(self.model, 'feature_name_'):
            self.model.feature_name_ = X.columns if isinstance(X, DataFrame) else [X.name]
        self._update_meta()
        if report is not None:
            if isinstance(report, (list, tuple)):
                prediction = self.model.predict(X)
                print(
                    DataFrame([[x.__name__, x(y, prediction)] for x in report])
                    .rename({0: 'Metrics', 1: 'Value'}, axis=1)
                    .set_index('Metrics')
                )

    def predict(self, X, **kwargs):
        """Predict using RF with feature matrix X.

        Args:
            X (pd.DataFrame, pd.Series): Samples.
            **kwargs: Other parameters passed to Scikit-learn API .predict().

        Returns:
            array: Returns predicted values.
        """
        return self.model.predict(
            X if not hasattr(self.model, 'feature_name_') else X[self.model.feature_name_], **kwargs
        )

    def cross_val(self, X, y, scoring=None, cv=None, **kwargs):
        """Method for performing cross-validation given the hyperparameters of initialized or fitted model.

        Args:
            X (pd.DataFrame, pd.Series): Training data.
            y (pd.DataFrame, pd.Series): Training target values.
            scoring (callable): Metrics passed to sklearn.model_selection.cross_validate calculation.
            cv (int, iterable, cross-validation generator, optional): Cross-validation strategy from
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
            scorers = {
                scorer.__name__.replace('_', ' '): array([scorer(y, self.model.predict(X))]) for scorer in scoring
            }
        elif isinstance(scoring, str):
            if scoring in SCORERS:
                scorers = {scoring.replace('_', ' '): array([SCORERS[scoring](self.model, X=X, y=y)])}
            else:
                raise ValueError(f'Scorer {scoring} is not supported.')
        else:
            raise NotImplementedError(f'Scoring of type {scoring} is not supported')
        metrics = DataFrame({key: concatenate((scorers[key], metrics[key])) for key in scorers.keys()}).T
        metrics.columns = [f'Fold {i}' if i != 0 else 'Overall' for i in range(metrics.shape[1])]
        return metrics
