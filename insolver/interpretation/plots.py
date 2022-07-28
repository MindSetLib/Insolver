import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from alibi.explainers import ALE, plot_ale
from .base import InterpretBase

warnings.filterwarnings('ignore')


class ExplanationPlot(InterpretBase):
    """
    Class for creating a plot for interpretation.
    Partial Dependence Plot (PDP), Individual Condition Expectation (ICE), Accumulated Local Effects (ALE) are
    supported.

    Parameters:
        method (str): Plot creation method. Values `pdp`, `ice`, `ale`, `ale_aleplot` are supported.
        x (pandas.Dataframe, optional): Data to plot.
        estimator: A fitted estimator object implementing `predict`,`predict_proba`, or `decision_function`.
        features (list): Indices of features for a given plot. A tuple of one integer will plot a partial dependence
         curve of one feature. A tuple of two integers will plot a two-way partial dependence curve as a contour plot.
        target_names (list): A list of target names for the `ALE` method.

    """

    def __init__(self, method, x, estimator, features, target_names=None):
        self.method = method
        self.x = x
        self.estimator = estimator
        self.features = features
        self.target_names = target_names

    def plot(self, figsize=(10, 10), **kwargs):
        """
        Create plot.

        Parameters:
            figsize (tuple): Figsize of the plot.
            **kwargs: Arguments for the selected function.

        Raises:
            NotImplementedError: If method is not supported.
        """

        # initialize all methods
        self._init_methods_dict()

        # raise error if method is not supported
        if self.method:
            if self.method not in self.methods_dict.keys():
                raise NotImplementedError(f'Method {self.method} is not supported.')

        # get function and call it
        method_func = self.methods_dict[self.method]
        method_func(figsize, **kwargs)

    def show_explanation(self, instance):
        self.plot()

    def get_model(self):
        """
        Get model.
        """
        return self.model

    def _plot_pdp(self, figsize, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)

        self.model = PartialDependenceDisplay.from_estimator(
            kind='average', estimator=self.estimator, X=self.x, features=self.features, ax=ax, **kwargs
        )

    def _plot_ice(self, figsize, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)

        self.model = PartialDependenceDisplay.from_estimator(
            kind='both', estimator=self.estimator, X=self.x, features=self.features, ax=ax, **kwargs
        )

    def _plot_ale(self, figsize, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)

        self.model = ALE(predictor=self.estimator.predict, feature_names=self.features, target_names=self.target_names)

        # convert x to numpy if x is pandas.DataFrame
        if isinstance(self.x, pd.DataFrame):
            X = self.x.to_numpy()
        else:
            X = self.x

        explanation = self.model.explain(X=X)
        plot_ale(exp=explanation, ax=ax, **kwargs)

    def _init_methods_dict(self):
        self.methods_dict = {
            'pdp': self._plot_pdp,
            'ice': self._plot_ice,
            'ale': self._plot_ale,
        }
