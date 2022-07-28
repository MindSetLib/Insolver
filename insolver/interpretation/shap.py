import pandas as pd
import numpy as np
from plotly.graph_objects import Figure, Waterfall
from shap import TreeExplainer, LinearExplainer, summary_plot
from insolver.wrappers import InsolverGBMWrapper, InsolverBaseWrapper, InsolverGLMWrapper
from .base import InterpretBase


class SHAPExplanation(InterpretBase):
    """
    SHapley Additive exPlanations (SHAP). Uses shap package.

    Parameters:
        estimator: A fitted estimator object implementing `predict` or `predict_proba`.
        estimator_type (str): Type of the estimator, supported values are `tree` and `linear`.
        data (pandas.Dataframe): Data for creating LinearExplainer.
    """

    def __init__(self, estimator, estimator_type='tree', data=None):
        self.estimator = estimator.model if isinstance(estimator, InsolverBaseWrapper) else estimator
        self.estimator = estimator.model['glm'] if isinstance(estimator, InsolverGLMWrapper) else self.estimator
        self.estimator_type = estimator_type

        if self.estimator_type not in ['tree', 'linear']:
            raise NotImplementedError(
                f'''
            estimator_type {estimator_type} is not supported. Supported values are "tree" and "linear".'''
            )

        self.explainer = (
            TreeExplainer(self.estimator)
            if self.estimator_type == 'tree'
            else LinearExplainer(self.estimator, masker=data)
        )

    def get_features_shap(self, X, show=False, plot_type='bar'):
        """Method for shap values calculation and corresponding plot of feature importances.

        Parameters:
            X (pd.DataFrame, pd.Series): Data for shap values calculation.
            show (boolean, optional): Whether to plot a graph.
            plot_type (str, optional): Type of feature importance graph, takes value in ['dot', 'bar'].

        Raises:
            TypeError: If the type of X is not supported.
        """
        # check type of X
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise TypeError(f'Type {type(X)} is not supported. Supported types are pandas.Dataframe and pandas.Series.')
        X = pd.DataFrame(X).T if isinstance(X, pd.Series) else X
        # get shap values
        shap_values = self.explainer.shap_values(X)
        shap_values = shap_values[0] if isinstance(shap_values, list) and (len(shap_values) == 2) else shap_values
        variables = list(X.columns)
        mean_shap = np.abs(shap_values).mean(axis=0).tolist()
        # show plot if True
        if show:
            summary_plot(shap_values, X, plot_type=plot_type, feature_names=variables)

        return {variables[i]: round(mean_shap[i], 4) for i in range(len(variables))}

    def show_explanation(self, instance, link=None, show=True):
        """Show explanation.

        Parameters:
            instance (pd.Series, np.ndarray): Data for shap values calculation.
            link (callable, optional): A function for transforming shap values into predictions.
            show (boolean, optional): Whether to plot a graph or return a json.

        Raises:
            TypeError: If the instance type is not supported.
        """
        # transformation function
        def logit(x):
            return np.true_divide(1, np.add(1, np.exp(x)))

        # check type of instance
        if not isinstance(instance, np.ndarray) and not isinstance(instance, pd.Series):
            raise TypeError(
                f'''
            Type {type(instance)} is not supported. Supported types are numpy.ndarray and pandas.Series.'''
            )
        # get feature_names OR check shape and create features names
        if isinstance(instance, pd.Series):
            feature_names = list(instance.index)
        else:
            if len(instance.shape) > 1:
                raise NotImplementedError('Only (*,) shape is supported.')
            feature_names = []
            for f in range(instance.shape[0]):
                feature_names.append(f'Feature {f}')
        # get shap values
        shap_values = self.explainer.shap_values(instance)
        cond_bool = isinstance(shap_values, list) and (len(shap_values) == 2)
        shap_values = shap_values[0] if cond_bool else shap_values
        expected_value = (
            self.explainer.expected_value[0]
            if isinstance(self.explainer.expected_value, np.ndarray)
            else self.explainer.expected_value
        )
        print(self.explainer.expected_value)
        # create predictions Dataframe
        prediction = pd.DataFrame(
            [expected_value] + shap_values.reshape(-1).tolist(),
            index=['E[f(x)]'] + feature_names,
            columns=['SHAP Value'],
        )
        prediction['CumSum'] = np.cumsum(prediction['SHAP Value'])
        prediction['Value'] = np.append(np.nan, instance)
        # transform result if objective or link
        objective = self.estimator.objective if isinstance(self.estimator, InsolverGBMWrapper) else None
        if (objective is not None) and (link is None):
            link = np.exp if objective in ['poisson', 'gamma'] else logit if objective == 'binary' else None
        if link is not None:
            prediction['Link'] = link(prediction['CumSum'])
            prediction['Contribution'] = [link(expected_value)] + list(np.diff(prediction['Link']))
        else:
            prediction['Contribution'] = list(prediction['SHAP Value'])
        # save instance and prediction for plotting result
        self.instance = instance
        self.prediction = prediction

        return self.prediction

    def plot(self, figsize=None, **kwargs):
        """
        Plot waterfall chart using values created in the `show_explanation` method.

        Raises:
            Exception: If plot() is called before show_explanation().
        """
        try:
            fig = Figure(
                Waterfall(
                    name='Prediction',
                    orientation='h',
                    measure=['relative'] * len(self.prediction),
                    y=[
                        self.prediction.index[i]
                        if i == 0
                        else f'{self.prediction.index[i]} = {round(self.instance[i-1], 4)}'
                        for i in range(len(self.prediction.index))
                    ],
                    x=self.prediction['Contribution'],
                )
            )
            fig.update_layout(kwargs)
            fig.show()
        except AttributeError:
            raise AttributeError('First call the "show_explanation()" method to create the prediction!')

    def get_model(self):
        return self.explainer
