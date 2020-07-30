import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go


class ShapPlots(object):
    """Class for plotting some graphs from SHAP package.

    Attributes:
        model: Model object (XGBoost, LightGBM, Catboost boosters) supported by SHAP package.
        model_feature_names (array_like): Array containing feature names.
        data (pd.DataFrame): DataFrame for SHAP values calculation.
    """
    def __init__(self, model, model_feature_names, data):
        self.model = model
        self.data = data
        self.feature_names = list(model_feature_names)
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(data)

    def plot_shap_importance(self, plot_type="bar"):
        """Plotting summary graph from SHAP package.

        Args:
            plot_type (:obj:`str`, optional): String containing name of SHAP summary plot type (default="bar").
        """
        shap.summary_plot(self.shap_values, self.data, plot_type=plot_type)

    def plot_shap_waterfall(self, instance_index=0, exponential=False, height=700, title=None):
        """Plotting waterfall graph for individual observation using plotly package.

        Args:
            instance_index (int): Index of the observation in DataFrame passed to SHAP values calculation.
            exponential (:obj:`bool`, optional): Whether to use exponential for plotting waterfall graph.
                Used when SHAP plots values before log-link (default=False).
            height (:obj:`int`, optional): Height of the plotly waterfall graph (default=700).
            title (:obj:`str`, optional): Title for plotly waterfall graph (default=None).
        """
        prediction = pd.DataFrame([self.explainer.expected_value] + self.shap_values[instance_index, :].tolist(),
                                  index=['Intercept'] + self.feature_names, columns=['SHAP Values'])
        prediction['CumSum'] = np.cumsum(prediction['SHAP Values'])
        if exponential:
            prediction['Exp'] = np.exp(prediction['CumSum'])
            prediction['Contribution'] = [np.exp(self.explainer.expected_value)] + list(np.diff(prediction['Exp']))
        else:
            prediction['Contribution'] = [self.explainer.expected_value] + list(np.diff(prediction['CumSum']))

        fig = go.Figure(go.Waterfall(name=f'Prediction {instance_index}', orientation='h',
                                     measure=['relative'] * len(prediction), y=prediction.index,
                                     x=prediction['Contribution']))
        fig.update_layout(title=title, height=height)
        fig.show()
