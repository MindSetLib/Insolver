import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go


class ShapPlots(object):
    def __init__(self, model, model_feature_names, data):
        self.model = model
        self.data = data
        self.feature_names = model_feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(data)

    def plot_shap_importances(self, plot_type="dot"):
        shap.summary_plot(self.shap_values, self.data, plot_type=plot_type)

    def plot_shap_waterfall(self, instance_index=0, exponential=False, height=700, title=None):
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
