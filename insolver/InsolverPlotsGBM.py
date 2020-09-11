import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


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

    def plot_shap_waterfall(self, instance_index=0, link=None, height=700, title=None):
        """Plotting waterfall graph for individual observation using plotly package.

        Args:
            instance_index (int): Index of the observation in DataFrame passed to SHAP values calculation.
            link (:obj:`str`, optional): Whether to use link function for plotting waterfall graph.
                Used when SHAP plots values before link function. Takes arguments 'exp' and 'logit' (default=None).
            height (:obj:`int`, optional): Height of the plotly waterfall graph (default=700).
            title (:obj:`str`, optional): Title for plotly waterfall graph (default=None).
        """

        def logit(x):
            return np.true_divide(1, np.add(1, np.exp(x)))

        if isinstance(self.shap_values, list) and (len(self.shap_values) == 2):
            shap_values = self.shap_values[0]
            expected_value = self.explainer.expected_value[0]
        else:
            shap_values = self.shap_values
            expected_value = self.explainer.expected_value

        prediction = pd.DataFrame([expected_value] + shap_values[instance_index, :].tolist(),
                                  index=['Intercept'] + self.feature_names, columns=['SHAP Values'])
        prediction['CumSum'] = np.cumsum(prediction['SHAP Values'])
        if link == 'exp':
            prediction['Exp'] = np.exp(prediction['CumSum'])
            prediction['Contribution'] = [np.exp(expected_value)] + list(np.diff(prediction['Exp']))
        elif link == 'logit':
            prediction['Logit'] = logit(prediction['CumSum'])
            prediction['Contribution'] = [logit(expected_value)] + list(np.diff(prediction['Logit']))
        else:
            prediction['Contribution'] = [expected_value] + list(np.diff(prediction['CumSum']))

        fig = go.Figure(go.Waterfall(name=f'Prediction {instance_index}', orientation='h',
                                     measure=['relative'] * len(prediction),
                                     y=[(prediction.index[i] if i == 0
                                         else '{}={}'.format(prediction.index[i],
                                                             self.data.loc[instance_index,
                                                                           self.feature_names][i-1])) for i
                                        in range(len(prediction.index))],
                                     x=prediction['Contribution']))
        fig.update_layout(title=title, height=height)
        fig.show()


class PredictionPlots(object):
    """Class for plotting some graphs using predictions of the models.

    Attributes:
        df_predictions (pd.DataFrame): DataFrame containing all predictions needed to be visualized.
        df_targets (pd.DataFrame): DataFrame containing all targets corresponding to df_predictions.
        target_list (list): List of target names from df_targets for every prediction in df_predictions.
    """
    # TODO: Enable configuring custom visualization parameters?
    def __init__(self, df_predictions, df_targets, target_list):
        self.predictions = df_predictions
        self.targets = df_targets
        self.target_list = target_list

    def plot_metric_function(self, function, plotly=False):
        """Plotting scatter plot of metrics values of individual predictions vs. metrics.

        Args:
            function (function): Function to calculate metrics for individual observations.
                The function must take two arguments: array of prediction values and array of target values.
            plotly (:obj:`bool`, optional): Whether to plot graph with plotly or matplotlib (default=False).
        """
        metric_name = function.__name__
        metric = pd.DataFrame()
        for i in range(len(self.predictions.columns)):
            pred_name = self.predictions.columns[i]
            metric[f'{metric_name}_{pred_name}'] = function(self.predictions[pred_name],
                                                            self.targets[self.target_list[i]])

        if plotly:
            fig = go.Figure()
            for i in range(len(self.predictions.columns)):
                fig.add_trace(go.Scatter(x=self.predictions.iloc[:, i].values,
                                         y=metric.iloc[:, i].values,
                                         name=self.predictions.columns[i],
                                         mode='markers'))
            fig.update_xaxes(title_text='Prediction')
            fig.update_yaxes(title_text=metric_name.capitalize())
            fig.show()
        else:
            scatters, names = [], []
            for i in range(len(self.predictions.columns)):
                scatters.append(plt.scatter(self.predictions.iloc[:, i].values,
                                            metric.iloc[:, i].values))
                names.append(self.predictions.columns[i])
            plt.legend(scatters, names)
            plt.xlabel('Prediction')
            plt.ylabel(metric_name.capitalize())
            plt.show()

    def plot_distribution_plots(self, plot_target=False):
        """Plotting kernel density plots for prediction distributions.

        Args:
            plot_target (:obj:`bool`, optional): Whether to plot actual target KDE plot (default=False).
        """
        for pred in self.predictions.columns:
            sns.kdeplot(self.predictions[pred])
        if plot_target:
            if len(set(self.target_list)) == 1:
                sns.kdeplot(self.targets[self.target_list[0]])
        plt.show()
