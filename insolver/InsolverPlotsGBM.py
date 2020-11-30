import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


class PredictionPlots:
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
