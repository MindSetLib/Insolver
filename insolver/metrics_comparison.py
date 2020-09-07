import pandas as pd


class PredictionMetrics(object):
    # TODO: Support of train/val/test splits.
    """Class for calculating metrics using predictions of the models.

    Attributes:
        df_predictions (pd.DataFrame): DataFrame containing all predictions that are used to metrics calculation.
        df_targets (pd.DataFrame): DataFrame containing all targets corresponding to df_predictions.
        target_list (list): List of target names from df_targets for every prediction in df_predictions.
    """
    def __init__(self, df_predictions, df_targets, target_list):
        self.predictions = df_predictions
        self.targets = df_targets
        self.target_list = target_list

    def make_summary(self, function):
        metric_name = function.__name__
        metric = pd.DataFrame()
        for i in range(len(self.predictions.columns)):
            pred_name = self.predictions.columns[i]
            metric[f'{pred_name}'] = [function(self.predictions[pred_name],
                                               self.targets[self.target_list[i]]),
                                      self.targets[self.target_list[i]].mean(),
                                      self.predictions[pred_name].mean()]
        metric.index = [metric_name, 'Mean target', 'Mean prediction']
        return metric.T
