import os
from glob import glob

from numpy import min, max, mean, var, std, quantile, median
from pandas import DataFrame

from insolver.wrappers.base import InsolverWrapperBase
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper


class _InsolverWrapperDummy(InsolverWrapperBase):
    def __init__(self, backend='dummy', y=None):
        super(_InsolverWrapperDummy, self).__init__(backend)
        self.algo, self.y = 'actual', y

    def predict(self, X):
        if len(X) == len(self.y):
            return self.y


class ModelCompare:
    """Class for model comparison.

    Args:
        source (:obj:`str`, :obj:`list`, :obj:`tuple`, :ibj:`None`): List or tuple of insolver wrappers or path to the
        folder with models. If `None`, taking current working directory as source.
    """
    def __init__(self, source=None):
        wrappers = {'glm': InsolverGLMWrapper, 'gbm': InsolverGBMWrapper}
        self.stats, self.metrics = None, None
        if (source is None) or isinstance(source, str):
            source = os.getcwd() if source is None else source
            files = glob(os.path.join(source, '*'))
            files = [file for file in files if os.path.basename(file).split('_')[0] == 'insolver']
            if len(files) > 0:
                model_list = []
                for file in files:
                    algo, backend = os.path.basename(file).split('_')[1:3]
                    model_list.append(wrappers[algo](backend=backend, load_path=file))
                self.models = model_list
            else:
                raise Exception('No models with appropriate name format found.')
        elif isinstance(source, (list, tuple)):
            self.models = source
        else:
            raise NotImplementedError(f'Source of type {type(source)} is not supported.')

    def compare_metrics(self, X, y, metrics=None, stats=None):
        """Computing metrics and statistics for models.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Data for making predictions.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Actual target values for X.
            metrics (:obj:`list`, :obj:`tuple`, :obj:`callable`, optional): Metrics or list of metrics to compute.
            stats (:obj:`list`, :obj:`tuple`, :obj:`callable`, optional): Statistics or list of statistics to compute.

        Returns:
            Returns `None`, but results available in `self.stats`, `self.metrics`.
        """
        stats_df, model_metrics, model_names = DataFrame(), DataFrame(), []
        for model in [_InsolverWrapperDummy(y=y)] + self.models:
            p = model.predict(X)
            stats_val = [mean(p), var(p), std(p), min(p), quantile(p, 0.25), median(p), quantile(p, 0.75), max(p)]
            name_stats = ['Mean', 'Variance', 'St. Dev.', 'Min', 'Q1', 'Median', 'Q3', 'Max']
            if stats is not None:
                if isinstance(stats, (list, tuple)):
                    for stat in stats:
                        if callable(stat):
                            stats_val.append(stat(p))
                            name_stats.append(stat.__name__.replace('_', ' '))
                        else:
                            raise NotImplementedError(f'Statistics with type {type(stat)} are not supported.')
                elif callable(stats):
                    stats_val.append(stats(p))
                    name_stats.append(stats.__name__.replace('_', ' '))
                else:
                    raise NotImplementedError(f'Statistics with type {type(stats)} are not supported.')
            stats_df = stats_df.append(DataFrame([stats_val], columns=name_stats))
            model_names.append(f'{model.algo.upper()} {model.backend.capitalize()}')
            stats_df.index = ['Actual'] + model_names[1:]
            self.stats = stats_df

            if (metrics is not None) and not isinstance(model, _InsolverWrapperDummy):
                if isinstance(metrics, (list, tuple)):
                    m_metrics = []
                    for metric in metrics:
                        if callable(metric):
                            m_metrics.append(metric(y, p))
                        else:
                            raise NotImplementedError(f'Metrics with type {type(metric)} are not supported.')
                    model_metrics = model_metrics.append(DataFrame(m_metrics, index=[m.__name__.replace('_', ' ') for m
                                                                                     in metrics]).T)
                elif callable(metrics):
                    model_metrics = model_metrics.append(DataFrame([metrics(y, p)],
                                                                   index=[metrics.__name__.replace('_', ' ')]).T)
                else:
                    raise NotImplementedError(f'Metrics with type {type(metrics)} are not supported.')
                model_metrics.index = model_names[1:]
                self.metrics = model_metrics
