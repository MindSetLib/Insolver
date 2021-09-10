import os
import traceback
from glob import glob

from numpy import min, max, mean, var, std, quantile, median
from pandas import DataFrame

from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper, InsolverRFWrapper, InsolverTrivialWrapper


class ModelMetricsCompare:
    """Class for model comparison.

    Attributes:
        X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Data for making predictions.
        y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Actual target values for X.
        source (:obj:`str`, :obj:`list`, :obj:`tuple`, :ibj:`None`): List or tuple of insolver wrappers or path to the
        metrics (:obj:`list`, :obj:`tuple`, :obj:`callable`, optional): Metrics or list of metrics to compute.
        stats (:obj:`list`, :obj:`tuple`, :obj:`callable`, optional): Statistics or list of statistics to compute.
        folder with models. If `None`, taking current working directory as source.
        h2o_init_params (:obj:`dict`, optional): Parameters passed to `h2o.init()`, when `backend` == 'h2o'.
        predict_params (:obj:`list`, optional): List of dictionaries containing parameters passed to predict methods
         for each model.
        features (:obj:`list`, optional): List of lists containing features for predict method for each model.
        names (:obj:`list`, optional): List of model names.
    """

    def __init__(self, X, y, source=None, metrics=None, stats=None, h2o_init_params=None, predict_params=None,
                 features=None, names=None):
        assert True if names is None else len(names) == len(source), 'Check length of list containing model names.'
        wrappers = {'glm': InsolverGLMWrapper, 'gbm': InsolverGBMWrapper, 'rf': InsolverRFWrapper}
        self.stats, self.metrics = None, None
        if (source is None) or isinstance(source, str):
            source = os.getcwd() if source is None else source
            files = glob(os.path.join(source, '*'))
            files = [file for file in files if os.path.basename(file).split('_')[0] == 'insolver']
            if files:
                model_list = []
                for file in files:
                    algo, backend = os.path.basename(file).split('_')[1:3]
                    model_list.append(wrappers[algo](backend=backend, load_path=file) if backend != 'h2o' else
                                      wrappers[algo](backend=backend, load_path=file, h2o_init_params=h2o_init_params))
                self.models = model_list
            else:
                raise Exception('No models with appropriate name format found.')
        elif isinstance(source, (list, tuple)):
            self.models = source
        else:
            raise TypeError(f'Source of type {type(source)} is not supported.')

        self._calc_metrics(X=X, y=y, metrics=metrics, stats=stats, predict_params=predict_params, features=features,
                           names=names)

    def __repr__(self):
        stk = traceback.extract_stack()
        if not ('IPython' in stk[-2][0] and 'info' == stk[-2][2]):
            import IPython.display
            print('Model comparison statistics:')
            IPython.display.display(self.stats)
            print('\nModels comparison metrics:')
            IPython.display.display(self.metrics)
        else:
            print('Model comparison statistics:')
            print(self.stats)
            print('\nModels comparison metrics:')
            print(self.metrics)
        return ''

    def _calc_metrics(self, X, y, metrics=None, stats=None, predict_params=None, features=None, names=None):
        """Computing metrics and statistics for models.

        Args:
            X (:obj:`pd.DataFrame`, :obj:`pd.Series`): Data for making predictions.
            y (:obj:`pd.DataFrame`, :obj:`pd.Series`): Actual target values for X.
            metrics (:obj:`list`, :obj:`tuple`, :obj:`callable`, optional): Metrics or list of metrics to compute.
            stats (:obj:`list`, :obj:`tuple`, :obj:`callable`, optional): Statistics or list of statistics to compute.
            predict_params (:obj:`list`, optional): List of dictionaries containing parameters passed to predict methods
         for each model.
            features (:obj:`list`, optional): List of lists containing features for predict method for each model.
            names (:obj:`list`, optional): List of model names.

        Returns:
            Returns `None`, but results available in `self.stats`, `self.metrics`.
        """
        stats_df, model_metrics, algos, backend = DataFrame(), DataFrame(), [], []
        trivial = InsolverTrivialWrapper(agg=lambda x: x)
        trivial.fit(X, y)
        models = [trivial] + self.models
        features = [None] + features if features is not None else None
        for model in models:
            algos.append(model.algo.upper()) if hasattr(model, 'algo') else algos.append('-')
            backend.append(model.backend.capitalize()) if hasattr(model, 'backend') else backend.append(
                model.__class__.__name__)
            p = model.predict(X if (features is None) or (features[models.index(model)] is None)
                              else X[features[models.index(model)]],
                              **({} if (predict_params is None) or (predict_params[models.index(model)] is None)
                                 else predict_params[models.index(model)]))
            stats_val = [mean(p), var(p), std(p), min(p), quantile(p, 0.25), median(p), quantile(p, 0.75), max(p)]
            name_stats = ['Mean', 'Variance', 'St. Dev.', 'Min', 'Q1', 'Median', 'Q3', 'Max']
            if stats is not None:
                if isinstance(stats, (list, tuple)):
                    for stat in stats:
                        if callable(stat):
                            stats_val.append(stat(p))
                            name_stats.append(stat.__name__.replace('_', ' '))
                        else:
                            raise TypeError(f'Statistics with type {type(stat)} are not supported.')
                elif callable(stats):
                    stats_val.append(stats(p))
                    name_stats.append(stats.__name__.replace('_', ' '))
                else:
                    raise TypeError(f'Statistics with type {type(stats)} are not supported.')
            stats_df = stats_df.append(DataFrame([stats_val], columns=name_stats))

            if (metrics is not None) and not models.index(model) == 0:
                if isinstance(metrics, (list, tuple)):
                    m_metrics = []
                    for metric in metrics:
                        if callable(metric):
                            m_metrics.append(metric(y, p))
                        else:
                            raise TypeError(f'Metrics with type {type(metric)} are not supported.')
                    metrics_names = [m.__name__.replace('_', ' ') for m in metrics]
                    model_metrics = model_metrics.append(DataFrame(dict(zip(metrics_names, m_metrics), index=[0])))
                elif callable(metrics):
                    name = [metrics.__name__.replace('_', ' ')]
                    model_metrics = model_metrics.append(DataFrame(dict(zip(name, [metrics(y, p)])), index=[0]))
                else:
                    raise TypeError(f'Metrics with type {type(metrics)} are not supported.')
                model_metrics = model_metrics.reset_index(drop=True)
                model_metrics = (model_metrics if 'index' not in model_metrics.columns
                                 else model_metrics.drop(['index'], axis=1))

        model_metrics.index = (list(range(len(model_metrics))) if names is None else names)
        stats_df.index = ['Actual'] + model_metrics.index.tolist()
        stats_df[['Algo', 'Backend']] = DataFrame({'Algo': ['-'] + algos[1:], 'Backend': ['-'] + backend[1:]},
                                                  index=stats_df.index)
        model_metrics[['Algo', 'Backend']] = DataFrame({'Algo': algos[1:], 'Backend': backend[1:]},
                                                       index=model_metrics.index)
        stats_df = stats_df[list(stats_df.columns[-2:]) + list(stats_df.columns[:-2])]
        model_metrics = model_metrics[list(model_metrics.columns[-2:]) + list(model_metrics.columns[:-2])]
        self.stats, self.metrics = stats_df, model_metrics
