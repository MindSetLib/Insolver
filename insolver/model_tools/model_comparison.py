import os
import traceback
from glob import glob

from numpy import min, max, mean, var, std, quantile, median
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper, InsolverRFWrapper, InsolverTrivialWrapper


class ModelMetricsCompare:
    """Class for model comparison.
    It will compute statistics and metrics for the regression task and metrics for the classification task.
    You can compare created models with the `source` parameter or if `source` is `None` it use current working directory
    as a source. If you want to create new models set the `create_models` parameter to True. If you already have source
    parameter and set `create_models` parameter to `True`, new models will be added to the source list.

    Parameters:
        X (pd.DataFrame, pd.Series): Data for making predictions.
        y (pd.DataFrame, pd.Series): Actual target values for X.
        task (str, None): A task for models and metrics. If `task` new models will be created
         Supports 'reg' and 'class'.
        create_models (bool): If True, new models will be created and added to the comparison list.
        source (str, list, tuple, None): List or tuple of insolver wrappers or path to the
         folder with models. If `None`, taking current working directory as source.
        metrics (list, tuple, callable, optional): Metrics or list of metrics to compute.
        stats (list, tuple, callable, optional): Statistics or list of statistics to compute.
        h2o_init_params (dict, optional): Parameters passed to `h2o.init()`, when `backend` == 'h2o'.
        predict_params (list, optional): List of dictionaries containing parameters passed to predict methods
         for each model.
        features (list, optional): List of lists containing features for predict method for each model.
        names (list, optional): List of model names.

    """

    def __init__(
        self,
        X,
        y,
        task=None,
        create_models=False,
        source=None,
        metrics=None,
        stats=None,
        h2o_init_params=None,
        predict_params=None,
        features=None,
        names=None,
    ):
        self.X = X
        self.y = y
        self.task = task
        self.create_models = create_models
        self.source = source
        self.metrics = [] if metrics is None else metrics
        self.stats = stats
        self.h2o_init_params = h2o_init_params
        self.predict_params = predict_params
        self.features = features
        self.names = names
        self.stats_results, self.metrics_results = DataFrame(), DataFrame()

    def __repr__(self):
        stk = traceback.extract_stack()
        if not ('IPython' in stk[-2][0] and 'info' == stk[-2][2]):
            import IPython.display

            if self.task == 'reg':
                print('Model comparison statistics:')
                IPython.display.display(self.stats_results)
            print('\nModels comparison metrics:')
            IPython.display.display(self.metrics_results)
        else:
            if self.task == 'reg':
                print('Model comparison statistics:')
                print(self.stats_results)
            print('\nModels comparison metrics:')
            print(self.metrics_results)
        return ''

    def compare(self):
        """Compares models using initialized parameters.
        If `self.create_models` == True, new models will be created and added to the source list.

        Raises:
            Exception: `task` parameter must be initialized and be `class` or `reg`.

        """
        if self.task not in ['reg', 'class']:
            raise Exception('Task must be "reg" or "class".')

        if self.create_models:
            self._init_new_models()

        self._init_default_metrics()
        self._init_source_models()

        self._calc_metrics()
        self.__repr__()

    def _init_new_models(self):
        """Initializes new models using the `task` parameter.
        If `class` then Gradient Boosting model with the catboost backend and Random Forest with the sklearn backend
        will be created.
        If `reg` then Gradient Boosting model with the catboost backend, Random Forest with the sklearn backend and
        Linear Model with the sklearn backend will be created.
        This method uses train_test_split from sklearn.model_selection, fits models with train values and changes
        `self.X`, `self.y` to test values. Thus, when calculating metrics, it will use test values.

        """

        self.source = [] if self.source is None else self.source
        models_dict = {}
        if self.task == 'class':
            models_dict = {
                'new_gbm': InsolverGBMWrapper(backend='catboost', task='class', n_estimators=10),
                'new_rf': InsolverRFWrapper(backend='sklearn', task='class'),
            }
        elif self.task == 'reg':
            models_dict = {
                'new_glm': InsolverGLMWrapper(backend='sklearn', family=0, standardize=True),
                'new_gbm': InsolverGBMWrapper(backend='catboost', task='reg', n_estimators=10),
                'new_rf': InsolverRFWrapper(backend='sklearn', task='reg'),
            }

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        for model_name in models_dict:
            _model = models_dict[model_name]
            _model.fit(X_train, y_train)
            _model.algo = model_name
            self.source.insert(0, _model)

        self.X, self.y = X_test, y_test

    def _init_default_metrics(self):
        """Initializes default metrics and adds them to the metrics list.
        If `class` then accuracy score and f1 score will be added.
        If `reg` then mean absolute error and r2 score will be added.
        If `self.metrics` is callable it will be changed to the list type.

        """
        if callable(self.metrics):
            self.metrics = [self.metrics]

        if self.task == 'class':
            self.metrics.insert(0, accuracy_score)
            self.metrics.insert(1, f1_score)

        elif self.task == 'reg':
            self.metrics.insert(0, mean_absolute_error)
            self.metrics.insert(1, r2_score)

    def _init_source_models(self):
        """Initializes source models.
        if `source` is `None` it use current working directory as a source.

        Raises:
            Exception: Models with the insolver name format were not found in the current working directory.
            TypeError: Source type is not supported.

        """
        assert (
            True if self.names is None else len(self.names) == len(self.source)
        ), 'Check length of list containing model names.'
        wrappers = {'glm': InsolverGLMWrapper, 'gbm': InsolverGBMWrapper, 'rf': InsolverRFWrapper}

        if (self.source is None) or isinstance(self.source, str):
            self.source = os.getcwd() if self.source is None else self.source
            files = glob(os.path.join(self.source, '*'))
            files = [file for file in files if os.path.basename(file).split('_')[0] == 'insolver']
            if files:
                model_list = []
                for file in files:
                    algo, backend = os.path.basename(file).split('_')[1:3]
                    model_list.append(
                        wrappers[algo](backend=backend, load_path=file)
                        if backend != 'h2o'
                        else wrappers[algo](backend=backend, load_path=file, h2o_init_params=self.h2o_init_params)
                    )
                self.models = model_list

            else:
                raise Exception('No models with the insolver name format found.')

        elif isinstance(self.source, (list, tuple)):
            self.models = self.source

        else:
            raise TypeError(f'Source of type {type(self.source)} is not supported.')

    def _calc_metrics(self):
        """Computes metrics and statistics for the models.

        Raises:
            TypeError: Statistics type are not supported.
            TypeError: Metrics type are not supported.

        Returns:
            Returns `None`, but results available in `self.stats`, `self.metrics`.

        """

        stats_df, model_metrics = DataFrame(), DataFrame()
        algos, backend = [], []
        trivial = InsolverTrivialWrapper(task='reg', agg=lambda x: x)
        trivial.fit(self.X, self.y)
        models = [trivial] + self.models
        features = [None] + self.features if self.features is not None else None
        for model in models:
            algos.append(model.algo.upper()) if hasattr(model, 'algo') else algos.append('-')
            (
                backend.append(model.backend.capitalize())
                if hasattr(model, 'backend')
                else backend.append(model.__class__.__name__)
            )

            p = model.predict(
                self.X
                if (features is None) or (features[models.index(model)] is None)
                else self.X[features[models.index(model)]],
                **(
                    {}
                    if (self.predict_params is None) or (self.predict_params[models.index(model)] is None)
                    else self.predict_params[models.index(model)]
                ),
            )

            stats_val = [mean(p), var(p), std(p), min(p), quantile(p, 0.25), median(p), quantile(p, 0.75), max(p)]

            name_stats = ['Mean', 'Variance', 'St. Dev.', 'Min', 'Q1', 'Median', 'Q3', 'Max']

            if self.stats is not None:
                if isinstance(self.stats, (list, tuple)):
                    for stat in self.stats:
                        if callable(stat):
                            stats_val.append(stat(p))
                            name_stats.append(stat.__name__.replace('_', ' '))
                        else:
                            raise TypeError(f'Statistics with type {type(stat)} are not supported.')

                elif callable(self.stats):
                    stats_val.append(self.stats(p))
                    name_stats.append(self.stats.__name__.replace('_', ' '))
                else:
                    raise TypeError(f'Statistics with type {type(self.stats)} are not supported.')

            stats_df = stats_df.append(DataFrame([stats_val], columns=name_stats))

            if (self.metrics is not None) and not models.index(model) == 0:
                if isinstance(self.metrics, (list, tuple)):
                    m_metrics = []
                    for metric in self.metrics:
                        if callable(metric):
                            m_metrics.append(metric(self.y, p))
                        else:
                            raise TypeError(f'Metrics with type {type(metric)} are not supported.')

                    metrics_names = [m.__name__.replace('_', ' ') for m in self.metrics]
                    model_metrics = model_metrics.append(DataFrame(dict(zip(metrics_names, m_metrics), index=[0])))

                else:
                    raise TypeError(f'Metrics with type {type(self.metrics)} are not supported.')

                model_metrics = model_metrics.reset_index(drop=True)
                model_metrics = (
                    model_metrics if 'index' not in model_metrics.columns else model_metrics.drop(['index'], axis=1)
                )

        model_metrics.index = list(range(len(model_metrics))) if self.names is None else self.names
        stats_df.index = ['Actual'] + model_metrics.index.tolist()
        stats_df[['Algo', 'Backend']] = DataFrame(
            {'Algo': ['-'] + algos[1:], 'Backend': ['-'] + backend[1:]}, index=stats_df.index
        )
        model_metrics[['Algo', 'Backend']] = DataFrame(
            {'Algo': algos[1:], 'Backend': backend[1:]}, index=model_metrics.index
        )
        stats_df = stats_df[list(stats_df.columns[-2:]) + list(stats_df.columns[:-2])]
        model_metrics = model_metrics[list(model_metrics.columns[-2:]) + list(model_metrics.columns[:-2])]

        self.stats_results = self.stats_results.append(stats_df)
        self.metrics_results = self.metrics_results.append(model_metrics)
