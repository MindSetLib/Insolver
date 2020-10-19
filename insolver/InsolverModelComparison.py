import pickle
from glob import glob

import h2o
from h2o.exceptions import H2OResponseError, H2OServerError
from numpy import array
from pandas import DataFrame

from insolver.InsolverWrapperGBM import InsolverGradientBoostingWrapper
from insolver.InsolverWrapperGLM import InsolverGLMWrapper


class ModelCompare:
    def __init__(self, path='./'):
        self.path = f'{path}*' if path.endswith('/') else f'{path}/*'
        self.models = None
        h2o.init()

    def search_for_models(self):
        files, models = glob(f'{self.path}'), dict()
        for file in files:
            try:
                h2o.load_model(file)
                print(f'Model found: {file}')
                models[file] = 'h2o'
            except (H2OResponseError, H2OServerError):
                try:
                    with open(file, 'rb') as f:
                        obj = pickle.load(f)
                except (pickle.UnpicklingError, EOFError, PermissionError):
                    continue
                if isinstance(obj, dict):
                    if 'model' in obj.keys():
                        obj, value = obj['model'], 'booster'
                    else:
                        continue
                else:
                    value = 'model'
                if callable(getattr(obj, "predict", None)):
                    models[file] = value
                    print(f'Model found: {file}')
        self.models = models

    def metrics_comparison(self, metrics, x_test, y_test):
        model_metrics = list()
        for model in self.models.keys():
            if self.models[model] in ['booster', 'model']:
                mdl = InsolverGradientBoostingWrapper(algorithm='xgboost', task='regression')
                mdl.load_model(model) if self.models[model] == 'model' else mdl.load_booster(model)
            elif self.models[model] == 'h2o':
                mdl = InsolverGLMWrapper()
                mdl.load_model(model)
            else:
                raise Exception

            predict = mdl.predict(x_test)
            if isinstance(metrics, (list, tuple)):
                for metric in metrics:
                    model_metrics.append(metric(y_test, predict))
            else:
                model_metrics.append(metrics(y_test, predict))

        if isinstance(metrics, (list, tuple)):
            output = DataFrame(array(model_metrics).reshape(-1, len(metrics)).T,
                               index=[x.__name__ for x in metrics], columns=self.models.keys()).T
        else:
            output = DataFrame(array([model_metrics]), index=[metrics.__name__], columns=self.models.keys()).T
        return output
