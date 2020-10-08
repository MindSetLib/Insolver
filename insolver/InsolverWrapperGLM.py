import warnings

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch


class InsolverGLMWrapper:
    def __init__(self, **kwargs):
        self.best_params, self.model, self.train, self.valid = None, None, None, None
        h2o.init(**kwargs)

    def model_init(self, training_frame, validation_frame, **kwargs):
        self.model = H2OGeneralizedLinearEstimator(**kwargs)
        self.train, self.valid = [h2o.H2OFrame(x) for x in [training_frame, validation_frame]]

    def grid_search_cv(self, features, target, hyper_params, **kwargs):
        model_grid = H2OGridSearch(model=self.model, hyper_params=hyper_params, **kwargs)
        model_grid.train(y=target, x=features, training_frame=self.train, validation_frame=self.valid)
        sorted_grid = model_grid.get_grid(sort_by='residual_deviance', decreasing=False)
        self.best_params = sorted_grid[0]
        self.model = sorted_grid.models[0]

    def fit(self, features, target, **kwargs):
        if self.model is None:
            warnings.warn('Model is not initiated, please use .model_init() method.')
        else:
            if self.best_params is None:
                self.model.train(y=target, x=features, training_frame=self.train, validation_frame=self.valid, **kwargs)

    def predict(self, X, **kwargs):
        if self.model is None:
            warnings.warn('Please fit or load a model first.')
        else:
            h2o_predict = X if isinstance(X, h2o.H2OFrame) else h2o.H2OFrame(X)
            return self.model.predict(h2o_predict, **kwargs).as_data_frame().values

    def save_model(self, path, **kwargs):
        h2o.save_model(model=self.model, path=path, **kwargs)

    def load_model(self, path):
        self.model = h2o.load_model(path)
