import warnings

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch


class InsolverGLMWrapper(object):
    def __init__(self, **kwargs):
        self.best_params, self.model = None, None
        h2o.init(**kwargs)

    def model_init(self, train, valid, **kwargs):
        h2o_train, h2o_valid = [h2o.H2OFrame(x) for x in [train, valid]]
        self.model = H2OGeneralizedLinearEstimator(**kwargs)
        return h2o_train, h2o_valid

    def grid_search_cv(self, X, y, h2odf_train, h2odf_valid, hyper_params, **kwargs):
        model_grid = H2OGridSearch(model=self.model, hyper_params=hyper_params, **kwargs)
        model_grid.train(y=y, x=X, training_frame=h2odf_train, validation_frame=h2odf_valid)
        sorted_grid = model_grid.get_grid(sort_by='residual_deviance', decreasing=False)
        self.best_params = sorted_grid[0]
        self.model = sorted_grid.models[0]

    def fit(self, X, y, training_frame, validation_frame, **kwargs):
        if self.model is None:
            warnings.warn('Model is not initiated, please use .model_init() method.')
        else:
            if self.best_params is None:
                self.model.train(y=y, x=X, training_frame=training_frame, validation_frame=validation_frame, **kwargs)

    def predict(self, X, **kwargs):
        if self.model is None:
            warnings.warn('Please fit or load a model first.')
        else:
            h2o_predict = X if isinstance(X, h2o.H2OFrame) else h2o.H2OFrame(X)
            return self.model.predict(h2o_predict, **kwargs).as_data_frame()

    def save_model(self, path, **kwargs):
        h2o.save_model(model=self.model, path=path, **kwargs)

    def load_model(self, path):
        self.model = h2o.load_model(path)
