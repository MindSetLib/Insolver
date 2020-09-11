import warnings

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch


class InsolverGLMWrapper(object):
    def __init__(self, **kwargs):
        self.best_params, self.model = None, None
        h2o.init(**kwargs)

    def model_init(self, data, ratios, **kwargs):
        data_model = h2o.H2OFrame(data)
        train, test, valid = data_model.split_frame(ratios=ratios)
        self.model = H2OGeneralizedLinearEstimator(**kwargs)
        return train, test, valid

    def grid_search_cv(self, X, y, h2odf_train, h2odf_valid, hyper_params, **kwargs):
        model_grid = H2OGridSearch(model=self.model, hyper_params=hyper_params, **kwargs)
        model_grid.train(y=y, x=X, training_frame=h2odf_train, validation_frame=h2odf_valid)
        sorted_grid = model_grid.get_grid(sort_by='residual_deviance', decreasing=False)
        self.best_params = sorted_grid[0]
        self.model = sorted_grid.models[0]

    def fit(self, X, y, **kwargs):
        if self.model is None:
            warnings.warn('Model is not initiated, please use .model_init() method.')
        else:
            if self.best_params is None:
                self.model.train(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self.model is None:
            warnings.warn('Please fit or load a model first.')
        else:
            return self.model.predict(X, **kwargs)
