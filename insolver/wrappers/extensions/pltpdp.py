from sklearn.inspection import plot_partial_dependence
from h2o.frame import H2OFrame

from matplotlib.pyplot import show, tight_layout
from pdpbox.pdp import pdp_isolate, pdp_plot


class InsolverPDPExtension:
    def pdp(self, X, features, feature_name, plot_backend='sklearn', **kwargs):
        if self.backend == 'h2o':
            self.model.partial_plot(H2OFrame(X), features, **kwargs)
        else:
            if plot_backend == 'sklearn':
                if self.backend in ['catboost', 'lightgbm']:
                    self.model.dummy_ = True
                plot_partial_dependence(self.model, X, features=features, **kwargs)
                tight_layout()
                show()
            elif plot_backend == 'pdpbox':
                pdp_plot(pdp_isolate(self.model, X, features, feature_name), feature_name, **kwargs)
                show()
            else:
                raise NotImplementedError(f'Plot backend {plot_backend} is not implemented.')
