from sklearn.inspection import PartialDependenceDisplay
from matplotlib.pyplot import show, tight_layout
from .h2oext import to_h2oframe


class InsolverPDPExtension:
    def pdp(self, X, features, feature_name, plot_backend='sklearn', **kwargs):
        if self.backend == 'h2o':
            self.model.partial_plot(to_h2oframe(X), features, **kwargs)
        else:
            if plot_backend == 'sklearn':
                if self.backend in ['catboost', 'lightgbm']:
                    self.model.dummy_ = True
                display = PartialDependenceDisplay.from_estimator(self.model, X, features=features, **kwargs)
                display.plot()
                tight_layout()
                show()
            elif plot_backend == 'pdpbox':
                try:
                    from pdpbox.pdp import pdp_isolate, pdp_plot

                    pdp_plot(pdp_isolate(self.model, X, features, feature_name), feature_name, **kwargs)
                    show()
                except ImportError:
                    print('Package PDPbox is not installed')
            else:
                raise NotImplementedError(f'Plot backend {plot_backend} is not implemented.')
