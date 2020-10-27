import os
import time
import pickle

from pandas import DataFrame, Series, concat

from h2o import no_progress, cluster, init, load_model, save_model
from h2o.frame import H2OFrame

from . import __version__


class InsolverWrapperMain:
    def __init__(self, backend):
        self.backend, self._backends = backend, None
        self._back_load_dict, self._back_save_dict = None, None
        self.model, self.features, self.best_params = None, None, None

    def __call__(self):
        return self.model

    def load_model(self, load_path):
        """Loading a model to the wrapper.

        Args:
            load_path (:obj:`str`): Path to the model that will be loaded to wrapper.
        """
        load_path = os.path.normpath(load_path)
        if self.backend in self._back_load_dict.keys():
            self._back_load_dict[self.backend](load_path)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def _pickle_load(self, load_path):
        with open(load_path, 'rb') as _model:
            self.model = pickle.load(_model)

    def _pickle_save(self, path, name):
        with open(os.path.join(path, name), 'wb') as _model:
            pickle.dump(self.model, _model, pickle.HIGHEST_PROTOCOL)

    def save_model(self, path=None, name=None, **kwargs):
        """Saving the model contained in wrapper.

        Args:
            path (:obj:`str`, optional): Path to save the model. Using current working directory by default.
            name (:obj:`str`, optional): Optional, name of the model.
            **kwargs: Other parameters passed to, e.g. h2o.save_model().
        """
        path = os.getcwd() if path is None else os.path.normpath(path)
        def_name = f"insolver_{__version__.replace('.', '-')}_glm_{self.backend}_{round(time.time() * 1000)}"
        name = name if name is not None else def_name

        if self.backend in self._back_save_dict.keys():
            self._back_save_dict[self.backend](path, name, **kwargs)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')


class InsolverWrapperH2O:
    @staticmethod
    def _h2o_init(h2o_init_params):
        no_progress()
        if cluster() is None:
            if h2o_init_params is not None:
                init(**h2o_init_params)
            else:
                init()

    def _h2o_load(self, load_path, h2o_init_params):
        self._h2o_init(h2o_init_params)
        self.model = load_model(load_path)

    def _h2o_save(self, path, name, **kwargs):
        model_path = save_model(model=self.model, path=path, **kwargs)
        os.rename(model_path, os.path.join(os.path.dirname(model_path), name))

    @staticmethod
    def _x_y_to_h2o_frame(X, y, sample_weight, params, X_valid, y_valid, sample_weight_valid):
        if isinstance(X, (DataFrame, Series)) & isinstance(y, (DataFrame, Series)):
            features = X.columns.tolist() if isinstance(X, DataFrame) else X.name
            target = y.columns.tolist() if isinstance(y, DataFrame) else y.name
            if (sample_weight is not None) & isinstance(sample_weight, (DataFrame, Series)):
                params['offset_column'] = (sample_weight.columns.tolist() if isinstance(sample_weight, DataFrame)
                                           else sample_weight.name)
                # noinspection PyPep8Naming
                X = concat([X, sample_weight], axis=1)
            train_set = H2OFrame(concat([X, y], axis=1))
        else:
            raise NotImplementedError('X, y are supposed to be pandas DataFrame or Series')

        if (X_valid is not None) & (y_valid is not None):
            if isinstance(X_valid, (DataFrame, Series)) & isinstance(y_valid, (DataFrame, Series)):
                if ((sample_weight_valid is not None) & isinstance(sample_weight_valid, (DataFrame, Series)) &
                        (sample_weight is not None)):
                    # noinspection PyPep8Naming
                    X_valid = concat([X_valid, sample_weight_valid], axis=1)
                valid_set = H2OFrame(concat([X_valid, y_valid], axis=1))
                params['validation_frame'] = valid_set
            else:
                raise NotImplementedError('X_valid, y_valid are supposed to be pandas DataFrame or Series')
        return features, target, train_set, params
