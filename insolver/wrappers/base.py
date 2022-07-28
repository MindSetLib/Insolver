import os
import copy
import time
import pickle


class InsolverBaseWrapper:
    """Base wrapper serving as a building block for other wrappers.

    Parameters:
        backend (str): Name of the backend to build the model.
    """

    def __init__(self, backend):
        self.algo, self.backend, self._backends = None, backend, None
        self._back_load_dict, self._back_save_dict = None, None
        self.meta, self.object, self.model = None, None, None
        self.best_params, self.trials = None, None

    def __call__(self):
        return self.model

    @staticmethod
    def _get_init_args(vars_):
        c_vars = copy.deepcopy(vars_)
        for key in ['__class__', 'self']:
            del c_vars[key]
        return c_vars

    def load_model(self, load_path):
        """Loading a model to the wrapper.

        Args:
            load_path (str): Path to the model that will be loaded to wrapper.
        """
        load_path = os.path.normpath(load_path)
        if self.backend in self._back_load_dict.keys():
            self._back_load_dict[self.backend](load_path)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def save_model(self, path=None, name=None, suffix=None, **kwargs):
        """Saving the model contained in wrapper.

        Args:
            path (str, optional): Path to save the model. Using current working directory by default.
            name (str, optional): Optional, name of the model.
            suffix (str, optional): Optional, suffix in the name of the model.
            **kwargs: Other parameters passed to, e.g. h2o.save_model().
        """
        path = os.getcwd() if path is None else os.path.normpath(path)
        def_name = f"insolver_{self.algo}_{self.backend}_{round(time.time() * 1000)}"
        name = name if name is not None else def_name
        name = name if suffix is None else f'{name}_{suffix}'

        self.model.insolver_meta = self.meta
        self.model.algo = self.algo
        self.model.backend = self.backend

        if self.backend in self._back_save_dict.keys():
            self._back_save_dict[self.backend](path, name, **kwargs)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def _pickle_load(self, load_path):
        with open(load_path, 'rb') as _model:
            self.model = pickle.load(_model)

    def _pickle_save(self, path, name, **kwargs):
        with open(os.path.join(path, f'{name}.pickle'), 'wb') as _model:
            pickle.dump(self.model, _model, **kwargs)

    def _update_meta(self):
        self.meta = self.__dict__.copy()
        for key in ['_backends', '_back_load_dict', '_back_save_dict', 'object', 'model', 'meta']:
            self.meta.pop(key)
