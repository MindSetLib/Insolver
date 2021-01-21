import os
import time
import pickle

from numpy import array, mean, broadcast_to
from pandas import concat, merge


class InsolverBaseWrapper:
    def __init__(self, backend):
        """Base wrapper serving as a building block for other wrappers.

        Attributes:
            backend (str): Name of the backend to built the model.
        """
        self.algo, self.backend, self._backends = None, backend, None
        self._back_load_dict, self._back_save_dict = None, None
        self.object, self.model = None, None
        self.best_params, self.trials = None, None

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

    def save_model(self, path=None, name=None, suffix=None, **kwargs):
        """Saving the model contained in wrapper.

        Args:
            path (:obj:`str`, optional): Path to save the model. Using current working directory by default.
            name (:obj:`str`, optional): Optional, name of the model.
            suffix (:obj:`str`, optional): Optional, suffix in the name of the model.
            **kwargs: Other parameters passed to, e.g. h2o.save_model().
        """
        path = os.getcwd() if path is None else os.path.normpath(path)
        def_name = f"insolver_{self.algo}_{self.backend}_{round(time.time() * 1000)}"
        name = name if name is not None else def_name
        name = name if suffix is None else f'{name}_{suffix}'

        self.model.algo = self.algo
        self.model.backend = self.backend

        if self.backend in self._back_save_dict.keys():
            self._back_save_dict[self.backend](path, name, **kwargs)
        else:
            raise NotImplementedError(f'Error with the backend choice. Supported backends: {self._backends}')

    def _pickle_load(self, load_path):
        with open(load_path, 'rb') as _model:
            self.model = pickle.load(_model)

    def _pickle_save(self, path, name):
        with open(os.path.join(path, name), 'wb') as _model:
            pickle.dump(self.model, _model, pickle.HIGHEST_PROTOCOL)


class InsolverTrivialWrapper(InsolverBaseWrapper):
    """Dummy wrapper for returning trivial "predictions" for metric comparison and statistics.

    Attributes:
        col_name (:obj:`str` or :obj:`list`, optional): String or list of strings containing column name(s) to perform
         groupby operation.
        agg (:obj:`callable`, optional): Aggregation function.
        **kwargs: Other arguments.
    """
    def __init__(self, col_name=None, agg=None, **kwargs):
        super(InsolverTrivialWrapper, self).__init__(backend='trivial')
        self._backends, self.x_train, self.y_train = ['trivial'], None, None
        self._back_load_dict, self._back_save_dict = {'trivial': self._pickle_load}, {'trivial': self._pickle_save}

        if (isinstance(col_name, (str, list, tuple)) or col_name is None or
                (isinstance(col_name, (list, tuple)) and all([isinstance(element, str) for element in col_name]))):
            self.col_name = col_name
        else:
            raise TypeError(f'Column of type {type(self.col_name)} is not supported.')
        self.fitted, self.agg, self.kwargs = None, agg, kwargs
        self.agg = mean if self.agg is None else self.agg

        if self.col_name is None:
            self.algo = self.agg.__name__.replace('_', ' ')
        else:
            self.algo = f"{self.agg.__name__} target: {self.col_name}"

    def fit(self, X, y):
        """Fitting dummy model.

        Args:
            X (:obj:`pd.DataFrame`): Data.
            y (:obj:`pd.Series`): Target values.
        """
        self.x_train, self.y_train = X, y
        if self.col_name is None:
            self.fitted = self.agg(self.y_train)
        else:
            _df = concat([self.y_train, self.x_train[self.col_name]], axis=1)
            self.fitted = _df.groupby(self.col_name).aggregate(self.agg).reset_index()

    def predict(self, X):
        """Making dummy predictions.

        Args:
            X (:obj:`pd.DataFrame` or :obj:`pd.Series`): Data.

        Returns:
            array: Trivial model "prediction".
        """
        if self.col_name is None:
            output = broadcast_to(self.fitted, X.shape[0])
        else:
            output = merge(X[self.col_name], self.fitted, how='left', on=self.col_name)[self.y_train.name].fillna(
                self.agg(self.y_train))
        return array(output)
