from numpy import array, mean, broadcast_to, where
from pandas import concat, merge

from .base import InsolverBaseWrapper


class InsolverTrivialWrapper(InsolverBaseWrapper):
    """Dummy wrapper for returning trivial "predictions" for metric comparison and statistics.

    Parameters:
        col_name (str, list, optional): String or list of strings containing column name(s) to perform
         groupby operation.
        agg (callable, optional): Aggregation function.
        thresh (float, optional): Threshold for continuous prediction in dummy classification.
        **kwargs: Other arguments.
    """

    def __init__(self, task=None, col_name=None, agg=None, thresh=0.5, **kwargs):
        super(InsolverTrivialWrapper, self).__init__(backend='trivial')
        self._tasks = ['class', 'reg']
        self.init_args = self._get_init_args(vars())
        self._backends, self.x_train, self.y_train = ['trivial'], None, None
        self._back_load_dict, self._back_save_dict = {'trivial': self._pickle_load}, {'trivial': self._pickle_save}
        if task in self._tasks:
            self.task = task
            self.thresh = thresh if task == 'class' else None
        else:
            raise NotImplementedError(f'Task parameter supports values in {self._tasks}.')

        if (
            isinstance(col_name, (str, list, tuple))
            or col_name is None
            or (isinstance(col_name, (list, tuple)) and all([isinstance(element, str) for element in col_name]))
        ):
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
            X (pd.DataFrame): Data.
            y (pd.Series): Target values.
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
            X (pd.DataFrame, pd.Series): Data.

        Returns:
            array: Trivial model "prediction".
        """
        if self.col_name is None:
            output = broadcast_to(self.fitted, X.shape[0])
        else:
            output = merge(X[self.col_name], self.fitted, how='left', on=self.col_name)[self.y_train.name].fillna(
                self.agg(self.y_train)
            )
        return array(output) if self.task != 'class' else where(array(output) >= self.thresh, 1, 0)
