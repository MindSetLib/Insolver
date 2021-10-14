from numpy import log10, std, percentile, subtract

from insolver.discretization.discretizer_utils import *


class InsolverDiscretizer:
    """
    X: column
    method: 'uniform', 'quantile', 'kmeans'
    n_bins: int or str
    """

    _methods = {
        'uniform': EqualWidthDiscretizer,
        'quantile': EqualFrequencyDiscretizer,
        'kmeans': KMeansDiscretizer,
    }

    _n_bins_formula = ('square-root', 'sturgers', 'rice-rule',
                       'scotts-rule', 'freedman-diaconis')

    def __init__(self, X, method='uniform', n_bins=5):
        if method not in self._methods.keys():
            raise ValueError(f'Accepted methods are {self._methods}, got {method} instead.')

        if not ((type(n_bins) == int and n_bins > 1)
                or n_bins in self._n_bins_formula):
            raise ValueError(f'Invalid number of bins. '
                             f'Accepted integer value or one of the following options: {self._n_bins_formula},'
                             f'got {n_bins} instead.')

        self.X = X
        self.method = method
        self.n_bins = n_bins

    def transform(self):
        len_X = self.__check_data_shape()

        if self.n_bins in self._n_bins_formula:
            self.n_bins = self.__calculate_n_bins(self.n_bins, len_X)

        return self._methods[self.method]. \
            _transform(self.X, self.n_bins)

    def __calculate_n_bins(self, n_bins, len_X):
        if n_bins == 'square-root':
            return round(len_X ** 1 / 2)

        if n_bins == 'sturgers':
            return round(1 + 3.322 * log10(len_X))

        if n_bins == 'rice-rule':
            return round(len_X ** 1 / 3 * 2)

        if n_bins == 'scotts-rule':
            return round(3.49 * std(self.X) / len_X ** 1 / 3)

        if n_bins == 'freedman-diaconis':
            iqr = subtract(*percentile(self.X, [75, 25]))
            return round(2 * iqr / len_X ** 1 / 3)

    def __check_data_shape(self):
        if len(self.X.shape) not in (1, 2):
            raise ValueError(f'Expected 1D or 2D array, '
                             f'got shape={self.X.shape} instead.')

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)
            return self.X.shape[0]

        if len(self.X.shape) == 2:
            if self.X.shape[0] == 1:
                self.X = self.X.reshape(-1, 1)

        return self.X.shape[0]
