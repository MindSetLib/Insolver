from numpy import log10, std, percentile, subtract, sqrt, power, ndarray

from insolver.discretization.discretizer_utils import *


class InsolverDiscretizer:
    """Trasform continuous variable into discrete form.

    Parameters:
        method (str): The method used to discretize. Should be in {'uniform', 'quantile', 'kmeans', 'cart'}.
    
    """

    _methods = ['uniform', 'quantile', 'kmeans', 'cart']

    _n_bins_formula = ('square-root', 'sturges', 'rice-rule',
                       'scotts-rule', 'freedman-diaconis')

    def __init__(self, method='uniform'):
        if method not in self._methods:
            raise ValueError(f'Accepted methods are {self._methods}, got {method} instead.')

        self.method = method
        self.X = None

    def transform(self, X, y=None, n_bins=None, min_samples_leaf=None):
        """Apply discretization to given data

        Args:
            X: 1-D array, The data to be descretized.
            y: 1-D array, The target values, ignored for unsupervised transformations.
            n_bins (:obj:`int ` or :obj:`str`): The number of bins; Either integer number or value in 
              {'square-root', 'sturges', 'rice-rule', 'scotts-rule', 'freedman-diaconis'}.
            min_samples_leaf (:obj:`int ` or :obj:`float`):  The minimum number of samples required to be at a leaf node. 
              Used for 'cart' method only, ignored otherwise.

        Returns:
            1-D array, The transformed data.

        """
        self.X = X

        # if isinstance(self.X, list):  # TODO add proper way to control input array type
        #     self.X = np.array(self.X)
        # elif not isinstance(self.X, ndarray):
        #     raise ValueError(f'Invalid array type'
        #                      f'Expected ndarray or list, got {type(self.X)} instead')

        if self.method in ['uniform', 'quantile', 'kmeans']:
            if not ((isinstance(n_bins, int) and n_bins > 1)
                    or n_bins in self._n_bins_formula):
                raise ValueError(f'Invalid number of bins. '
                                 f'Accepted integer value or one of the following options: {self._n_bins_formula},'
                                 f'got {n_bins} instead.')

            len_X = self.__check_data_shape()

            if n_bins in self._n_bins_formula:
                n_bins = self.__calculate_n_bins(n_bins, len_X)

            return KBinsDiscretizer._transform(self.X, n_bins, self.method)

        if self.method == 'cart':
            # if y is None or not isinstance(y, np.ndarray): # TODO add proper way to control input target type
            #     raise ValueError(f'Invalid target value. '
            #                      f'Expected 1-D array, got {y} instead')
            if len(y.shape) != 1 or y.shape[0] != self.X.shape[0]:
                raise ValueError(f'Invalid target shape. '
                                 f'Expected 1-D array with shape {(self.X.shape[0],)}, got {y.shape} instead')

            return CARTDiscretizer._transform(self.X, y, min_samples_leaf)

    def __calculate_n_bins(self, n_bins, len_X):
        """
        Calculate number of bins

        Args:
            X: 1-D array, data to be descretized
            n_bins: string, formula to calculate number of bins
            len_X: int, length of X

        Returns:
            int, number of bins
        """
        if n_bins == 'square-root':
            return round(sqrt(len_X))
        elif n_bins == 'sturges':
            return round(1 + 3.322 * log10(len_X))
        elif n_bins == 'rice-rule':
            return round(power(len_X, 1/3) * 2)
        elif n_bins == 'scotts-rule':
            return round(3.49 * std(self.X) / power(len_X, 1/3))
        elif n_bins == 'freedman-diaconis':
            iqr = subtract(*percentile(self.X, [75, 25]))
            return round(2 * iqr / power(len_X, 1/3))

    def __check_data_shape(self):
        """
        Check shape of X

        Args:
            X: 1-D array, data to be descretized

        Returns:
            int, length of X
        """
        if len(self.X.shape) not in (1, 2):
            raise ValueError('Expected 1D or 2D array, '
                             f'got shape={self.X.shape} instead.')

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)
            return self.X.shape[0]
        elif len(self.X.shape) == 2:
            if self.X.shape[0] == 1:
                self.X = self.X.reshape(-1, 1)

        return self.X.shape[0]
