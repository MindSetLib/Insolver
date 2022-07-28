from pandas import DataFrame, Series
from numpy import log10, log2, std, percentile, subtract, sqrt, power, ndarray, array, min as npmin, max as npmax
from insolver.discretization.discretizer_utils import SklearnDiscretizer, CARTDiscretizer, ChiMergeDiscretizer


class InsolverDiscretizer:
    """Trasform continuous variable into discrete form.

    Parameters:
        method (str): The method used to discretize. Should be in {'uniform', 'quantile', 'kmeans', 'cart'}.

    """

    _methods = ['uniform', 'quantile', 'kmeans', 'cart', 'chimerge']

    _n_bins_formula = (
        'square-root',
        'sturges',
        'huntsberger',
        'brooks-carrther',
        'cencov',
        'rice-rule',
        'terrell-scott',
        'scott',
        'freedman-diaconis',
    )

    def __init__(self, method='uniform'):
        if method not in self._methods:
            raise ValueError(f'Accepted methods are {self._methods}, got {method} instead.')

        self.method = method
        self.X = None

    def transform(self, X, y=None, n_bins=None, min_samples_leaf=None):
        """Apply discretization to given data.

        Args:
            X: 1-D array, The data to be descretized.
            y: 1-D array, The target values, ignored for unsupervised transformations.
            n_bins (int, str): The number of bins; Either integer number or value in
              {'square-root', 'sturges', 'rice-rule', 'scotts-rule', 'freedman-diaconis'}.
            min_samples_leaf (int, float):  The minimum number of samples required to be at a leaf
            node. Used for 'cart' method only, ignored otherwise.

        Returns:
            1-D array, The transformed data.

        Examples:

        Unsupervised discretization

        >>> import numpy as np
        >>> from insolver.discretization import InsolverDiscretizer
        >>> X = np.array([85, 90, 78, 96, 80, 70, 65, 95])
        >>> insolverDisc = InsolverDiscretizer(method='uniform')
        >>> insolverDisc.transform(X, n_bins=3)
        array([1., 2., 1., 2., 1., 0., 0., 2.])

        Supervised discretization

        >>> import numpy as np
        >>> from insolver.discretization import InsolverDiscretizer
        >>> X = np.array([85, 90, 78, 96, 80, 70, 65, 95])
        >>> y = np.array([1, 0, 1, 0, 0, 1, 1, 1])
        >>> insolverDisc = InsolverDiscretizer(method='chimerge')
        >>> insolverDisc.transform(X, y, n_bins=3)
        array([1, 1, 0, 2, 1, 0, 0, 1], dtype=int64)

        """

        self.X = X
        self.__check_X_type()

        if self.method in ['uniform', 'quantile', 'kmeans']:
            if not ((isinstance(n_bins, int) and n_bins > 1) or n_bins in self._n_bins_formula):
                raise ValueError(
                    'Invalid number of bins. '
                    f'Accepted integer value or one of the following options: {self._n_bins_formula},'
                    f'got {n_bins} instead.'
                )

            len_X = self.__check_X_shape()

            if n_bins in self._n_bins_formula:
                n_bins = self.__calculate_n_bins(n_bins, len_X)

            return SklearnDiscretizer._transform(self.X, n_bins, self.method)

        if self.method == 'cart':
            y = self.__check_y(y)
            return CARTDiscretizer._transform(self.X, y, min_samples_leaf)

        if self.method == 'chimerge':
            y = self.__check_y(y)
            return ChiMergeDiscretizer()._transform(self.X, y, n_bins)

    def __check_y(self, y):
        """Check y type."""
        if isinstance(y, DataFrame):
            y = y.values.reshape(-1)
        elif isinstance(y, Series):
            y = y.values
        elif isinstance(y, list):
            y = array(y)
        elif not (isinstance(y, ndarray)):
            raise ValueError(
                'Invalid target type. '
                'Accepted pandas DataFrame and Series instancies, list and numpy array, got '
                f'{type(y)} instead.'
            )

        if (not y.shape != (len(y), 1) or len(y.shape) != 1) and len(y) != self.X.shape[0]:
            raise ValueError(
                'Invalid target shape. Expected 1-D array with shape '
                f'{(self.X.shape[0],)} or {(self.X.shape[0], 1)}, got {y.shape} instead'
            )

        return y

    def __calculate_n_bins(self, n_bins, len_X):
        """Calculate number of bins.

        Args:
            n_bins(string): The formula to calculate number of bins.
            len_X(int): length of X.

        Returns:
            int, The number of bins.

        References:
            Cebeci, Z. and Yıldız, F. (2017) Unsupervised Discretization of Continuous Variables in a Chicken Egg
            Quality Traits Dataset. Turkish Journal of Agriculture-Food Science and Technology, 5.4, 315-320.
            Available from: http://agrifoodscience.com/index.php/TURJAF/article/view/1056
        """
        if n_bins == 'square-root':
            return round(sqrt(len_X))

        elif n_bins == 'sturges':
            return round(1 + log2(len_X))

        elif n_bins == 'huntsberger':
            return round(1 + 3.322 * log10(len_X))

        elif n_bins == 'brooks-carrther':
            return round(5 * log10(len_X))

        elif n_bins == 'cencov':
            return round(pow(len_X, 1 / 3))

        elif n_bins == 'rice-rule':
            return round(power(len_X, 1 / 3) * 2)

        elif n_bins == 'terrell-scott':
            return round(power(2 * len_X, 1 / 3))

        elif n_bins == 'scott':
            return round((npmax(self.X) - npmin(self.X)) / 3.5 * std(self.X) * power(len_X, -1 / 3))

        elif n_bins == 'freedman-diaconis':
            iqr = subtract(*percentile(self.X, [75, 25]))
            h = 2 * iqr / power(len_X, 1 / 3)
            return round((npmax(self.X) - npmin(self.X)) / h)

    def __check_X_shape(self):
        """Check shape of X.


        Returns:
            int, length of X.
        """

        if len(self.X.shape) not in (1, 2):
            raise ValueError(f'Expected 1D or 2D array, got shape={self.X.shape} instead.')

        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)
            return self.X.shape[0]
        elif len(self.X.shape) == 2:
            if self.X.shape[0] == 1:
                self.X = self.X.reshape(-1, 1)

        return self.X.shape[0]

    def __check_X_type(self):
        """Check X type."""
        if isinstance(self.X, DataFrame) or isinstance(self.X, Series):
            self.X = self.X.values
        elif isinstance(self.X, list):
            self.X = array(self.X)
        elif not (isinstance(self.X, ndarray)):
            raise ValueError(
                'Invalid data type. '
                'Accepted pandas DataFrame and Series instancies, list and numpy array, got '
                f'{type(self.X)} instead.'
            )
