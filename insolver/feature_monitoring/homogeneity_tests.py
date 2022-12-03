import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.preprocessing import LabelEncoder
from typing import Callable, List, Any
from .chi2_homogeneity_test import chi2_discr_2samp
from .psi_homogeneity_test import psi_discr_2samp, psi_cont_2samp, sec_min


def gen_sample(x: np.ndarray, samp_size: int, replace: bool = False) -> np.ndarray:
    """
    This function generates subsample of given size from main sample (without replaces by default).

    Parameters:
        x (np.array): main sample.
        samp_size (int): size of subsample.
        replace (bool): whether to return chosen subelements while after random choices or not.
        'False' by default.

    Returns:
        samp (np.array): subsample from the main sample.
    """

    samp = np.random.choice(x, samp_size, replace)
    return samp


def bootstrap(x1: np.ndarray, x2: np.ndarray, bootstrap_num: int, samp_size: int, test: Callable) -> float:
    """
    This function runs same test many times on subsamples of main 2 samples.
    Counted pvalues are used to get average estimate of pvalue. (Bootstrap idea).

    Parameters:
        x1 (np.array): sample from base period.
        x2 (np.array): sample from current period.
        bootstrap_num (int): number of times to run same test.
        samp_size (int): size of subsamples to use in bootstrap.
        test (callable): function which counts statistic and corresponding pvalue.

    Returns:
        pvalue (float): average estimate of pvalue (for x1, x2 homogeneity).
    """

    bootstrap_res = np.zeros(bootstrap_num, dtype=float)

    # generate subsamples
    for i in range(bootstrap_num):
        samp1, samp2 = gen_sample(x1, samp_size), gen_sample(x2, samp_size)
        test_res = test(samp1, samp2)
        pvalue = test_res.pvalue
        bootstrap_res[i] = pvalue

    pvalue = bootstrap_res.mean()
    return pvalue


def fillna_discr(x1_ref: np.ndarray, x2_ref: np.ndarray, inplace: bool = False) -> Any:
    """
    This function fills missing values in x1 and x2 safely for homogeneity tests.
    It guarantees that missing values will be filled with unique constant.

    Parameters:
        x1_ref (np.array): sample from base period.
        x2_ref (np.array): sample from current period.
        inplace (bool): whether to modify original x1, x2
        or to make copy and return it.

    Returns:
        x1 (np.array): sample from base period without missing values.
        x2 (np.array): sample from current period without missing values.
        nan_value (x1/x2.dtype): counted optimal nan value.
    """

    # copy inputs to avoid side effects
    if inplace:
        x1, x2 = x1_ref, x2_ref
    else:
        x1, x2 = x1_ref.copy(), x2_ref.copy()

    # if we have numeric data we define nan_value as (minimum - 1)
    if x1.dtype == 'int':
        nan_value = min(np.min(x1), np.min(x2)) - 1
        return x1, x2, nan_value
    if x1.dtype == 'float':
        nan_value = min(np.min(x1[~pd.isna(x1)]), np.min(x2[~pd.isna(x2)])) - 1
        x1[pd.isna(x1)] = nan_value
        x2[pd.isna(x2)] = nan_value
        return x1, x2, nan_value

    # if we have object data we fill nan with 'nan' str
    if x1.dtype == object:
        x1[pd.isna(x1)] = 'nan'
        x2[pd.isna(x2)] = 'nan'
        return x1, x2, 'nan'


def fillna_cont(x1_ref: np.ndarray, x2_ref: np.ndarray, inplace: bool = False) -> Any:
    """
    This function fills missing values in x1 and x2 safely for homogeneity tests.
    In case when nan value is just set to some constant less than all elements
    it can cause a big gap between nan value and the majority of samples.

    Parameters:
        x1_ref (np.array): sample from base period.
        x2_ref (np.array): sample from current period.
        inplace (bool): whether to modify original x1, x2
        or to make copy and return it.

    Returns:
        x1 (np.array): sample from base period without missing values.
        x2 (np.array): sample from current period without missing values.
        nan_value (x1/x2.dtype): counted optimal nan value.
    """

    # copy inputs to avoid side effects
    if inplace:
        x1, x2 = x1_ref, x2_ref
    else:
        x1, x2 = x1_ref.copy(), x2_ref.copy()

    # we fill nans with value less than all data
    # but it is smaller than minimum on gap between minimum and second minimum
    # it helps to avoid a lot of empty buckets in grids when running stat. tests
    min_ = min(np.min(x1[~pd.isna(x1)]), np.min(x2[~pd.isna(x2)]))

    sec_min1 = sec_min(x1[~pd.isna(x1)])
    sec_min2 = sec_min(x2[~pd.isna(x2)])

    sec_min_ = min(sec_min1, sec_min2)

    gap = sec_min_ - min_
    x1[pd.isna(x1)] = min_ - gap
    x2[pd.isna(x2)] = min_ - gap

    return x1, x2, min_ - gap


class DiscreteHomogeneityTests:
    """
    This class runs discrete homogeneity tests for two samples to check if feature has changed.
    Supported tests: 'chi2', 'psi'.
    'Chi2' is run with bootstrap, 'psi' is run without it.

    Parameters:
        pval_thresh (float): threshold for pvalue to use in tests.
        samp_size (int): size of subsamples to use during bootstrap.
        bootstrap_num (int): number of generating subsamples.
    """

    def __init__(self, pval_thresh: float, samp_size: int, bootstrap_num: int):
        """
        Raises:
            ValueError: if 'samp_size' is too small (< 100).
            ValueError: if threshold for pvalue is not between 0 and 1.
            ValueError: if given too small number of bootstrapping procedure.
        """

        # check for invalid parameters
        if samp_size < 100:
            raise ValueError("Too small value for 'samp_size' attribute.")

        if pval_thresh < 0 or pval_thresh > 1:
            raise ValueError("P-value threshold must be between 0 and 1")

        if bootstrap_num < 10:
            raise ValueError("Number of bootstrapping procedure should be not less than 10.")

        self.pval_thresh = pval_thresh
        self.samp_size = samp_size
        self.bootstrap_num = bootstrap_num

    def run_all(self, x1_ref: np.ndarray, x2_ref: np.ndarray, inplace: bool = False) -> List:
        """
        Runs all discrete tests for two samples: 'chi2', 'psi'.

        Parameters:
            x1_ref (np.array): sample from base period.
            x2_ref (np.array): sample from current period.
            inplace (bool): whether to modify original x1, x2
            or to make copy and return it.

        Returns:
            res (list of tuples): contains tuples of 3 elemets.
            The elements are name of the test, pvalue/psi_value
            and the conclusion about homogeneity.

        Raises:
            TypeError: if x1 or x2 is not a numpy array.
            TypeError: if x1 and x2 don't have same data type.
            TypeError: if x1 or x2 have unsupported data type (not int, float or object).
            ValueError: if size of x1 or x2 is smaller than bootstrap sample size 'samp_size'.
            Warning: if size of x1 or x2 is rather small to give robust estimations of pvalues.
        """

        # check for error situations
        if (not isinstance(x1_ref, np.ndarray)) or (not isinstance(x2_ref, np.ndarray)):
            raise TypeError("Only numpy.ndarray can be used as x1 and x2.")

        if x1_ref.dtype != x2_ref.dtype:
            raise TypeError("x1 and x2 must be of same data type.")

        if x1_ref.dtype not in [
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'float16',
            'float32',
            'float64',
            'object',
        ]:
            raise TypeError(f"Only int, float or object datatypes are supported as x1/x2.dtype. Got {x1_ref.dtype}.")

        if (x1_ref.shape[0] < self.samp_size) or (x2_ref.shape[0] < self.samp_size):
            raise ValueError("Sizes of x1 and x2 must be not less than 'samp_size' attribute.")

        if (x1_ref.shape[0] < self.samp_size * 1.7) or (x2_ref.shape[0] < self.samp_size * 1.7):
            raise Warning(
                "Sizes of x1 and x2 are better to be several times greater "
                + "than 'samp_size' to get more robust pvalue."
            )

        # copy inputs to avoid side effects
        if inplace:
            x1, x2 = x1_ref, x2_ref
        else:
            x1, x2 = x1_ref.copy(), x2_ref.copy()

        # fill nan values with special method to avoid collisions of category labels
        if np.any(pd.isna(x1)) or np.any(pd.isna(x2)):
            x1, x2, _ = fillna_discr(x1, x2, inplace=True)

        # encode categorical data with integer nums
        enc = LabelEncoder()
        enc.fit(np.concatenate([x1, x2]))
        x1, x2 = enc.transform(x1), enc.transform(x2)

        # manually run all tests
        res = []

        pvalue = bootstrap(x1, x2, self.bootstrap_num, self.samp_size, chi2_discr_2samp)
        conclusion = 'Different distributions' if pvalue < self.pval_thresh else 'Same distributions'
        res.append(('chi2', pvalue, conclusion))

        psi_value = psi_discr_2samp(x1, x2)

        if psi_value >= 0.2:
            conclusion = 'Different distributions'
        elif psi_value >= 0.1:
            conclusion = 'Small difference'
        else:
            conclusion = 'Same distributions'

        res.append(('psi', psi_value, conclusion))

        return res


class ContinuousHomogeneityTests:
    """
    This class runs continuous homogeneity tests for two samples to check if feature has changed.
    Supported tests: 'kolmogorov-smirnov',
    'cramer-von-mises', 'epps-singleton, 'psi'.
    All tests except 'psi' are run with bootstrap.

    Parameters:
        pval_thresh (float): threshold for pvalue to use in tests.
        samp_size (int): size of subsamples to use during boostrap.
        bootstrap_num (int): number of generating subsamples.
        psi_bins (int): number of bins or buckets for counting 'psi'.
    """

    def __init__(self, pval_thresh: float, samp_size: int, bootstrap_num: int, psi_bins: int = 20):
        """
        Raises:
            ValueError: if 'samp_size' is too small (< 100).
            ValueError: if threshold for pvalue is not between 0 and 1.
            ValueError: if given too small number of bootstrap procedure.
            ValueError: if number of bins for 'psi' is not 10 or 20.
        """

        # check for invalid parameters
        if samp_size < 100:
            raise ValueError("Too small value for 'samp_size' attribute.")

        if pval_thresh < 0 or pval_thresh > 1:
            raise ValueError("P-value threshold must be between 0 and 1")

        if bootstrap_num < 10:
            raise ValueError("Number of bootstrapping procedure should be not less than 10.")

        if psi_bins != 10 and psi_bins != 20:
            raise ValueError("Number of buckets for psi must be either 10 or 20.")

        self.pval_thresh = pval_thresh
        self.samp_size = samp_size
        self.bootstrap_num = bootstrap_num
        self.psi_bins = psi_bins

    def run_all(self, x1_ref: np.ndarray, x2_ref: np.ndarray, inplace: bool = False) -> List:
        """
        Runs all continuous tests for two samples: 'ks', 'cr-vonmis', 'epps-sing', 'psi'.

        Parameters:
            x1_ref (np.array): sample from base period.
            x2_ref (np.array): sample from current period.
            inplace (bool): whether to modify original x1, x2
            or to make copy and work with it.

        Returns:
            res (list of tuples): contains tuples of 3 elements.
            The elements are name of the test, pvalue/psi_value
            and the conclusion about homogeneity.

        Raises:
            TypeError: if x1 or x2 is not a numpy array.
            TypeError: if x1 and x2 don't have same data type.
            TypeError: if x1 or x2 have unsupported data type (not int, float or object).
            ValueError: if size of x1 or x2 is smaller than bootstrap sample size 'samp_size'.
            Warning: if size of x1 or x2 is rather small to give robust estimations of pvalues.
        """

        # check for error situations
        if (not isinstance(x1_ref, np.ndarray)) or (not isinstance(x2_ref, np.ndarray)):
            raise TypeError("Only numpy.ndarray can be used as x1 and x2.")

        if x1_ref.dtype != x2_ref.dtype:
            raise TypeError("x1 and x2 must be of same data type.")

        if x1_ref.dtype not in [
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'float16',
            'float32',
            'float64',
            'object',
        ]:
            raise TypeError(f"Only int, float or object datatypes are supported as x1/x2.dtype. Got {x1_ref.dtype}.")

        if (x1_ref.shape[0] < self.samp_size) or (x2_ref.shape[0] < self.samp_size):
            raise ValueError("Sizes of x1 and x2 must be not less than 'samp_size' attribute.")

        if (x1_ref.shape[0] < self.samp_size * 1.7) or (x2_ref.shape[0] < self.samp_size * 1.7):
            raise Warning(
                "Sizes of x1 and x2 are better to be several times greater "
                + "than 'samp_size' to get more robust pvalue."
            )

        # copy inputs to avoid side effects
        if inplace:
            x1, x2 = x1_ref, x2_ref
        else:
            x1, x2 = x1_ref.copy(), x2_ref.copy()

        # fill nan values with special method; usual 'fillna' don't fully suit homogeneity tests
        if np.any(pd.isna(x1)) or np.any(pd.isna(x2)):
            x1, x2, nan_value = fillna_cont(x1, x2, inplace=True)
        else:
            # this value will indicate psi that there are no nans
            nan_value = min(np.min(x1), np.min(x2)) - 1

        # manually run all tests
        res = []

        pvalue = bootstrap(x1, x2, self.bootstrap_num, self.samp_size, sps.ks_2samp)
        conclusion = 'Different distributions' if pvalue < self.pval_thresh else 'Same distributions'
        res.append(('ks', pvalue, conclusion))

        pvalue = bootstrap(x1, x2, self.bootstrap_num, self.samp_size, sps.cramervonmises_2samp)
        conclusion = 'Different distributions' if pvalue < self.pval_thresh else 'Same distributions'
        res.append(('crvonmis', pvalue, conclusion))

        pvalue = bootstrap(x1, x2, self.bootstrap_num, self.samp_size, sps.epps_singleton_2samp)
        conclusion = 'Different distributions' if pvalue < self.pval_thresh else 'Same distributions'
        res.append(('epps-sing', pvalue, conclusion))

        psi_value = psi_cont_2samp(x1, x2, nan_value=nan_value, buckets=self.psi_bins)

        if psi_value >= 0.2:
            conclusion = 'Different distributions'
        elif psi_value >= 0.1:
            conclusion = 'Small difference'
        else:
            conclusion = 'Same distributions'

        res.append(('psi', psi_value, conclusion))

        return res
