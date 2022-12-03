import numpy as np
from scipy import stats as sps
from collections import defaultdict


class Chi2Result:
    """
    This class is made for returning result of chi-square test in scipy style
    (like a structure with two named fields).

    Parameters:
        statistic (float): value of counted chi-square statistic.
        pvalue (float): pvalue corresponding to this statistic.
    """

    def __init__(self, statistic: float, pvalue: float):
        self.statistic = statistic
        self.pvalue = pvalue


def chi2_discr_2samp(x1: np.ndarray, x2: np.ndarray) -> "Chi2Result":
    """
    This function runs chi-square test checking homogeneity of two samples
    of discrete variables.

    Parameters:
        x1 (np.array): sample from base period.
        x2 (np.array): sample from current period.

    Returns:
        res (Chi2Result): object containing counted statistic and corresponding pvalue.
    """

    n1, n2 = len(x1), len(x2)

    # find unique categories and their frequencies in both arrays
    cats1, counts1 = np.unique(x1, return_counts=True)
    counts1 = defaultdict(int, zip(cats1, counts1))

    cats2, counts2 = np.unique(x2, return_counts=True)
    counts2 = defaultdict(int, zip(cats2, counts2))

    cats = np.union1d(cats1, cats2)
    num_cats = len(cats)

    # if both samples consist of only one constant
    # value we consider statistic to be zero
    if num_cats == 1:
        return Chi2Result(0.0, 1.0)

    # calculate statistic
    chi2 = 0.0
    for cat in cats:
        mu_i = counts1[cat]
        nu_i = counts2[cat]
        chi2 += ((mu_i / n1 - nu_i / n2) ** 2) / (mu_i + nu_i)
    chi2 *= n1 * n2

    # count pvalue
    pvalue = 1 - sps.chi2.cdf(chi2, num_cats - 1)
    res = Chi2Result(chi2, pvalue)
    return res
