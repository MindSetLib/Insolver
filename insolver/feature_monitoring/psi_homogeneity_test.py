import numpy as np
from math import inf
from collections import defaultdict
from typing import Iterable, Union


def sec_min(x: Iterable) -> Union[float, int]:
    """
    This function counts second minimum of an array.

    Parameters:
        x (iterable): array for counting minimum. Must support finding min.

    Returns:
        sec_min (float or int): second minimum after main.
    """

    min1 = min(x)
    min2 = inf
    for val in x:
        if val == min1:
            continue
        min2 = min(min2, val)
    return min2


def psi_cont_2samp(x1: np.ndarray, x2: np.ndarray, nan_value: float = -1.0, buckets: int = 20) -> float:
    """
    This function counts population stability index (PSI)
    between two samples of continuous variables.

    Parameters:
        x1 (np.array): sample from base period.
        x2 (np.array): sample from current period.
        nan_value (float): value used to fill nans in arrays. Must be the smallest element in each array.
        buckets (int): number of bins for calculating psi. 20 by default.

    Returns:
        psi_value (float): psi between x1 and x2.

    Raises:
        ValueError: if x1 or x2 contain elements smaller than 'nan_value'.
    """

    if (np.min(x1) < nan_value) or (np.min(x2) < nan_value):
        raise ValueError("Elements of x1 and x2 can't be smaller than 'nan_value' for counting psi.")

    # build grid for histograms
    min_ = min(np.min(x1), np.min(x2))
    max_ = max(np.max(x1), np.max(x2))
    grid = np.array([])
    if min_ > nan_value:
        grid = np.linspace(min_, max_, buckets + 1)
    elif min_ == nan_value:
        sec_min1 = sec_min(x1)
        sec_min2 = sec_min(x2)
        sec_min_ = min(sec_min1, sec_min2)
        main_grid = np.linspace(sec_min_, max_, buckets)
        grid = np.concatenate([[nan_value], main_grid])

    # count number of elements in buckets
    old_percents = np.histogram(x1, grid)[0] / len(x1)
    new_percents = np.histogram(x2, grid)[0] / len(x2)

    # fill empty buckets with nonzero value (to avoid zero-division)
    old_percents[old_percents == 0] = 1e-4
    new_percents[new_percents == 0] = 1e-4

    # resulting psi
    psi_value = (old_percents - new_percents) @ np.log(old_percents / new_percents)
    return psi_value


def psi_discr_2samp(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    This function counts psi_value between two samples of discrete variables.

    Parameters:
        x1 (np.array): sample from base period.
        x2 (np.array): sample from current period.

    Returns:
        psi_value (float): psi between x1 and x2.
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
        return 0.0

    # count frequencies for each category
    old_percents = np.zeros([num_cats], dtype=float)
    new_percents = np.zeros([num_cats], dtype=float)

    for i, cat in enumerate(cats):
        old_percents[i] = counts1[cat] / n1
        new_percents[i] = counts2[cat] / n2

    # fill empty buckets with nonzero value (to avoid zero-division)
    old_percents[old_percents == 0] = 1e-4
    new_percents[new_percents == 0] = 1e-4

    # resulting psi
    psi_value = (old_percents - new_percents) @ np.log(old_percents / new_percents)
    return psi_value
