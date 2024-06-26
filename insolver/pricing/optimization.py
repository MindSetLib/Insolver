import numpy as np
from pandas import DataFrame, concat
from numba import jit
from typing import Union, Literal


@jit(cache=True, nopython=True)
def max_profit(x: np.ndarray) -> np.ndarray:
    max_profit_ = x[x[:, -2] == x[:, -2].max()]  # [:,-2] for profit
    if max_profit_.shape[0] > 1:
        max_profit_ = max_profit_[max_profit_[:, 2] == max_profit_[:, 2].max()]  # [:,2] for pred
        if max_profit_.shape[0] > 1:
            max_profit_ = max_profit_[max_profit_[:, 0] == max_profit_[:, 0].min()]  # [:,0] for price
    return max_profit_


@jit(cache=True, nopython=True)
def max_conversion(x: np.ndarray, threshold: Union[float, int] = 0.5) -> np.ndarray:
    orig = x[np.abs(x[:, 0] - x[:, 1].max()) < 0.01]  # [:, 0] for price, [:, 1] for orig_price
    if orig.shape[0] == 0:
        orig = max_profit(x)
    if x[:, 3].min() >= threshold:  # [:, 3] for orig_pred
        choice = orig
    else:
        converted = x[x[:, 2] >= threshold]
        if converted.shape[0] == 0:
            choice = orig
        else:
            choice = max_profit(converted)
    return choice


def maximize(
    df: DataFrame, method: Literal['profit', 'conversion'] = 'profit', threshold: Union[float, int] = 0.5
) -> DataFrame:
    if method == 'profit':
        res = df.apply(lambda x: DataFrame(max_profit(x.to_numpy(dtype=float)), columns=x.columns))
    elif method == 'conversion':
        res = df.apply(
            lambda x: DataFrame(max_conversion(x.to_numpy(dtype=float), threshold=threshold), columns=x.columns)
        )
    else:
        raise ValueError('method should be one of ["profit", "conversion"]')
    result = concat(res.to_numpy())
    result.index = df.index
    return result
