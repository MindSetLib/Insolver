import numpy as np
from pandas import DataFrame, Series
from numba import jit
from typing import Union


@jit(cache=True, nopython=True)
def gen_prices(
    price: Union[int, float], lower_bound: Union[int, float], upper_upper: Union[int, float], step: float
) -> np.ndarray:
    percents = np.arange(lower_bound, upper_upper, step)
    prices = price * percents
    return prices


@jit(cache=True, nopython=True)
def filter_candidates(
    prices: np.ndarray,
    minimum: Union[None, int, float] = None,
    maximum: Union[None, int, float] = None,
    frac_min: Union[int, float] = 1,
    frac_max: Union[int, float] = 1,
) -> np.ndarray:
    min_ = np.min(prices) if not minimum else minimum
    max_ = np.max(prices) if not maximum else maximum
    bound_filtered = prices[(prices >= frac_min * min_) & (prices <= frac_max * max_)]
    if len(bound_filtered) == 0:
        raise ValueError('Filters is too restrictive: no candidates left.')
    else:
        return bound_filtered


def gen_potential_prices(
    entity: Series,
    price_name: str,
    lower_bound: Union[float, int] = 0.25,
    upper_upper: Union[float, int] = 2.05,
    step: Union[float, int] = 0.05,
    decimals: int = 2,
    filter_minimum: Union[None, str, int, float] = None,
    filter_maximum: Union[None, str, int, float] = None,
    filter_frac_min: Union[int, float] = 1,
    filter_frac_max: Union[int, float] = 1,
    dtypes: Union[None, dict] = None,
) -> DataFrame:
    prices = gen_prices(entity[price_name], lower_bound, upper_upper, step)
    if filter_minimum or filter_maximum:
        if isinstance(filter_minimum, str):
            filter_minimum = entity[filter_minimum]
        if isinstance(filter_maximum, str):
            filter_maximum = entity[filter_maximum]
        prices_filtered = filter_candidates(
            prices, minimum=filter_minimum, maximum=filter_maximum, frac_min=filter_frac_min, frac_max=filter_frac_max
        )
    else:
        prices_filtered = prices
    c_df = np.column_stack((np.round(prices_filtered, decimals), np.vstack([entity.to_numpy()] * len(prices_filtered))))
    df_c = DataFrame(c_df, columns=[price_name, *[x if x != price_name else 'orig_premium' for x in entity.index]])
    if dtypes:
        df_c['orig_premium'] = df_c['orig_premium'].astype(dtypes[price_name])
        df_c = df_c.astype(dtypes)
    return df_c


def candidates(
    df: DataFrame,
    price_name: str,
    lower_bound: Union[float, int] = 0.25,
    upper_upper: Union[float, int] = 2.05,
    step: Union[float, int] = 0.05,
    decimals: int = 2,
    filter_minimum: Union[None, str, int, float] = None,
    filter_maximum: Union[None, str, int, float] = None,
    filter_frac_min: Union[int, float] = 1.0,
    filter_frac_max: Union[int, float] = 1.0,
) -> Series:
    if (df[price_name].isnull().sum() == 0) & ((df[price_name] == 0).sum() == 0):
        return df.apply(
            lambda x: gen_potential_prices(
                entity=x,
                price_name=price_name,
                lower_bound=lower_bound,
                upper_upper=upper_upper,
                step=step,
                decimals=decimals,
                filter_minimum=filter_minimum,
                filter_maximum=filter_maximum,
                filter_frac_min=filter_frac_min,
                filter_frac_max=filter_frac_max,
                dtypes=df.dtypes.to_dict(),
            ),
            axis=1,
        )
    else:
        raise ValueError(f"Dataframe contains empty or zero values in price column '{price_name}'!")
