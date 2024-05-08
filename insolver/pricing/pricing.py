from pandas import DataFrame
from typing import Any, Literal, Union, Iterable

from .generation import candidates
from .evaluation import eval_candidate
from .optimization import maximize


def dynamic_price(
    df: DataFrame,
    price_name: str,
    model: Any,
    feature_names: Iterable,
    burning_cost_pct: Union[float, int] = 0.8,
    threshold: Union[float, int] = 0.5,
    method: Literal['profit', 'conversion'] = 'profit',
    lower_bound: Union[float, int] = 0.25,
    upper_upper: Union[float, int] = 2.05,
    step: Union[float, int] = 0.05,
    decimals: int = 2,
    filter_minimum: Union[None, str, int, float] = None,
    filter_maximum: Union[None, str, int, float] = None,
    filter_frac_min: Union[float, int] = 1,
    filter_frac_max: Union[float, int] = 1,
) -> DataFrame:
    _df = eval_candidate(
        candidates(
            df,
            price_name=price_name,
            lower_bound=lower_bound,
            upper_upper=upper_upper,
            step=step,
            decimals=decimals,
            filter_minimum=filter_minimum,
            filter_maximum=filter_maximum,
            filter_frac_min=filter_frac_min,
            filter_frac_max=filter_frac_max,
        ),
        model,
        feature_names,
        burning_cost_pct=burning_cost_pct,
        threshold=threshold,
    )
    return maximize(_df, method=method, threshold=threshold)
