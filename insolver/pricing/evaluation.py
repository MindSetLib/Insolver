import numpy as np
from pandas import DataFrame, Series
from typing import Union, Any, Iterable


def price_eval(
    x: DataFrame,
    model: Any,
    feature_names: Iterable,
    burning_cost_pct: Union[float, int] = 0.8,
    threshold: Union[float, int] = 0.5,
) -> DataFrame:
    if not (hasattr(model, 'predict_proba') and callable(model.predict_proba)):
        raise ValueError("Model has no predict_proba() method.")
    prices = x.to_numpy()[:, 0]
    old_price = x['orig_premium'].to_numpy()
    pred = model.predict_proba(x[feature_names])[:, 1]
    price_name = x.columns[0]
    x_orig = x.copy().drop(price_name, axis=1).rename({'orig_premium': price_name}, axis=1)
    pred_orig = model.predict_proba(x_orig[feature_names])[:, 1]

    profit = pred * prices * (1 - burning_cost_pct * old_price / prices)
    profit_orig = pred_orig * old_price * (1 - burning_cost_pct)

    act_profit = profit * (pred >= threshold)
    act_profit_orig = profit_orig * (pred_orig >= threshold)

    df = DataFrame(np.dstack((prices, old_price, pred, pred_orig, profit, profit_orig, act_profit, act_profit_orig))[0])
    df.columns = ['price', 'orig_price', 'pred', 'orig_pred', 'profit', 'profit_orig', 'act_profit', 'act_profit_orig']
    return df


def eval_candidate(
    df: DataFrame,
    model: Any,
    feature_names: Iterable,
    burning_cost_pct: Union[float, int] = 0.8,
    threshold: Union[float, int] = 0.5,
) -> Series:
    return df.apply(
        lambda x: price_eval(x, model, feature_names, burning_cost_pct=burning_cost_pct, threshold=threshold)
    )
