import pytest
import numpy as np
from pandas import DataFrame, Series
from insolver.pricing.generation import (
    candidates,
    filter_candidates,
    gen_potential_prices,
    gen_prices,
)


# Test gen_prices function
def test_gen_prices():
    price = 100
    lower_bound = 0.5
    upper_upper = 2.0
    step = 0.1
    expected_prices = np.array(
        [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    )
    assert np.allclose(gen_prices(price, lower_bound, upper_upper, step), expected_prices)


# Test filter_candidates function
def test_filter_candidates():
    prices = np.array(
        [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    )
    filtered_prices = filter_candidates(prices, minimum=80, maximum=300, frac_min=1, frac_max=0.5)
    expected_filtered_prices = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    assert np.allclose(filtered_prices, expected_filtered_prices)


def test_filter_candidates2():
    prices = np.array(
        [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    )
    filtered_prices = filter_candidates(prices)
    expected_filtered_prices = np.array(
        [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    )
    assert np.allclose(filtered_prices, expected_filtered_prices)


def test_filter_candidates3():
    prices = np.array(
        [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    )
    with pytest.raises(ValueError):
        filter_candidates(prices, minimum=200)


# Test gen_potential_prices function
def test_gen_potential_prices():
    entity = Series({'price': 100, 'foo': True, 'bar': 'text'})
    price_name = 'price'
    lower_bound = 0.5
    upper_upper = 2.0
    step = 0.1
    decimals = 2
    filter_minimum = 160
    filter_maximum = 300
    filter_frac_min = 0.5
    filter_frac_max = 0.5
    expected_df = DataFrame(
        {
            'price': [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            'orig_premium': [100] * 8,
            'foo': [True] * 8,
            'bar': ['text'] * 8,
        }
    )
    assert gen_potential_prices(
        entity,
        price_name,
        lower_bound,
        upper_upper,
        step,
        decimals,
        filter_minimum,
        filter_maximum,
        filter_frac_min,
        filter_frac_max,
        dtypes=expected_df.dtypes.to_dict(),
    ).equals(expected_df)


def test_gen_potential_prices2():
    entity = Series({'price': 100, 'foo': True, 'bar': 'text'})
    price_name = 'price'
    lower_bound = 0.5
    upper_upper = 2.0
    step = 0.1
    decimals = 2
    expected_df = DataFrame(
        {
            'price': [
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
                150.0,
                160.0,
                170.0,
                180.0,
                190.0,
            ],
            'orig_premium': [100] * 15,
            'foo': [True] * 15,
            'bar': ['text'] * 15,
        }
    )
    assert gen_potential_prices(
        entity, price_name, lower_bound, upper_upper, step, decimals, dtypes=expected_df.dtypes.to_dict()
    ).equals(expected_df)


def test_gen_potential_prices3():
    entity = Series({'price': 100.0, 'foo': 300, 'bar': 80})
    price_name = 'price'
    lower_bound = 0.5
    upper_upper = 2.0
    step = 0.1
    decimals = 2
    filter_minimum = 'bar'
    filter_maximum = 'foo'
    filter_frac_min = 1
    filter_frac_max = 0.5
    expected_df = DataFrame(
        {
            'price': [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            'orig_premium': [100.0] * 8,
            'foo': [300.0] * 8,
            'bar': [80.0] * 8,
        }
    )
    assert gen_potential_prices(
        entity,
        price_name,
        lower_bound,
        upper_upper,
        step,
        decimals,
        filter_minimum,
        filter_maximum,
        filter_frac_min,
        filter_frac_max,
    ).equals(expected_df)


# Test candidates function
def test_candidates():
    df = DataFrame({'price': [100, 150], 'foo': [True, None], 'bar': [-2.0, 'text']})
    price_name = 'price'
    lower_bound = 0.5
    upper_upper = 2.0
    filter_maximum = 275
    step = 0.1
    decimals = 2
    expected_series = Series(
        [
            DataFrame(
                {
                    'price': np.arange(50, 200, 10),
                    'orig_premium': [100] * 15,
                    'foo': [True] * 15,
                    'bar': [-2.0] * 15,
                }
            ).astype({'price': 'int64', 'foo': 'object', 'bar': 'object'}),
            DataFrame(
                {
                    'price': np.arange(75, 275, 15),
                    'orig_premium': [150] * 14,
                    'foo': [None] * 14,
                    'bar': ['text'] * 14,
                },
            ).astype({'price': 'int64', 'foo': 'object', 'bar': 'object'}),
        ]
    )
    results = candidates(
        df,
        price_name=price_name,
        lower_bound=lower_bound,
        upper_upper=upper_upper,
        step=step,
        decimals=decimals,
        filter_maximum=filter_maximum,
    )
    for i in range(len(results)):
        assert results.iloc[i].equals(expected_series.iloc[i])


def test_candidates2():
    df = DataFrame({'price': [None, 150], 'foo': [True, None], 'bar': [-2.0, 'text']})
    price_name = 'price'
    lower_bound = 0.5
    upper_upper = 2.0
    filter_maximum = 275
    step = 0.1
    decimals = 2
    with pytest.raises(ValueError):
        candidates(
            df,
            price_name=price_name,
            lower_bound=lower_bound,
            upper_upper=upper_upper,
            step=step,
            decimals=decimals,
            filter_maximum=filter_maximum,
        )


def test_candidates3():
    df = DataFrame({'price': [150, 0], 'foo': [True, None], 'bar': [-2.0, 'text']})
    price_name = 'price'
    lower_bound = 0.5
    upper_upper = 2.0
    filter_maximum = 275
    step = 0.1
    decimals = 2
    with pytest.raises(ValueError):
        candidates(
            df,
            price_name=price_name,
            lower_bound=lower_bound,
            upper_upper=upper_upper,
            step=step,
            decimals=decimals,
            filter_maximum=filter_maximum,
        )
