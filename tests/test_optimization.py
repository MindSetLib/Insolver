import pytest
import numpy as np
from pandas import DataFrame, Series
from insolver.pricing.optimization import maximize, max_profit, max_conversion


# Test max_profit function
def test_max_profit():
    x = np.array([[100, 80, 0.7, 0.6], [150, 100, 0.8, 0.5], [120, 90, 0.6, 0.7]])
    expected_result = np.array([[150.0, 100.0, 0.8, 0.5]])
    assert np.array_equal(max_profit(x), expected_result)


def test_max_profit2():
    x = np.array([[100, 80, 0.7, 0.6], [150, 100, 0.7, 0.5], [120, 90, 0.7, 0.7]])
    expected_result = np.array([[100, 80, 0.7, 0.6]])
    assert np.array_equal(max_profit(x), expected_result)


# Test max_conversion function
def test_max_conversion():
    df = DataFrame(
        {
            'price': [95.0, 100.0, 105.0],
            'orig_price': [100.0, 100.0, 100.0],
            'pred': [0.71, 0.68, 0.28],
            'orig_pred': [0.72, 0.57, 0.56],
            'profit': [10.68, 13.68, 7.14],
            'profit_orig': [14.59, 11.55, 11.30],
            'act_profit': [10.68, 13.68, 0.0],
            'act_profit_orig': [14.59, 11.55, 11.30],
        }
    )
    expected_result = np.array([[100.0, 100.0, 0.68, 0.57, 13.68, 11.55, 13.68, 11.55]])
    assert np.array_equal(max_conversion(df.to_numpy(dtype=float)), expected_result)


# Test max_conversion function
def test_max_conversion2():
    df = DataFrame(
        {
            'price': [95.0, 101.0, 105.0],
            'orig_price': [100.0, 100.0, 100.0],
            'pred': [0.71, 0.68, 0.28],
            'orig_pred': [0.72, 0.49, 0.56],
            'profit': [10.68, 13.68, 7.14],
            'profit_orig': [14.59, 11.55, 11.30],
            'act_profit': [10.68, 13.68, 0.0],
            'act_profit_orig': [14.59, 11.55, 11.30],
        }
    )
    expected_result = np.array([[101.0, 100.0, 0.68, 0.49, 13.68, 11.55, 13.68, 11.55]])
    assert np.array_equal(max_conversion(df.to_numpy(dtype=float)), expected_result)


# Test max_conversion function
def test_max_conversion3():
    df = DataFrame(
        {
            'price': [95.0, 100.0, 105.0],
            'orig_price': [100.0, 100.0, 100.0],
            'pred': [0.49, 0.49, 0.28],
            'orig_pred': [0.49, 0.49, 0.49],
            'profit': [10.68, 13.68, 7.14],
            'profit_orig': [14.59, 11.55, 11.30],
            'act_profit': [10.68, 13.68, 0.0],
            'act_profit_orig': [14.59, 11.55, 11.30],
        }
    )
    expected_result = np.array([[100.0, 100.0, 0.49, 0.49, 13.68, 11.55, 13.68, 11.55]])
    assert np.array_equal(max_conversion(df.to_numpy(dtype=float)), expected_result)


# Test maximize function with method='profit'
def test_maximize_profit():
    df = DataFrame(
        {
            'price': [95.0, 100.0, 105.0],
            'orig_price': [100.0, 100.0, 100.0],
            'pred': [0.71, 0.68, 0.28],
            'orig_pred': [0.72, 0.57, 0.56],
            'profit': [10.68, 13.68, 7.14],
            'profit_orig': [14.59, 11.55, 11.30],
            'act_profit': [10.68, 13.68, 0.0],
            'act_profit_orig': [14.59, 11.55, 11.30],
        }
    )
    ser = Series([df])
    result = maximize(ser, method='profit')
    assert isinstance(result, DataFrame)
    assert result.shape == (1, 8)


# Test maximize function with method='conversion'
def test_maximize_conversion():
    df = DataFrame(
        {
            'price': [95.0, 100.0, 105.0],
            'orig_price': [100.0, 100.0, 100.0],
            'pred': [0.71, 0.68, 0.28],
            'orig_pred': [0.72, 0.57, 0.56],
            'profit': [10.68, 13.68, 7.14],
            'profit_orig': [14.59, 11.55, 11.30],
            'act_profit': [10.68, 13.68, 0.0],
            'act_profit_orig': [14.59, 11.55, 11.30],
        }
    )
    ser = Series([df])
    result = maximize(ser, method='conversion')
    assert isinstance(result, DataFrame)
    assert result.shape == (1, 8)


# Test maximize function with invalid method
def test_maximize_invalid_method():
    df = DataFrame({'price': [100, 150], 'orig_price': [80, 100], 'pred': [0.7, 0.8], 'profit': [0.6, 0.5]})
    with pytest.raises(ValueError):
        maximize(df, method='invalid_method')
