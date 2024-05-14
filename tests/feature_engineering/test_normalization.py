import pytest
import numpy as np
import pandas as pd
from insolver.feature_engineering import Normalization
from numpy.testing import assert_array_equal


@pytest.fixture
def sample_data():
    return pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 6, 7, 8, 9], 'C': [10, 11, 12, 13, 14]})


def test_methods(sample_data):
    for method in ['standard', 'minmax', 'robust', 'normalizer', 'yeo-johnson', 'box-cox', 'log', 'invalid']:
        normalizer = Normalization(method=method)
        if method == 'invalid':
            with pytest.raises(NotImplementedError):
                normalizer.transform(sample_data)
        else:
            normalizer.transform(sample_data)


def test_transform_method(sample_data):
    res = sample_data.copy()
    res['A'] = (res['A'] - np.mean(res['A'])) / np.std(res['A'])
    res['B'] = (res['B'] - np.mean(res['B'])) / np.std(res['B'])
    res['C'] = (res['C'] - np.mean(res['C'])) / np.std(res['C'])

    normalizer = Normalization(method='standard')
    transformed_data = normalizer.transform(sample_data)
    assert transformed_data.equals(res)

    transformed_data = normalizer.transform(sample_data.values)
    transformed_data.columns = res.columns
    assert transformed_data.equals(res)


def test_transform_method_specific_columns(sample_data):
    normalizer = Normalization(method='minmax', column_names=['A', 'B'])
    transformed_data = normalizer.transform(sample_data)
    assert_array_equal(transformed_data['A'], np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert_array_equal(transformed_data['B'], np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    normalizer = Normalization(method='minmax', column_names='A')
    transformed_data = normalizer.transform(sample_data)
    assert_array_equal(transformed_data['A'], np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_transform_method_before_data(sample_data):
    normalizer = Normalization(method='standard')
    with pytest.raises(Exception):
        normalizer.plot_transformed('A')


def test_plot_transformed(sample_data):
    normalizer = Normalization(method='standard')
    normalizer.transform(sample_data)
    normalizer.plot_transformed('A')


def test_duplicate_columns(sample_data):
    column_method = {'A': 'minmax'}
    normalizer = Normalization(method='standard', column_names='A', column_method=column_method)
    with pytest.raises(ValueError):
        normalizer.transform(sample_data)


def test_multiple_columns(sample_data):
    column_method = {'A': 'minmax'}
    normalizer = Normalization(method='standard', column_names='B', column_method=column_method)
    transformed_data = normalizer.transform(sample_data)
    assert_array_equal(transformed_data['A'], np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    column_method = {'A': 'standard'}
    normalizer = Normalization(method='minmax', column_method=column_method)
    transformed_data = normalizer.transform(sample_data)
    assert_array_equal(transformed_data['B'], np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
