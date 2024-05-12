import pytest
import pandas as pd
from insolver.feature_engineering import Smoothing


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            'x': range(10),
            'y': range(10, 20),
        }
    )


def test_transform_moving_average(sample_data):
    sm = Smoothing(method='moving_average', x_column='x', window=3)
    transformed_data = sm.transform(sample_data)
    assert 'x_Moving_Average' in transformed_data.columns
    assert len(transformed_data) == len(sample_data)


def test_transform_moving_average2(sample_data):
    sm = Smoothing(method='moving_average', window=3)
    transformed_data = sm.transform(sample_data['x'].values)
    assert 'data_Moving_Average' in transformed_data.columns
    assert len(transformed_data) == len(sample_data)


def test_transform_lowess(sample_data):
    sm = Smoothing(method='lowess', x_column='x', y_column='y')
    transformed_data = sm.transform(sample_data)
    assert 'x_Lowess' in transformed_data.columns
    assert 'y_Lowess' in transformed_data.columns
    assert len(transformed_data) == len(sample_data)


def test_transform_savitzky_golay(sample_data):
    sm = Smoothing(method='s_g_filter', x_column='x', window=3, polyorder=1)
    transformed_data = sm.transform(sample_data)
    assert 'x_Savitzky_Golaay' in transformed_data.columns
    assert len(transformed_data) == len(sample_data)


def test_transform_fft(sample_data):
    sm = Smoothing(method='fft', x_column='x', threshold=0.5)
    transformed_data = sm.transform(sample_data)
    assert 'x_FFT' in transformed_data.columns
    assert len(transformed_data) == len(sample_data)


def test_transform_unsupported_method_raises_error(sample_data):
    sm = Smoothing(method='unsupported_method')
    with pytest.raises(NotImplementedError):
        sm.transform(sample_data)


def test_plot_transformed_lowess(sample_data):
    sm = Smoothing(method='lowess', x_column='x', y_column='y')
    sm.transform(sample_data)
    sm.plot_transformed()


def test_plot_transformed_lowess2(sample_data):
    sm = Smoothing(method='lowess', x_column='x', y_column='y')
    sm.transform(sample_data, return_sorted=False)
    sm.plot_transformed()


def test_plot_transformed_other_methods(sample_data):
    sm = Smoothing(method='moving_average', x_column='x', window=3)
    sm.transform(sample_data)
    sm.plot_transformed()
