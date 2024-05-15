import pytest
import pandas as pd
from insolver.feature_engineering import DataPreprocessing


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [6.0, 7.0, 8.0, 9.0, 10.0],
            'D': [11, 12, None, 14, 15],
            'E': [0, 1, 1, 2, 1],
        }
    )


def test_smoothing(sample_data):
    data_preprocessing = DataPreprocessing(smoothing=True, smoothing_column='A')
    smoothed_data = data_preprocessing.preprocess(sample_data)
    assert isinstance(smoothed_data, pd.DataFrame)
    assert 'A' in smoothed_data.columns

    data_preprocessing = DataPreprocessing(smoothing=True)
    with pytest.raises(AttributeError):
        data_preprocessing.preprocess(sample_data)


def test_feature_selection(sample_data):
    data_preprocessing = DataPreprocessing(feature_selection=True, feat_select_task='class')
    data_preprocessing.preprocess(sample_data, target='E')

    with pytest.raises(NotImplementedError):
        data_preprocessing.preprocess(sample_data, target=['B', 'E'])

    data_preprocessing = DataPreprocessing(feature_selection=True)
    with pytest.raises(AttributeError):
        data_preprocessing.preprocess(sample_data, target='E')


def test_sampling(sample_data):
    data_preprocessing = DataPreprocessing(sampling=True)
    data_preprocessing.preprocess(sample_data)


def test_dimension_reduction(sample_data):
    data_preprocessing = DataPreprocessing(dim_red=True)
    data_preprocessing.preprocess(sample_data)

    data_preprocessing = DataPreprocessing(dim_red=True)
    data_preprocessing.preprocess(sample_data, target='E')

    data_preprocessing = DataPreprocessing(dim_red='t_sne', dim_red_n_neighbors=3)
    data_preprocessing.preprocess(sample_data)

    data_preprocessing = DataPreprocessing(dim_red='isomap', dim_red_n_neighbors=3)
    data_preprocessing.preprocess(sample_data)

    data_preprocessing = DataPreprocessing(dim_red='lda', normalization=False, dim_red_n_components=2)
    with pytest.raises(NotImplementedError):
        data_preprocessing.preprocess(sample_data, target=['A', 'E'])
    data_preprocessing.preprocess(sample_data, target='E')


def test_features(sample_data):
    data_preprocessing = DataPreprocessing(categorical_columns=['B'], numerical_columns=['A', 'C'])
    with pytest.raises(NotImplementedError):
        data_preprocessing.preprocess(sample_data)

    data_preprocessing = DataPreprocessing(categorical_columns=['B'], numerical_columns=['A', 'C'])
    data_preprocessing.preprocess(sample_data, target='E')

    data_preprocessing = DataPreprocessing(categorical_columns=['B'], numerical_columns=['A', 'F'])
    with pytest.raises(AttributeError):
        data_preprocessing.preprocess(sample_data, target='E')

    data_preprocessing = DataPreprocessing(categorical_columns=['B', 'G'], numerical_columns=['A', 'C'])
    with pytest.raises(AttributeError):
        data_preprocessing.preprocess(sample_data, target='E')

    data_preprocessing = DataPreprocessing(numerical_columns=['A', 'C'])
    data_preprocessing.preprocess(sample_data, target='E')

    data_preprocessing = DataPreprocessing(categorical_columns=['B'])
    data_preprocessing.preprocess(sample_data, target='E')

    data_preprocessing = DataPreprocessing(
        numerical_columns=['A', 'C'], categorical_columns=['B'], smoothing=True, smoothing_column='A'
    )
    data_preprocessing.preprocess(sample_data)
