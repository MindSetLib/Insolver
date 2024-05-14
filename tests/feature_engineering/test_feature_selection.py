import pytest
import numpy as np
import pandas as pd
from insolver.feature_engineering import FeatureSelection


@pytest.fixture
def sample_data():
    np.random.seed(0)
    df = pd.DataFrame(
        {'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'target': np.random.randint(0, 2, size=100)}
    )
    return df


@pytest.fixture
def sample_data2():
    np.random.seed(0)
    df = pd.DataFrame(
        {'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'target': np.random.randint(0, 3, size=100)}
    )
    return df


def test_create_model(sample_data):
    fs = FeatureSelection(y_column='target', task='class', permutation_importance=True)
    fs.create_model(sample_data)
    df2 = sample_data.copy()
    df2.loc[2, 'feature2'] = None
    with pytest.raises(ValueError):
        fs.create_model(df2)
    df3 = sample_data.copy()
    df3['feature2'] = df3['feature2'].astype('object')
    with pytest.raises(ValueError):
        fs.create_model(df3)
    fs2 = FeatureSelection(y_column='target', task='class', method='invalid_method')
    with pytest.raises(NotImplementedError):
        fs2.create_model(sample_data)


def test_permutation_importance(sample_data):
    fs = FeatureSelection(y_column='target', task='class', permutation_importance=True)
    with pytest.raises(AttributeError):
        fs.create_permutation_importance()
    fs = FeatureSelection(y_column='target', task='class', permutation_importance=True)
    fs.create_model(sample_data)
    fs.model = 1
    with pytest.raises(TypeError):
        fs.create_permutation_importance()


def test_init_methods(sample_data):
    fs = FeatureSelection('target', 'reg')
    fs.create_model(sample_data)
    fs2 = FeatureSelection('target', 'multiclass')
    fs2.create_model(sample_data)
    fs3 = FeatureSelection('target', 'multiclass_multioutput')
    fs3.create_model(sample_data)
    fs4 = FeatureSelection('target', 'invalid_task')
    with pytest.raises(NotImplementedError):
        fs4.create_model(sample_data)


def test_call(sample_data):
    fs = FeatureSelection('target', 'reg')
    fs(sample_data)


def test_create_new_dataset(sample_data):
    fs = FeatureSelection('target', 'class')
    fs.create_model(sample_data)
    fs.create_new_dataset()
    assert isinstance(fs.new_dataframe, pd.DataFrame)

    fs2 = FeatureSelection('target', 'class')
    fs2.create_model(sample_data)
    fs2.create_new_dataset(threshold='median')
    assert isinstance(fs2.new_dataframe, pd.DataFrame)

    fs3 = FeatureSelection('target', 'class')
    with pytest.raises(AttributeError):
        fs3.create_new_dataset(threshold='median')

    fs4 = FeatureSelection('target', 'multiclass', 'elasticnet')
    fs4.create_model(sample_data)
    fs4.create_new_dataset(threshold=0.4)
    assert isinstance(fs4.new_dataframe, pd.DataFrame)


def test_plot_importance(sample_data):
    fs = FeatureSelection('target', 'class')
    with pytest.raises(AttributeError):
        fs.plot_importance()

    fs = FeatureSelection('target', 'multiclass')
    fs.create_model(sample_data)
    fs.plot_importance(importance_threshold=0.4)


def test_plot_importance2(sample_data2):
    fs = FeatureSelection('target', 'multiclass', method='elasticnet', permutation_importance=True)
    fs.create_model(sample_data2)
    fs.importances = np.random.uniform(0, 1, size=(3, 2))
    fs.plot_importance()

    fs = FeatureSelection('target', 'multiclass', method='elasticnet', permutation_importance=True)
    fs.create_model(sample_data2)
    fs.importances = np.random.uniform(0, 1, size=(3, 2))
    fs.plot_importance(importance_threshold=0.4)
