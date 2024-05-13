import pytest
import pandas as pd
import numpy as np
from insolver.feature_engineering import Sampling


@pytest.fixture
def sample_data():
    return pd.DataFrame({'A': np.arange(100), 'B': np.random.rand(100), 'Cluster': np.repeat(np.arange(10), 10)})


def test_simple_sampling(sample_data):
    sampler = Sampling(n=5, method='simple')
    sampled_df = sampler.sample_dataset(sample_data)
    assert len(sampled_df) == 5


def test_systematic_sampling(sample_data):
    sampler = Sampling(n=10, method='systematic')
    sampled_df = sampler.sample_dataset(sample_data)
    assert len(sampled_df) == 10


def test_cluster_sampling(sample_data):
    sampler = Sampling(n=2, method='cluster')
    sampled_df = sampler.sample_dataset(sample_data)
    assert len(sampled_df) == 20  # Two clusters chosen, each with 10 elements


def test_stratified_sampling(sample_data):
    sampler = Sampling(n=2, method='stratified', cluster_column='Cluster')
    sampled_df = sampler.sample_dataset(sample_data)
    assert len(sampled_df) == 20  # Two elements chosen from each cluster


def test_method_not_supported(sample_data):
    sampler = Sampling(n=5, method='unsupported_method')
    with pytest.raises(NotImplementedError):
        sampler.sample_dataset(sample_data)


def test_cluster_sampling_with_invalid_n(sample_data):
    sampler = Sampling(n=150, method='cluster')
    with pytest.raises(ValueError):
        sampler.sample_dataset(sample_data)
    sampler = Sampling(n=55, method='cluster')
    sampler.sample_dataset(sample_data)


def test_create_clusters_with_null_values():
    with pytest.raises(ValueError):
        sampler = Sampling(n=5, method='cluster', cluster_column='Cluster')
        sampler._create_clusters(pd.DataFrame({'Cluster': [1, 2, np.nan, 4]}))


def test_create_clusters_with_insufficient_data():
    with pytest.raises(ValueError):
        sampler = Sampling(n=10, n_clusters=10, method='cluster')
        sampler._create_clusters(pd.DataFrame({'A': [1, 2, 3]}))


def test_create_clusters_with_extra_data():
    data = {'A': np.arange(7)}
    df = pd.DataFrame(data)
    sampler = Sampling(n=5, n_clusters=2, method='cluster')
    new_df = sampler._create_clusters(df)
    assert len(new_df) == 7
    assert all(1 <= cluster_id <= 2 for cluster_id in new_df['cluster_id'])
