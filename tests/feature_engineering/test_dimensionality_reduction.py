import pytest
import pandas as pd
import numpy as np
from insolver.feature_engineering import DimensionalityReduction


np.random.seed(0)
X = pd.DataFrame(np.random.rand(100, 10))
y = pd.Series(np.random.randint(0, 2, 100))
y2 = pd.DataFrame(np.random.randint(0, 2, 100))


def test_transform_supported_method():
    supported_methods = ['pca', 'svd', 'fa', 'nmf', 'lda', 't_sne', 'isomap', 'lle']
    for method in supported_methods:
        dr = DimensionalityReduction(method)
        transformed = dr.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape[0] == X.shape[0]


def test_transform_unsupported_method():
    with pytest.raises(NotImplementedError):
        dr = DimensionalityReduction('invalid_method')
        dr.transform(pd.DataFrame())


def test_plot_transformed():
    dr = DimensionalityReduction()
    dr.transform(X)
    with pytest.raises(TypeError):
        dr.plot_transformed(X.values)
    dr = DimensionalityReduction()
    with pytest.raises(AttributeError):
        dr.plot_transformed(y)


def test_plot_transformed_output():
    dr = DimensionalityReduction()
    dr.transform(X.iloc[:, :1])
    dr.plot_transformed(y)


def test_plot_transformed_output2():
    dr2 = DimensionalityReduction()
    dr2.transform(X)
    dr2.plot_transformed(y2)
