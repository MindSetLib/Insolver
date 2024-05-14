import pytest
import numpy as np
import pandas as pd
from insolver.frame import InsolverDataFrame
from insolver.discretization import InsolverDiscretizer

X = [85, 90, 78, 96, 80, 70, 65, 95]
y = [1, 0, 1, 0, 0, 1, 1, 1]
data = InsolverDataFrame(pd.DataFrame({'X': X, 'y': y}))


def test_method_uniform():
    insolver_disc = InsolverDiscretizer(method='uniform')
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolver_disc.transform(data.X, n_bins=3))


def test_method_quantile():
    insolver_disc = InsolverDiscretizer(method='quantile')
    expected = np.array([1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolver_disc.transform(data.X, n_bins=3))


def test_method_kmeans():
    insolver_disc = InsolverDiscretizer(method='kmeans')
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolver_disc.transform(data.X, n_bins=3))


def test_method_cart():
    insolver_disc = InsolverDiscretizer(method='cart')
    expected = np.array([0.4, 0.4, 1.0, 0.4, 0.4, 1.0, 1.0, 0.4])
    assert np.all(expected == insolver_disc.transform(data.X, data.y, n_bins=3))


def test_method_chimerge():
    insolver_disc = InsolverDiscretizer(method='chimerge')
    expected = np.array([1, 1, 0, 2, 1, 0, 0, 1], dtype=np.int64)
    assert np.all(expected == insolver_disc.transform(data.X, data.y.values, n_bins=3))


def test_data_type():
    insolver_disc = InsolverDiscretizer(method='uniform')
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolver_disc.transform(X, n_bins=3))
    assert np.all(expected == insolver_disc.transform(np.array(X), n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X, n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X.values, n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X.to_frame(), n_bins=3))


def test_target_type():
    insolver_disc = InsolverDiscretizer(method='chimerge')
    expected = np.array([1, 1, 0, 2, 1, 0, 0, 1], dtype=np.int64)
    assert np.all(expected == insolver_disc.transform(data.X, y, n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X, np.array(y), n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X, data.y, n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X, data.y.values, n_bins=3))
    assert np.all(expected == insolver_disc.transform(data.X, data.y.to_frame(), n_bins=3))


def test_n_bins_formulas():
    insolver_disc = InsolverDiscretizer(method='uniform')
    expected_1 = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    expected_2 = np.array([2.0, 3.0, 1.0, 3.0, 1.0, 0.0, 0.0, 3.0])
    expected_3 = np.array([3.0, 4.0, 2.0, 4.0, 2.0, 0.0, 0.0, 4.0])
    expected_4 = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    expected_5 = np.array([30.0, 37.0, 19.0, 46.0, 22.0, 7.0, 0.0, 45.0])
    x_ = np.array([[25]])

    assert np.all(expected_1 == insolver_disc.transform(data.X, n_bins='square-root'))
    assert np.all(expected_2 == insolver_disc.transform(data.X, n_bins='sturges'))
    assert np.all(expected_2 == insolver_disc.transform(data.X, n_bins='huntsberger'))
    assert np.all(expected_3 == insolver_disc.transform(data.X, n_bins='brooks-carrther'))
    assert np.all(expected_4 == insolver_disc.transform(data.X, n_bins='cencov'))
    assert np.all(expected_2 == insolver_disc.transform(data.X, n_bins='rice-rule'))
    assert np.all(expected_1 == insolver_disc.transform(data.X, n_bins='terrell-scott'))
    assert np.all(expected_5 == insolver_disc.transform(data.X, n_bins='scott'))
    assert np.all(expected_4 == insolver_disc.transform(data.X, n_bins='freedman-diaconis'))
    insolver_disc.transform(x_, n_bins=3)


def test_errors():
    for method in ['invalid_method', 'uniform', 'quantile', 'kmeans']:
        if method == 'invalid_method':
            with pytest.raises(NotImplementedError):
                InsolverDiscretizer(method=method)
        else:
            insolver_disc = InsolverDiscretizer(method=method)
            for n_bin in [0, 1, -1, 2.0]:
                with pytest.raises(ValueError):
                    insolver_disc.transform(data.X, n_bins=n_bin)

            with pytest.raises(ValueError):
                insolver_disc.transform(set(data.X), n_bins=n_bin)

            insolver_disc = InsolverDiscretizer(method=method)
            x_ = np.array([[[25], [10], [30]], [[25], [10], [30]]])
            with pytest.raises(ValueError):
                insolver_disc.transform(x_, n_bins=3)

    for method in ['cart', 'chimerge']:
        insolver_disc = InsolverDiscretizer(method=method)
        with pytest.raises(ValueError):
            insolver_disc.transform(data.X, set(range(data.X.shape[0])), n_bins=3)

        with pytest.raises(ValueError):
            insolver_disc.transform(data.X, list(set(data.y)), n_bins=3)
