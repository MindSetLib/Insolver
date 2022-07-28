import numpy as np
import pandas as pd
from insolver.frame import InsolverDataFrame
from insolver.discretization import InsolverDiscretizer

X = [85, 90, 78, 96, 80, 70, 65, 95]
y = [1, 0, 1, 0, 0, 1, 1, 1]
data = InsolverDataFrame(pd.DataFrame({'X': X, 'y': y}))


def test_method_uniform():
    insolverDisc = InsolverDiscretizer(method='uniform')
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolverDisc.transform(data.X, n_bins=3))


def test_method_quantile():
    insolverDisc = InsolverDiscretizer(method='quantile')
    expected = np.array([1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolverDisc.transform(data.X, n_bins=3))


def test_method_kmeans():
    insolverDisc = InsolverDiscretizer(method='kmeans')
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolverDisc.transform(data.X, n_bins=3))


def test_method_cart():
    insolverDisc = InsolverDiscretizer(method='cart')
    expected = np.array([0.4, 0.4, 1.0, 0.4, 0.4, 1.0, 1.0, 0.4])
    assert np.all(expected == insolverDisc.transform(data.X, data.y, n_bins=3))


def test_method_chimerge():
    insolverDisc = InsolverDiscretizer(method='chimerge')
    expected = np.array([1, 1, 0, 2, 1, 0, 0, 1], dtype=np.int64)
    assert np.all(expected == insolverDisc.transform(data.X, data.y.values, n_bins=3))


def test_data_type():
    insolverDisc = InsolverDiscretizer(method='uniform')
    expected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert np.all(expected == insolverDisc.transform(X, n_bins=3))
    assert np.all(expected == insolverDisc.transform(np.array(X), n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X, n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X.values, n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X.to_frame(), n_bins=3))


def test_target_type():
    insolverDisc = InsolverDiscretizer(method='chimerge')
    expected = np.array([1, 1, 0, 2, 1, 0, 0, 1], dtype=np.int64)
    assert np.all(expected == insolverDisc.transform(data.X, y, n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X, np.array(y), n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X, data.y, n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X, data.y.values, n_bins=3))
    assert np.all(expected == insolverDisc.transform(data.X, data.y.to_frame(), n_bins=3))


def test_n_bins_formulas():
    insolverDisc = InsolverDiscretizer(method='uniform')
    expected_1 = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    expected_2 = np.array([2.0, 3.0, 1.0, 3.0, 1.0, 0.0, 0.0, 3.0])
    expected_3 = np.array([3.0, 4.0, 2.0, 4.0, 2.0, 0.0, 0.0, 4.0])
    expected_4 = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    expected_5 = np.array([30.0, 37.0, 19.0, 46.0, 22.0, 7.0, 0.0, 45.0])
    assert np.all(expected_1 == insolverDisc.transform(data.X, n_bins='square-root'))
    assert np.all(expected_2 == insolverDisc.transform(data.X, n_bins='sturges'))
    assert np.all(expected_2 == insolverDisc.transform(data.X, n_bins='huntsberger'))
    assert np.all(expected_3 == insolverDisc.transform(data.X, n_bins='brooks-carrther'))
    assert np.all(expected_4 == insolverDisc.transform(data.X, n_bins='cencov'))
    assert np.all(expected_2 == insolverDisc.transform(data.X, n_bins='rice-rule'))
    assert np.all(expected_1 == insolverDisc.transform(data.X, n_bins='terrell-scott'))
    assert np.all(expected_5 == insolverDisc.transform(data.X, n_bins='scott'))
    assert np.all(expected_4 == insolverDisc.transform(data.X, n_bins='freedman-diaconis'))
