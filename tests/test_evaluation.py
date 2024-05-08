import pytest
import numpy as np
from pandas import DataFrame, Series
from insolver.pricing.evaluation import price_eval, eval_candidate
from insolver.pricing.generation import candidates


# Mock model class with predict_proba method
class MockModel:
    @staticmethod
    def predict_proba(x):
        return np.random.rand(len(x), 2)


class MockModel2:
    @staticmethod
    def predict(x):
        return np.random.rand(len(x), 2)


# Test price_eval function
def test_price_eval():
    x = DataFrame({'price': [100, 150], 'orig_premium': [80, 100], 'feature1': [1, 2], 'feature2': [3, 4]})
    model = MockModel()
    feature_names = ['feature1', 'feature2']
    burning_cost_pct = 0.8
    threshold = 0.5
    result = price_eval(x, model, feature_names, burning_cost_pct, threshold)
    assert isinstance(result, DataFrame)
    assert result.shape == (2, 8)


def test_price_eval2():
    x = DataFrame({'price': [100, 150], 'orig_premium': [80, 100], 'feature1': [1, 2], 'feature2': [3, 4]})
    model = MockModel2()
    feature_names = ['feature1', 'feature2']
    burning_cost_pct = 0.8
    threshold = 0.5
    with pytest.raises(ValueError):
        price_eval(x, model, feature_names, burning_cost_pct, threshold)


# Test eval_candidate function
def test_eval_candidate():
    df = DataFrame({'price': [100, 150], 'feature1': [1, 2], 'feature2': [3, 4]})
    model = MockModel()
    feature_names = ['feature1', 'feature2']
    burning_cost_pct = 0.8
    threshold = 0.5
    result = eval_candidate(candidates(df, 'price'), model, feature_names, burning_cost_pct, threshold)
    assert isinstance(result, Series)
    assert len(result) == len(df)
