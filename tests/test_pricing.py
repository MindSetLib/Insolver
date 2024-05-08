import numpy as np
from pandas import DataFrame
from insolver.pricing import dynamic_price


class MockModel:
    @staticmethod
    def predict_proba(x):
        return np.random.rand(len(x), 2)


def test_dynamic_price(monkeypatch):
    df = DataFrame({'price': [100, 150], 'feature1': [1, 2], 'feature2': [3, 4]})
    model = MockModel()
    feature_names = ['feature1', 'feature2']
    burning_cost_pct = 0.8
    threshold = 0.5

    result = dynamic_price(
        df,
        price_name='price',
        model=model,
        feature_names=feature_names,
        burning_cost_pct=burning_cost_pct,
        threshold=threshold,
    )

    assert isinstance(result, DataFrame)
    assert result.shape == (2, 8)
