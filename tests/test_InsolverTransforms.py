import pandas as pd
from insolver.InsolverDataFrame import InsolverDataFrame

from insolver.InsolverTransforms import (
    TransformExp,
    InsolverTransformMain,
    InsolverTransforms,
    TransformAge,
    TransformMapValues,
    TransformPolynomizer,
    TransformAgeGender,
)

df = pd.read_csv('tests/data/freMPL-R_100.csv')


def test_TransformExp():
    # transform_exp = TransformExp('LicAge', 57)
    exp_max = 52
    assert TransformExp._exp(50, exp_max) == 50
    assert TransformExp._exp(60, exp_max) == 52
    assert TransformExp._exp(None, exp_max) is None
    assert TransformExp._exp(-5, exp_max) is None
    assert TransformExp._exp(0, exp_max) == 0
