import pandas as pd

from insolver.transforms import (
    TransformExp,
    TransformAge,
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


def test_TransformAge():
    # TransformAge('DrivAge', 18, 75)
    age_min = 18
    age_max = 70
    assert TransformAge._age(None, age_min, age_max) is None
    assert TransformAge._age(16, age_min, age_max) is None
    assert TransformAge._age(80, age_min, age_max) == 70
    assert TransformAge._age(50, age_min, age_max) == 50
