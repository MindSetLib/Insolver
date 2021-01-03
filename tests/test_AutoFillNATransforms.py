import numpy as np
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, AutoFillNATransforms


def test_fillna_numerical():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': [1, 2, np.nan]}))
    InsTransforms = InsolverTransform(df_test, [
        AutoFillNATransforms('col1'),
    ])
    InsTransforms.ins_transform()
    assert df_test['col1'][2] == 1.5


def test_fillna_numerical_all_na():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': [np.nan, np.nan, np.nan]}))
    InsTransforms = InsolverTransform(df_test, [
        AutoFillNATransforms('col1'),
    ])
    InsTransforms.ins_transform()
    assert df_test['col1'][0] == 1
    assert df_test['col1'][1] == 1
    assert df_test['col1'][2] == 1


def test_fillna_categorical():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A', None]}))
    InsTransforms = InsolverTransform(df_test, [
        AutoFillNATransforms('col1'),
    ])
    InsTransforms.ins_transform()
    assert df_test['col1'][4] == 'A'


def test_fillna_categorical_all_na():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': [None, None, None]}))
    InsTransforms = InsolverTransform(df_test, [
        AutoFillNATransforms('col1'),
    ])
    InsTransforms.ins_transform()
    assert df_test['col1'][0] == 1
    assert df_test['col1'][1] == 1
    assert df_test['col1'][2] == 1
