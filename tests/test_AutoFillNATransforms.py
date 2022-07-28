import numpy as np
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, AutoFillNATransforms


def test_fillna_numerical():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': [1, 2, np.nan]}))
    df_transformed = InsolverTransform(
        df_test,
        [
            AutoFillNATransforms(),
        ],
    )
    df_transformed.ins_transform()
    assert df_transformed['col1'][2] == 1.5


def test_fillna_numerical_all_na():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': [np.nan, np.nan, np.nan]}))
    df_transformed = InsolverTransform(
        df_test,
        [
            AutoFillNATransforms(),
        ],
    )
    df_transformed.ins_transform()
    assert df_transformed['col1'][0] == 1
    assert df_transformed['col1'][1] == 1
    assert df_transformed['col1'][2] == 1


def test_fillna_categorical():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A', None]}))
    df_transformed = InsolverTransform(
        df_test,
        [
            AutoFillNATransforms(),
        ],
    )
    df_transformed.ins_transform()
    assert df_transformed['col1'][4] == 'A'


def test_fillna_categorical_all_na():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': [None, None, None]}))
    df_transformed = InsolverTransform(
        df_test,
        [
            AutoFillNATransforms(),
        ],
    )
    df_transformed.ins_transform()
    assert df_transformed['col1'][0] == 1
    assert df_transformed['col1'][1] == 1
    assert df_transformed['col1'][2] == 1
