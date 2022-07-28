import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, OneHotEncoderTransforms


def test_OneHotEncoderTransforms():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A']}))
    df_transformed = InsolverTransform(
        df_test,
        [
            OneHotEncoderTransforms(['col1']),
        ],
    )
    df_transformed.ins_transform()
    assert 'col1_A' in df_transformed.columns
    assert 'col1_B' in df_transformed.columns
    assert 'col1_C' in df_transformed.columns
    assert df_transformed['col1_A'][0] == 1
    assert df_transformed['col1_B'][1] == 1
    assert df_transformed['col1_C'][2] == 1
    assert df_transformed['col1_A'][3] == 1
