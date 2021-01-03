import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, EncoderTransforms


def test_EncoderTransforms():
    df_test = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A']}))
    InsTransforms = InsolverTransform(df_test, [
        EncoderTransforms(['col1']),
    ])
    InsTransforms.ins_transform()
    assert df_test['col1'][0] == 0
    assert df_test['col1'][1] == 1
    assert df_test['col1'][2] == 2
    assert df_test['col1'][3] == 0
