import pandas as pd
from insolver import InsolverDataFrame


df = pd.read_csv('tests/data/freMPL-R_100.csv')


def test_InsolverDataFrame():
    InsDataFrame = InsolverDataFrame(df)
    assert hasattr(InsDataFrame, 'get_meta_info')
    assert hasattr(InsDataFrame, 'get_batch')
    assert hasattr(InsDataFrame, 'split_frame')
