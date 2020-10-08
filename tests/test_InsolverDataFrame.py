import pandas as pd
from insolver.InsolverDataFrame import InsolverDataFrame


df = pd.read_csv('tests/data/freMPL-R_100.csv')


def test_InsolverDataFrame():
    InsDataFrame = InsolverDataFrame(df)
    assert hasattr(InsDataFrame, 'get_meta_info')
    assert hasattr(InsDataFrame, 'get_data')
    assert hasattr(InsDataFrame, 'columns_match')
    assert hasattr(InsDataFrame, 'save_data_to_csv')
    assert hasattr(InsDataFrame, 'split_frame')

