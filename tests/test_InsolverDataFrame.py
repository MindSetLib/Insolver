import pandas as pd
from insolver import InsolverDataFrame

df = pd.DataFrame({'col1': [1], 'col2': [2]})


def test_InsolverDataFrame():
    InsDataFrame = InsolverDataFrame(df)
    assert hasattr(InsDataFrame, 'get_meta_info')
    assert hasattr(InsDataFrame, 'get_batch')
    assert hasattr(InsDataFrame, 'split_frame')
    assert hasattr(InsDataFrame, 'sample_request')


def test_sample_request():
    InsDataFrame = InsolverDataFrame(df)
    request = InsDataFrame.sample_request()
    assert request['df'] == {'col1': {'0': 1}, 'col2': {'0': 2}}
