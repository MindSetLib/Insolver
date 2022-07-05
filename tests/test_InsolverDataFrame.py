import pandas as pd
from insolver import InsolverDataFrame

df = pd.DataFrame({'col1': [1], 'col2': [2]})


def test_InsolverDataFrame():
    insdf = InsolverDataFrame(df)
    assert hasattr(insdf, 'get_meta_info')
    assert hasattr(insdf, 'split_frame')
    assert hasattr(insdf, 'sample_request')


def test_get_meta_info():
    insdf = InsolverDataFrame(df)
    meta_info = insdf.get_meta_info()
    assert meta_info['type'] == 'InsolverDataFrame'
    assert meta_info['len'] == 1
    assert meta_info['columns'][0]['name'] == 'col1'
    assert meta_info['columns'][1]['name'] == 'col2'
    assert meta_info['columns'][0]['use'] == 'unknown'
    assert meta_info['columns'][1]['use'] == 'unknown'
    assert meta_info['columns'][0]['dtype'] == df.col1.dtypes
    assert meta_info['columns'][1]['dtype'] == df.col2.dtypes


def test_sample_request():
    insdf = InsolverDataFrame(df)
    request = insdf.sample_request()
    assert request['df'] == {'col1': 1, 'col2': 2}
