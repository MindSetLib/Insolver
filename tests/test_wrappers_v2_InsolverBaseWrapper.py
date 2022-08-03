import pytest

from insolver.wrappers_v2 import InsolverBaseWrapper
from insolver.wrappers_v2.utils import save_pickle, save_dill


class DescendantInsolverBaseWrapper(InsolverBaseWrapper):
    algo = 'dummy'
    _backend_saving_methods = {'some_backend': {'pickle': save_pickle, 'dill': save_dill}}

    def __init__(self, backend, task):
        self._get_init_args(vars())
        self.backend = backend
        self.task = task


def test_InsolverBaseWrapper():
    descendant = DescendantInsolverBaseWrapper(backend='some_backend', task='nothing')
    assert descendant.metadata == {
        'init_params': {'backend': 'some_backend', 'task': 'nothing'},
        'is_fitted': False,
        'algo': 'dummy',
    }
    assert descendant.algo == 'dummy'
    assert descendant.model is None
    assert descendant() is None
    assert descendant._backend_saving_methods == {'some_backend': {'pickle': save_pickle, 'dill': save_dill}}
    descendant._update_metadata()
    assert descendant.metadata == {
        'init_params': {'backend': 'some_backend', 'task': 'nothing'},
        'backend': 'some_backend',
        'task': 'nothing',
        'algo': 'dummy',
        'is_fitted': False,
    }


def test_InsolverBaseWrapper_save_model():
    descendant = DescendantInsolverBaseWrapper(backend='some_backend', task='nothing')
    descendant._update_metadata()

    with pytest.raises(ValueError, match="No fitted model found. Fit model first."):
        descendant.save_model()
    descendant.model = {'dummy_object': "model"}

    with pytest.raises(ValueError, match=r'Invalid method ".*". Supported values for .* backend are .*'):
        descendant.save_model(method='some_new_method')

    # model_to_str = descendant.save_model()
    # assert isinstance(model_to_str, bytes)

    # model_to_str = descendant.save_model(method='pickle')
    # assert isinstance(model_to_str, bytes)

    # model_to_str = descendant.save_model(method='dill')
    # assert isinstance(model_to_str, bytes)
