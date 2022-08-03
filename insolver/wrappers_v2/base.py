import os
import time
import json
from io import BytesIO
from os import PathLike
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Union, Any, Dict, Callable

import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class InsolverWrapperWarning(Warning):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


class InsolverBaseWrapper:
    """Base wrapper serving as a building block for other wrappers."""

    model: Any = None
    metadata: Dict[str, Any] = dict()
    backend: str = ''
    task: str = ''
    algo: str = ''
    _backend_saving_methods: Dict[str, Dict[str, Callable]] = dict()
    _model_cached: Any = None

    def __call__(self) -> Any:
        return self.model

    def _get_init_args(self, vars_: Any) -> None:
        copy_vars = deepcopy(vars_)
        copy_vars.pop("self")
        self.metadata = {"init_params": copy_vars, 'is_fitted': False, 'algo': self.__class__.algo}

    def _update_metadata(self) -> None:
        _metadata = self.__dict__.copy()
        _metadata = {
            key: _metadata[key] for key in _metadata if not (key in ['model', 'metadata'] or key.startswith('_'))
        }
        self.metadata.update(_metadata)

    def _save_insolver(self, path_or_buf: Union[str, 'PathLike[str]'], method: Callable, **kwargs: Any) -> None:
        buffer = BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
            zip_file.writestr("metadata.json", json.dumps(self.metadata))
            zip_file.writestr(
                f"model_{os.path.basename(path_or_buf)}",
                BytesIO(method(self.model, path_or_buf=None, **kwargs)).getvalue(),
            )

        with open(path_or_buf if str(path_or_buf).endswith('.zip') else f'{path_or_buf}.zip', "wb") as f:
            f.write(buffer.getvalue())

    def save_model(
        self,
        path_or_buf: Union[None, str, 'PathLike[str]'] = None,
        mode: Literal['insolver', 'raw'] = "insolver",
        method: str = '',
        **kwargs: Any,
    ) -> Union[str, bytes]:
        """Saving the model contained in wrapper.

        Args:
            path_or_buf (str, os.PathLike[str]): Filepath or buffer object. If None, the result is returned as a string.
            mode (str, optional): Saving mode, values ['insolver', 'raw'] are supported. Option 'raw' saves fitted model
             without additional metadata. Option 'insolver' saves model as a zip-file with model and json with metadata
             inside.
            method (str, optional): Saving method.
            **kwargs: Other parameters passed to, e.g. h2o.save_model().
        """
        _modes = ["insolver", "raw"]

        if mode not in _modes:
            raise ValueError(f"Invalid mode argument {mode}. Mode must one of {_modes}")

        if method == '' and len(self._backend_saving_methods[self.backend].keys()) > 0:
            method = list(self._backend_saving_methods[self.backend].keys())[0]
        elif method not in self._backend_saving_methods[self.backend].keys():
            raise ValueError(
                f'Invalid method "{method}". '
                f'Supported values for "{self.backend}" backend are '
                f'{list(self._backend_saving_methods[self.backend].keys())}.'
            )

        if not self.metadata['is_fitted']:
            raise ValueError("No fitted model found. Fit model first.")

        if (path_or_buf is not None) and isinstance(path_or_buf, str):
            path_or_buf = os.path.abspath(path_or_buf)
            if os.path.isdir(path_or_buf):
                default_name = (
                    f"{'insolver' if mode == 'insolver' else method}"
                    f"_{self.algo}_{self.backend}_{self.task}_{round(time.time() * 1000)}"
                )
                path_or_buf = os.path.normpath(os.path.join(path_or_buf, default_name))

        if path_or_buf is None:
            if self._model_cached is None:
                return self._backend_saving_methods[self.backend][method](self.model, path_or_buf, **kwargs)
            else:
                return self._model_cached
        else:
            if mode == "insolver":
                self.metadata.update({"saving_method": method})
                if self._model_cached is None:
                    self._save_insolver(
                        path_or_buf, method=self._backend_saving_methods[self.backend][method], **kwargs
                    )
                else:
                    self._save_insolver(
                        path_or_buf,
                        method=self._backend_saving_methods[self.backend][method],
                        _model_cached=self._model_cached,
                        **kwargs,
                    )
                path_or_buf = f'{path_or_buf}.zip'
            else:
                self._backend_saving_methods[self.backend][method](self.model, path_or_buf, **kwargs)
            return f"Saved model: {os.path.normpath(path_or_buf)}"
