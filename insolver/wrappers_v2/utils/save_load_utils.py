import os
import json
import pickle
import dill
from os import PathLike
from typing import Union, Any, Optional, IO, Callable, Dict
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

from .h2o_utils import load_h2o


def load(path_or_buf: Union[str, 'PathLike[str]', bytes], saving_method: str, **kwargs: Any) -> Callable:
    load_config: Dict[str, Callable] = dict(pickle=load_pickle, dill=load_dill, h2o=load_h2o)
    return load_config[saving_method](path_or_buf, **kwargs)


def load_model(path_or_buf: Union[str, 'PathLike[str]', IO[bytes]], **kwargs: Any) -> Any:
    from insolver.wrappers_v2 import InsolverGLMWrapper

    wrapper_config = dict(glm=InsolverGLMWrapper)

    if isinstance(path_or_buf, str):
        path_or_buf = os.path.abspath(path_or_buf)

    try:
        with ZipFile(file=path_or_buf, mode="r", compression=ZIP_DEFLATED) as zip_file:
            filenames = zip_file.namelist()
            if (len(zip_file.filelist) == 2) and ("metadata.json" in filenames):
                metadata = json.loads(zip_file.read("metadata.json"))
                filenames.remove("metadata.json")
                model = zip_file.read(filenames[0])
            else:
                raise RuntimeError(
                    "File has inappropriate format. Currently `load_model` can load only models saved "
                    "with `mode='insolver'` option."
                )

            init_params = metadata["init_params"]
            init_params.update(init_params.pop("kwargs"))
            wrapper_ = wrapper_config[metadata["algo"]](**init_params)
            wrapper_.metadata.update(metadata)
            wrapper_.model = load(model, metadata["saving_method"], **kwargs)
            wrapper_.metadata.pop("saving_method")
            return wrapper_
    except BadZipFile:
        raise RuntimeError(
            "File has inappropriate format. Currently `load_model` can load only models saved "
            "with `mode='insolver'` option."
        )


def save_pickle(model: Any, path_or_buf: Union[None, str, 'PathLike[str]'] = None, **kwargs: Any) -> Optional[bytes]:
    if not ((path_or_buf is None) or (isinstance(path_or_buf, str))):
        raise ValueError(f"Invalid file path or buffer object {type(path_or_buf)}")

    if path_or_buf is None:
        return pickle.dumps(model, **kwargs)
    else:
        with open(path_or_buf, "wb") as _file:
            pickle.dump(model, _file, **kwargs)
        return None


def load_pickle(path_or_buf: Union[str, 'PathLike[str]', bytes], **kwargs: Any) -> Any:
    if isinstance(path_or_buf, (str, PathLike)):
        with open(path_or_buf, 'rb') as _file:
            return pickle.load(_file, **kwargs)
    else:
        return pickle.loads(path_or_buf, **kwargs)


def save_dill(model: Any, path_or_buf: Union[None, str, 'PathLike[str]'] = None, **kwargs: Any) -> Optional[bytes]:
    if not ((path_or_buf is None) or (isinstance(path_or_buf, str))):
        raise ValueError(f"Invalid file path or buffer object {type(path_or_buf)}")

    if path_or_buf is None:
        return dill.dumps(model, **kwargs)
    else:
        with open(path_or_buf, "wb") as _file:
            dill.dump(model, _file, **kwargs)
        return None


def load_dill(path_or_buf: Union[str, 'PathLike[str]', bytes], **kwargs: Any) -> Any:
    if isinstance(path_or_buf, (str, PathLike)):
        with open(path_or_buf, 'rb') as _file:
            return dill.load(_file, **kwargs)
    else:
        return dill.loads(path_or_buf, **kwargs)
