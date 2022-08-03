import os
from os import PathLike
from typing import Dict, Any, Union, Optional, Tuple

from pandas import DataFrame, Series, concat
from numpy import arange

from h2o.frame import H2OFrame
from h2o.backend import H2OLocalServer
from h2o.estimators import H2OEstimator
from h2o import no_progress, cluster, remove_all, connect, load_model, save_model


def h2o_start(h2o_server_params: Dict[str, Any] = None) -> None:
    # nthreads=-1, enable_assertions=True, max_mem_size=None, min_mem_size=None,
    # ice_root=None, log_dir=None, log_level=None, max_log_file_size=None, port="54321+", name=None,
    # extra_classpath=None, verbose=True, jvm_custom_args=None, bind_to_localhost=True
    h2o_server_params = {'verbose': False} if h2o_server_params is None else h2o_server_params
    no_progress()
    if (cluster() is None) or (not cluster().is_running()):
        h2oserver = H2OLocalServer.start(**h2o_server_params)
        connect(server=h2oserver, verbose=False)


def h2o_stop() -> None:
    if (cluster() is not None) or (cluster().is_running()):
        remove_all()
        cluster().shutdown()


def to_h2oframe(df: DataFrame) -> H2OFrame:
    """Function converts pandas.DataFrame to h2o.H2OFrame ensuring there is no bug duplicating rows in results.

    Args:
        df (pandas.DataFrame):  Dataset to convert to h2o.H2OFrame

    Returns:
        DataFrame converted to h2o.H2OFrame.
    """

    # https://stackoverflow.com/questions/45672118/h2oframe-in-python-is-adding-additional-duplicate-rows-to-the-pandas-dataframe
    df_h2o = df.copy().reset_index(drop=True)
    h2of = H2OFrame(df_h2o)

    if h2of.shape[0] != df_h2o.shape[0]:
        df_h2o['__insolver_temp_row_id'] = arange(len(df_h2o))
        h2of = H2OFrame(df_h2o)
        h2of = h2of.drop_duplicates(columns=['__insolver_temp_row_id'], keep='first')
        h2of = h2of.drop('__insolver_temp_row_id', axis=1)
    return h2of


def x_y_to_h2o_frame(
    x: Union[DataFrame, Series],
    y: Union[DataFrame, Series],
    sample_weight: Union[DataFrame, Series],
    params: Dict,
    x_valid: Union[DataFrame, Series],
    y_valid: Union[DataFrame, Series],
    sample_weight_valid: Union[DataFrame, Series],
) -> Tuple[H2OFrame, Dict]:
    if (sample_weight is not None) and isinstance(sample_weight, (DataFrame, Series)):
        params['offset_column'] = (
            list(sample_weight.columns) if isinstance(sample_weight, DataFrame) else sample_weight.name
        )
        x = concat([x, sample_weight], axis=1)
    train_set = to_h2oframe(concat([x, y], axis=1))

    if (x_valid is not None) and (y_valid is not None):
        if all([sam_weight is not None for sam_weight in [sample_weight_valid, sample_weight]]) and isinstance(
            sample_weight_valid, (DataFrame, Series)
        ):
            x_valid = concat([x_valid, sample_weight_valid], axis=1)
        valid_set = to_h2oframe(concat([x_valid, y_valid], axis=1))
        params['validation_frame'] = valid_set
    return train_set, params


def save_h2o(
    model: H2OEstimator, path_or_buf: Union[None, str, 'PathLike[str]'] = None, **kwargs: Any
) -> Optional[bytes]:
    if not ((path_or_buf is None) or (isinstance(path_or_buf, str))):
        raise ValueError(f"Invalid file path or buffer object {type(path_or_buf)}")

    _model_cached = None if '_model_cached' not in kwargs else kwargs.pop('_model_cached')

    if path_or_buf is None:
        # Since there no possibility to save h2o model to a variable, workaround is needed
        if _model_cached is None:
            save_model(model=model, filename='.temp_h2o_model_save', **kwargs)
            with open('.temp_h2o_model_save', 'rb') as file:
                saved = file.read()
            os.remove('.temp_h2o_model_save')
        else:
            saved = _model_cached
        return saved
    else:
        path, filename = os.path.split(path_or_buf)
        # force = False, export_cross_validation_predictions = False
        save_model(model=model, path=path, filename=filename, **kwargs)
        return None


def load_h2o(
    path_or_buf: Union[str, 'PathLike[str]', bytes],
    h2o_server_params: Optional[Dict[str, Any]] = None,
    terminate: bool = True,
) -> H2OEstimator:
    h2o_start(h2o_server_params)
    if isinstance(path_or_buf, (str, PathLike)):
        model = load_model(path_or_buf)
    else:
        # Since there no possibility to load h2o model from a variable, workaround is needed
        with open('.temp_h2o_model_load', 'wb') as file:
            file.write(path_or_buf)
        model = load_model('.temp_h2o_model_load')
        os.remove('.temp_h2o_model_load')
    if terminate:
        h2o_stop()
    return model
