import warnings
from typing import Literal, Type, Union, Optional


expected_type_warning = Literal["default", "error", "ignore", "always", "module", "once"]


def warn_insolver(msg: Union[Warning, str],
                  category_: Type[Warning],
                  filter_: expected_type_warning = 'always') -> None:
    def warning_format(message: Union[Warning, str],
                       category: Type[Warning],
                       filename: str,
                       lineno: int,
                       line: Optional[str] = None) -> str:
        return f"{category.__name__}: {message}\n"

    defailt_format = warnings.formatwarning
    warnings.formatwarning = warning_format
    warnings.simplefilter(filter_, category_)
    warnings.warn(msg, category_)
    warnings.formatwarning = defailt_format
