import warnings
import importlib
from typing import Type, Union, Optional
from typing import Literal


expected_type_warning = Literal["default", "error", "ignore", "always", "module", "once"]


def warn_insolver(msg: Union[Warning, str], category_: Type[Warning]) -> None:
    def warning_format(
        message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str] = None
    ) -> str:
        return f"{category.__name__}: {message}\n"

    default_format = warnings.formatwarning
    warnings.formatwarning = warning_format
    warnings.warn(msg, category_)
    warnings.formatwarning = default_format


def check_dependency(package_name: str, extra_name: str) -> None:
    try:
        importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(
            f"{package_name} is required for insolver.{extra_name}. "
            f"Please install it with `pip install insolver[{extra_name}]`. {e}"
        )
