import warnings
from typing import Type, Union, Optional

import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


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
