from typing import List, Dict, Type, Union, Optional, Any
import dill
import numpy as np
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.utils import warn_insolver


class TransformsWarning(Warning):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


class InsolverTransform(InsolverDataFrame):
    """Class to compose transforms to be done on InsolverDataFrame. Transforms may have the priority param.
    Priority=0: transforms which get values from other (TransformAgeGetFromBirthday, TransformRegionGetFromKladr, etc).
    Priority=1: main transforms of values (TransformAge, TransformVehPower, etc).
    Priority=2: transforms which get intersections of features (TransformAgeGender, etc);
    transforms which sort values (TransformParamSortFreq, TransformParamSortAC).
    Priority=3: transforms which get functions of values (TransformPolynomizer, TransformGetDummies, etc).

    Parameters:
        data: InsolverDataFrame to transform.
        transforms: List of transforms to be done.
    """
    _internal_names = pd.DataFrame._internal_names + ["transforms_done", "ins_output_cache"]
    _internal_names_set = set(_internal_names)
    _metadata = ["transforms", "ins_input_cache"]

    def __init__(self, data: Any, transforms: Union[List, Dict[str, Union[List, Dict]], None] = None) -> None:
        super(InsolverTransform, self).__init__(data)
        self.ins_output_cache: Optional[Dict[str, np.dtype]] = None
        if isinstance(data, (InsolverDataFrame, pd.DataFrame)):
            self.ins_input_cache = dict(zip(list(self.columns), list(self.dtypes)))

        if isinstance(transforms, list):
            self.transforms = transforms
        elif isinstance(transforms, dict) and _check_transforms(transforms):
            for key in transforms.keys():
                setattr(self, key, transforms[key])

        self.transforms_done: Dict = dict()

    @property
    def _constructor(self) -> Type["InsolverTransform"]:
        return InsolverTransform

    @staticmethod
    def _check_colnames_dtypes(expected: Dict[str, np.dtype], input_: Dict[str, np.dtype], step: str) -> None:
        missing_col_checks = set(expected.keys()).difference(set(input_.keys()))
        if missing_col_checks != set():
            warn_insolver(f'{step.capitalize()} data missing columns {list(missing_col_checks)}!', TransformsWarning)
            common_cols = set(expected.keys()).intersection(set(input_.keys()))
            input_ = {key: input_[key] for key in common_cols}
            expected = {key: expected[key] for key in common_cols}

        if expected != input_:
            for key in expected.keys():
                if expected[key] != input_[key]:
                    message = f"{key}: input {input_[key]}, expected {expected[key]}"
                    warn_insolver(f'{step.capitalize()} column dtype mismatch: Column {message}!', TransformsWarning)

    def ins_transform(self) -> Dict:
        """Transforms data in InsolverDataFrame.

        Returns:
            list: List of transforms have been done.
        """
        self._check_colnames_dtypes(self.ins_input_cache, dict(self.dtypes), 'input')

        if self.transforms:
            priority = 0
            for transform in self.transforms:
                if hasattr(transform, 'priority'):
                    if transform.priority < priority:
                        warn_insolver('Check the order of transforms. '
                                      'Transforms with higher priority should be done first!',
                                      TransformsWarning)
                        break
                    else:
                        priority = transform.priority

            for n, transform in enumerate(self.transforms):
                self._update_inplace(transform(self))
                attributes = dict()
                for attribute in dir(transform):
                    if not attribute.startswith('_'):
                        attributes.update({attribute: getattr(transform, attribute)})
                self.transforms_done.update({n: {'name': type(transform).__name__, 'attributes': attributes}})

            if hasattr(self, "ins_output_cache") and (self.ins_output_cache is not None):
                self._check_colnames_dtypes(self.ins_output_cache, dict(self.dtypes), "output")
            else:
                self.ins_output_cache = dict(zip(list(self.columns), list(self.dtypes)))
        return self.transforms_done

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            dill.dump({"transforms": self.transforms,
                       "ins_input_cache": self.ins_input_cache,
                       "ins_output_cache": self.ins_output_cache,
                       "transforms_done": self.transforms_done}, file)


def _check_transforms(obj: Any) -> bool:
    condition = False
    if isinstance(obj, dict):
        required = ["transforms", "transforms_done", "ins_output_cache", "ins_input_cache"]
        req_type = [list, dict, dict, dict]
        if (set(obj.keys()).difference(set(required)) == set()) and (list(map(type, obj.values())) == req_type):
            condition = True
    return condition


def load_transforms(path: str) -> Optional[Dict[str, Union[List, Dict]]]:
    with open(path, 'rb') as file:
        loaded_file = dill.load(file)

    if _check_transforms(loaded_file):
        return loaded_file
    else:
        ValueError('Loaded file is not supported by InsolverTransform.')
