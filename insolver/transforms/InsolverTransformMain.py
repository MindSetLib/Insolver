import json
import pickle
import warnings

from insolver.frame import InsolverDataFrame


class PriorityWarning(UserWarning):
    pass


class InsolverTransform(InsolverDataFrame):
    """Class to compose transforms to be done on InsolverDataFrame. Transforms may have the priority param.
    Priority=0: transforms which get values from other (TransformAgeGetFromBirthday, TransformRegionGetFromKladr, etc).
    Priority=1: main transforms of values (TransformAge, TransformVehPower, ets).
    Priority=2: transforms which get intersections of features (TransformAgeGender, ets);
    transforms which sort values (TransformParamSortFreq, TransformParamSortAC).
    Priority=3: transforms which get functions of values (TransformPolynomizer, TransformGetDummies, ets).

    Attributes:
        df: InsolverDataFrame to transform.
        transforms: List of transforms to be done.
    """
    _metadata = ['transforms', 'transforms_done']

    def __init__(self, df, transforms):
        super().__init__(df)
        if isinstance(transforms, list):
            self.transforms = transforms
        self.transforms_done = {}

    def ins_transform(self):
        """Transforms data in InsolverDataFrame.

        Returns:
            list: List of transforms have been done.
        """
        if self.transforms:

            priority = 0
            for transform in self.transforms:
                if hasattr(transform, 'priority'):
                    if transform.priority < priority:
                        warnings.warn('Check the order of transforms.'
                                      'Transforms with higher priority should be done first.', PriorityWarning)
                    else:
                        priority = transform.priority
                else:
                    pass

            n = 0
            for transform in self.transforms:
                transform(self)
                attributes = {}
                for attribute in dir(transform):
                    if attribute[0] != '_':
                        exec("attributes.update({attribute: transform.%s})" % attribute)
                self.transforms_done.update({n: {'name': type(transform).__name__, 'attributes': attributes}})
                n += 1

        return self.transforms_done

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.transforms_done, file)

    def save_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.transforms_done, file, separators=(',', ':'), sort_keys=True, indent=4)


def load_class(module_list, transform_name):
    for module in module_list:
        try:
            transform_class = getattr(module, transform_name)
            return transform_class
        except AttributeError:
            pass


def init_transforms(transforms, inference):
    """Function for creation transformations objects from the dictionary.

    Args:
        transforms (list): Dictionary with classes and their init parameters.
        inference (bool): Should be 'False' if transforms are applied while preparing data for modeling.
            Should be 'True' if transforms are applied on inference.

    Returns:
        list: List of transformations objects.
    """
    transforms_list = []
    module_list = [InsolverTransforms]

    try:
        import user_transforms
        module_list.append(user_transforms)

    except ModuleNotFoundError:
        pass

    for transform_name in transforms:
        transform_class = load_class(module_list, transform_name)
        if transform_class:
            transforms_list.append(transform_class(**transforms[transform_name]))

    return transforms_list