import json
import pickle

from insolver import InsolverDataFrame


class InsolverTransformMain:
    def __init__(self):
        self.inference = True
        if not hasattr(self, 'priority'):
            raise NotImplementedError("Transformation class should have the 'priority' property.")


class InsolverTransforms(InsolverDataFrame):
    """Class to compose transforms to be done on InsolverDataFrame. Each transform must have the priority param.
    Priority=0: transforms which get values from other (TransformAgeGetFromBirthday, TransformRegionGetFromKladr, etc).
    Priority=1: main transforms of values (TransformAge, TransformVehPower, ets).
    Priority=2: transforms which get intersections of features (TransformAgeGender, ets);
    transforms which sort values (TransformParamSortFreq, TransformParamSortAC).
    Priority=3: transforms which get functions of values (TransformPolynomizer, TransformGetDummies, ets).

    Attributes:
        df: InsolverDataFrame to transform.
        transforms: List of transforms to be done.

    Returns:
        Transformed InsolverDataFrame.
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
            i = 0
            priority_max = max([transform.priority for transform in self.transforms])
            for priority in range(priority_max + 1):
                for transform in self.transforms:
                    if transform.priority == priority:
                        transform(self)
                        attributes = {}
                        for attribute in dir(transform):
                            if attribute[0] != '_':
                                exec("attributes.update({attribute: transform.%s})" % attribute)
                        self.transforms_done.update({i: {'name': type(transform).__name__, 'attributes': attributes}})
                        i += 1
        return self.transforms_done

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.transforms_done, file)

    def save_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.transforms_done, file, separators=(',', ':'), sort_keys=True, indent=4)
