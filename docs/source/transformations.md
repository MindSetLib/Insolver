# Transformations

Transformations allow you to preprocess data, save your preprocessing steps to pickle file and using it in the model inference

## Build-in transformations

Example of using transformations

```python
import pandas as pd

from insolver import InsolverDataFrame
from insolver.InsolverTransforms import (
    TransformExp,
    InsolverTransforms,
    TransformAge,
    TransformMapValues,
    TransformPolynomizer,
    TransformAgeGender,
)

InsDataFrame = InsolverDataFrame(pd.read_csv('freMPL-R.csv', low_memory=False))

InsTransforms = InsolverTransforms(InsDataFrame, [
    TransformAge('DrivAge', 18, 75),
    TransformExp('LicAge', 57),
    TransformMapValues('Gender', {'Male': 0, 'Female': 1}),
    TransformMapValues('MariStat', {'Other': 0, 'Alone': 1}),
    TransformAgeGender('DrivAge', 'Gender', 'Age_m', 'Age_f', age_default=18, gender_male=0, gender_female=1),
    TransformPolynomizer('Age_m'),
    TransformPolynomizer('Age_f'),
])

InsTransforms.ins_transform()
InsTransforms.save('transforms.pkl')
```

## AutoFill NA Transforms

The AutoFillNATransforms allows to fill NA values in dataset. 
For numerical columns, it fills NA values with median values and for categorical columns - with most frequently used values.

```python
import pandas as pd
from insolver.transforms.InsolverTransforms import AutoFillNATransforms

InsDataFrame = InsolverDataFrame(pd.read_csv('freMPL-R.csv', low_memory=False))

InsTransforms = InsolverTransforms(InsDataFrame, [
    AutoFillNATransforms(),
])

InsTransforms.ins_transform()
InsTransforms.save('transforms.pkl')

```

## Encoder Transforms

EncoderTransforms based on [sklearn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Encode target labels with value between 0 and n_classes-1.

## OneHotEncoder Transforms

OneHotEncoderTransforms based on [sklearn's OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). Encode categorical features as a one-hot numeric array.

## Create users' transformations

Users' transformations can be created in special module `user_transforms.py`, which must be create in your project directory (exact name also required).

In this module you can create you own transformation classes.

The custom class must inherit from the `InsolverTransformMain` class and have `__call__` method:

```python
# user_transforms.py
import pandas as pd
from insolver.InsolverTransforms import InsolverTransformMain

class TransformToNumeric(InsolverTransformMain):
    def __init__(self, column_param, downcast='integer'):
        self.priority = 0
        super().__init__()
        self.column_param = column_param
        self.downcast = downcast

    def __call__(self, df):
        df[self.column_param] = pd.to_numeric(df[self.column_param], downcast=self.downcast)
        return df
```
