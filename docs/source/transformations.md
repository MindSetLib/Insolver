# Transformations

Transformations allow you to preprocess data, save your preprocessing steps to pickle file and using it in the model inference

## Build-in transformations

Example of using transformations

```python
import pandas as pd

from insolver import InsolverDataFrame
from insolver.transforms import (
    TransformExp,
    InsolverTransform,
    TransformAge,
    TransformMapValues,
    TransformPolynomizer,
    TransformAgeGender,
)

InsDataFrame = InsolverDataFrame(pd.read_csv('freMPL-R.csv', low_memory=False))

InsTransforms = InsolverTransform(InsDataFrame, [
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

### AutoFill NA Transforms

The AutoFillNATransforms allows to fill NA values in dataset. 
For numerical columns, it fills NA values with median values and for categorical columns - with most frequently used values.

```python
import numpy as np
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, AutoFillNATransforms

df = InsolverDataFrame(pd.DataFrame(data={'col1': [1, 2, np.nan]}))

print(df)
#    col1
# 0   1.0
# 1   2.0
# 2   NaN

df_transformed = InsolverTransform(df, [
        AutoFillNATransforms(),
    ])
df_transformed.ins_transform()

print(df_transformed)
#    col1
# 0   1.0
# 1   2.0
# 2   1.5
```

### Encoder Transforms

EncoderTransforms based on [sklearn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Encode target labels with value between 0 and n_classes-1.

```python
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, EncoderTransforms


df = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A']}))

print(df)
#   col1
# 0    A
# 1    B
# 2    C
# 3    A

df_transformed = InsolverTransform(df, [
    EncoderTransforms(['col1']),
])
df_transformed.ins_transform()

print(df_transformed)
#    col1
# 0     0
# 1     1
# 2     2
# 3     0
```

### OneHotEncoder Transforms

OneHotEncoderTransforms based on [sklearn's OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). Encode categorical features as a one-hot numeric array.

```python
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, OneHotEncoderTransforms


df = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A']}))

print(df)
#   col1
# 0    A
# 1    B
# 2    C
# 3    A

df_transformed = InsolverTransform(df, [
    OneHotEncoderTransforms(['col1']),
])
df_transformed.ins_transform()

print(df_transformed)
#    col1_A  col1_B  col1_C
# 0     1.0     0.0     0.0
# 1     0.0     1.0     0.0
# 2     0.0     0.0     1.0
# 3     1.0     0.0     0.0
```

## Custom transformations

Users' transformations can be created in special module `user_transforms.py`, which must be created in your project directory (exact name also required).

In this module you can create you own transformation classes.

The custom class must have `__call__` method, which gets initial dataframe and returns transformed one:

```python
# user_transforms.py
import pandas as pd

class TransformToNumeric:
    """Example of user-defined transformations. Transform values to numeric.

    Attributes:
        column_names (list): List of columns for transformations
        downcast (str): parameter from pd.to_numeric, default: 'float'
    """
    def __init__(self, column_names, downcast='float'):
        self.column_names = column_names
        self.downcast = downcast

    def __call__(self, df):
        for column in self.column_names:
            df[column] = pd.to_numeric(df[column], downcast=self.downcast)
        return df
```

After that you can user-defined transformations the same way as the build-in transformations:

```python
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform
from user_transforms import TransformToNumeric

df = InsolverDataFrame(pd.DataFrame(data={'col1': ['1.0', '2', -3]}))

print(df)
print(df.dtypes)
#   col1
# 0  1.0
# 1    2
# 2   -3
# col1    object
# dtype: object

df_transformed = InsolverTransform(df, [
    TransformToNumeric(['col1']),
])
df_transformed.ins_transform()

print(df_transformed)
print(df_transformed.dtypes)
#    col1
# 0   1.0
# 1   2.0
# 2  -3.0
# col1    float32
# dtype: object
```
