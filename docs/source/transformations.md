# Transformations

Transformations allow you to preprocess data, save your preprocessing steps to a pickle file, and use it in the model inference.

## Build-in transformations
Several built-in transformations can help you to preprocess some specific data.

An example of using build-in transformations:

```python
import pandas as pd

from insolver import InsolverDataFrame
from insolver.transforms import (
    TransformExp,
    InsolverTransform,
    TransformAge,
    TransformMapValues,
    TransformPolynomizer,
    TransformAgeGender
)
from insolver.model_tools import download_dataset

download_dataset('freMPL-R')
InsDataFrame = InsolverDataFrame(pd.read_csv('./datasets/freMPL-R.csv', low_memory=False))

InsTransforms = InsolverTransform(InsDataFrame, [
    TransformAge('DrivAge', 18, 75),
    TransformExp('LicAge', 57),
    TransformMapValues('Gender', {'Male': 0, 'Female': 1}),
    TransformMapValues('MariStat', {'Other': 0, 'Alone': 1}),
    TransformAgeGender('DrivAge', 'Gender', 'Age_m', 'Age_f', age_default=18,
                       gender_male=0, gender_female=1),
    TransformPolynomizer('Age_m'),
    TransformPolynomizer('Age_f')
])

InsTransforms.ins_transform()
InsTransforms.save('transforms.pickle')
```

### General preprocessing
These classes are used to encode categorical values.

* class `TransformToNumeric`
    Transforms parameter values to numeric types, uses [`pandas.to_numeric`](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html).

* class `TransformGetDummies`
    Gets dummy columns of the parameter, uses [`pandas.get_dummies`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).

* class `TransformMapValues`
    Transforms parameter values according to the given dictionary.

You can also generate the polynomial features.

* class `TransformPolynomizer`
    Gets the polynomials of parameters values.

### Grouping and sorting 

* class `TransformParamUselessGroup`:
    Groups all parameter values with a small amount of data into one group. 

* class `TransformParamSortFreq`:
    Sorts by the frequency values of the chosen column.

* class `TransformParamSortAC`:
    Sorts by the average sum values of the chosen column.

### Preprocessing data about a person
These classes are used to preprocess data about a person, such as gender, age, or name.
* class `TransformGenderGetFromName`:
    For Russian names only. Gets the gender of a person from Russian second names.

* class `TransformAgeGetFromBirthday`:
    Gets the age of a person in years from birth dates.

* class `TransformAge`:
    Transform a person's age to age for a specified `age_min` (lower values are invalid) and `age_max` (exceeding values will be grouped) age. 

* class `TransformAgeGender`:
    Gets the intersection of a person's minimum age and gender.

* class `TransformNameCheck`:
    Checks if the person's first name is on the particular list. Names may concatenate surnames, first names, and last names.

### Preprocessing of insurance data 
Since Insolver was made for the insurance industry, some classes are also available to handle the driving experience, vehicle, and region data.
* class `TransformExp`:
    Transforms the minimum driving experience values in years with a grouping of values greater than `exp_max`. 

* class `TransformAgeExpDiff`:
    Transforms records with the difference between the minimum driver age and the minimum experience less than `diff_min` years, sets the minimum driver experience equal to the minimum driver age minus `diff_min` years. 

* class `TransformVehPower`:
    Transforms vehicle power values. Values under `power_min` and over `power_max` will be grouped. Values between `power_min` and `power_max` will be grouped with step `power_step`.

* class `TransformVehAgeGetFromIssueYear`:
    Get the vehicles' age (in years) by year of issue and policy start dates. 

* class `TransformVehAge`:
    Transforms vehicle age values (in years). Values over `veh_age_max` will be grouped.

* class `TransformRegionGetFromKladr`:
    Gets the region number from KLADRs.

* class `TransformCarFleetSize`:
    Calculates fleet sizes for policyholders.

### AutoFill NA values

```{eval-rst}
.. autoclass:: insolver.transforms.AutoFillNATransforms
    :show-inheritance:
```

Class `AutoFillNATransforms` is used to fill NA values in a dataset. 
It fills NA values with median values for numerical columns and the most frequently used categorical columns.

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

#### Numerical AutoFillNA methods
There are several options for the `numerical_method` parameter available to fill the NA numerical values: 
- `median` (by default) - the value separating the higher half from the lower half;
- `mean` - the sum of the values divided by the number of values;
- `mode` - the value that appears most often in a set of data values, if several values are found, the first one is used;
- `remove` - removes all columns containing NA values.

#### Categorical AutoFillNA methods
There are several options for the `categorical_method` parameter available to fill the NA categorical values: 
- `frequent` (by default) - the category that appears most often in a set of data values, if several values are found, the first one is used;
- `new_category` - creates a new category "Unknown" for NA values; 
- `imputed_column` - fills with the frequent category and creates a new `bool` column containing whether a value was imputed or not;
- `remove` - removes all columns containing NA values.

#### Using constants
You can also use constants to fill NA values using the `numerical_constants` and `categorical_constants` parameters for numerical and categorical columns.

```python
import pandas as pd
from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, AutoFillNATransforms
from insolver.model_tools import download_dataset

download_dataset('freMPL-R')
df = InsolverDataFrame(pd.read_csv('./datasets/freMPL-R.csv', low_memory=False))
transform = InsolverTransform(df, [
    AutoFillNATransforms(numerical_constants={'col1': '111'}), 
])
transform.ins_transform()

print(df)
#    col1
# 0   1.0
# 1   2.0
# 2   111
```

### Date and datetime

```{eval-rst}
.. autoclass:: insolver.transforms.DatetimeTransforms
    :show-inheritance:
```

Class `DatetimeTransforms` is used to preprocess date and date-time columns. 
Unlike other transformations, this class does not change the date columns but creates new ones with the used feature in the name.

```python
import pandas as pd

from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, DatetimeTransforms

df = InsolverDataFrame(pd.DataFrame(data={'last_review': ['2018-10-19', '2019-05-21']}))

print(df)
#    last_review
# 0   2018-10-19
# 1   2019-05-21

transform = InsolverTransform(df, [
        DatetimeTransforms(['last_review']),
    ])
    
transform.ins_transform()

print(df)
#    last_review last_review_unix
# 0   2018-10-19   1.539907e+09     
# 1   2019-05-21   1.558397e+09
```

### Label Encoder

EncoderTransforms based on [sklearn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Encode target labels with value between 0 and `n_classes`-1.

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

### One Hot Encoder

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

Custom transformations can also be created. Transfromation can be defined as a `class` object, which has `__init__` and `__call__` methods.
The `__call__` method should take one argument, which is the initial dataframe, and return transformed one.
Also, all packages and assets used in the custom transformation should be imported explicitly in methods where they are used.
Otherwise, custom transformations may not work properly, since the saved transformation is serialized by `dill` package, which may not resolve all the references when transformation will be loaded.

```python
import pandas as pd
from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform


class TransformToNumeric:
    """Example of user-defined transformations. Transform values to numeric.

    Parameters:
        column_names (list): List of columns for transformations
        downcast (str): parameter from pd.to_numeric, default: 'float'
    """
    def __init__(self, column_names, downcast='float'):
        self.column_names = column_names
        self.downcast = downcast

    def __call__(self, df):
        import pandas as pd
        for column in self.column_names:
            df[column] = pd.to_numeric(df[column], downcast=self.downcast)
        return df

    
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

## Saving and loading transformations

Transformations can also be saved with `save()` method for both development and production use.

```python
import pandas as pd
from insolver.frame import InsolverDataFrame
from insolver.transforms import InsolverTransform, OneHotEncoderTransforms

df = InsolverDataFrame(pd.DataFrame(data={'col1': ['A', 'B', 'C', 'A']}))
df_transformed = InsolverTransform(df, [
    OneHotEncoderTransforms(['col1']),
])

df_transformed.ins_transform()
df_transformed.save('transforms')
```

Transformations saving is performed by serialization with `dill` package.
To use saved transformations (including user-defined) your workflow, you should pass their filepath into `load_transforms` function.
After that you can use them in the same way as the build-in imported transformations.

```python
import pandas as pd
from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, load_transforms

# load data
df = pd.read_json('request_example.json')
InsDataFrame = InsolverDataFrame(df)

# load transformations
transforms = load_transforms('transforms')
InsTransforms = InsolverTransform(InsDataFrame, transforms)
InsTransforms.ins_transform()
...
```
