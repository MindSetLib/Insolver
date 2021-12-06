# API References

## Insolver DataFrame

```{eval-rst}
 .. autoclass:: insolver.frame.InsolverDataFrame
   :members:
```

## Build-in transformations

### Person data methods

- TransformGenderGetFromName

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformGenderGetFromName
   :members:
```

- TransformAgeGetFromBirthday

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformAgeGetFromBirthday
   :members:
```

- TransformAge

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformAge
   :members:
```

- TransformAgeGender

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformAgeGender
   :members:
```

- TransformExp

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformExp
   :members:
```

- TransformAgeExpDiff

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformAgeExpDiff
   :members:
```

- TransformNameCheck

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformNameCheck
   :members:
```


### Vehicle data methods
- TransformVehPower

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformVehPower
   :members:
```

- TransformVehAgeGetFromIssueYear

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformVehAgeGetFromIssueYear
   :members:
```

- TransformVehAge

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformVehAge
   :members:
```

### Region data methods
- TransformRegionGetFromKladr

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformRegionGetFromKladr
   :members:
```

### Sorting data methods
- TransformParamUselessGroup

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformParamUselessGroup
   :members:
```

- TransformParamSortFreq

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformParamSortFreq
   :members:
```

- TransformParamSortAC

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformParamSortAC
   :members:
```

### Other data methods
- TransformToNumeric

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformToNumeric
   :members:
```

- TransformMapValues

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformMapValues
   :members:
```

- TransformPolynomizer

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformPolynomizer
   :members:
```

- TransformGetDummies

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformGetDummies
   :members:
```

- TransformCarFleetSize

```{eval-rst}
 .. autoclass:: insolver.transforms.TransformCarFleetSize
   :members:
```

### Fill NA and Encode methods
- AutoFillNATransforms

```{eval-rst}
 .. autoclass:: insolver.transforms.AutoFillNATransforms
   :members:
```

- EncoderTransforms

```{eval-rst}
 .. autoclass:: insolver.transforms.EncoderTransforms
   :members:
```

- OneHotEncoderTransforms

```{eval-rst}
 .. autoclass:: insolver.transforms.OneHotEncoderTransforms
   :members:
```

### Date and Datetime

```{eval-rst}
.. autoclass:: insolver.transforms.DatetimeTransforms
   :members:
```

## Selection
### Feature selection

```{eval-rst}
 .. autoclass:: insolver.selection.FeatureSelection
   :members:
```
### Dimensionality Reduction

```{eval-rst}
 .. autoclass:: insolver.selection.DimensionalityReduction
   :members:
```
### Sampling

```{eval-rst}
 .. autoclass:: insolver.selection.Sampling
   :members:
```

## Model Wrappers

### Base Wrapper

```{eval-rst}
 .. autoclass:: insolver.wrappers.base.InsolverBaseWrapper
   :members:
```

### Trivial Wrapper


```{eval-rst}
 .. autoclass:: insolver.wrappers.InsolverTrivialWrapper
   :members:
   :inherited-members:
```

### Generalized Linear Model Wrapper

```{eval-rst}
 .. autoclass:: insolver.wrappers.InsolverGLMWrapper
   :members:
   :inherited-members:
```

### Gradient Boosting Machine Wrapper

```{eval-rst}
 .. autoclass:: insolver.wrappers.InsolverGBMWrapper
   :members:
   :inherited-members:
```

### Random Forest Wrapper

```{eval-rst}
 .. autoclass:: insolver.wrappers.InsolverRFWrapper
   :members:
   :inherited-members:
```

## Model Tools

### Model Comparison

```{eval-rst}
 .. autoclass:: insolver.model_tools.ModelMetricsCompare
   :members:
```