# API References

## Insolver DataFrame

```eval_rst
 .. autoclass:: insolver.frame.InsolverDataFrame
   :members:
```

## Build-in transformations

### Person data methods

- TransformGenderGetFromName

```eval_rst
 .. autoclass:: insolver.transforms.TransformGenderGetFromName
   :members:
```

- TransformAgeGetFromBirthday

```eval_rst
 .. autoclass:: insolver.transforms.TransformAgeGetFromBirthday
   :members:
```

- TransformAge

```eval_rst
 .. autoclass:: insolver.transforms.TransformAge
   :members:
```

- TransformAgeGender

```eval_rst
 .. autoclass:: insolver.transforms.TransformAgeGender
   :members:
```

- TransformExp

```eval_rst
 .. autoclass:: insolver.transforms.TransformExp
   :members:
```

- TransformAgeExpDiff

```eval_rst
 .. autoclass:: insolver.transforms.TransformAgeExpDiff
   :members:
```

- TransformNameCheck

```eval_rst
 .. autoclass:: insolver.transforms.TransformNameCheck
   :members:
```


### Vehicle data methods
- TransformVehPower

```eval_rst
 .. autoclass:: insolver.transforms.TransformVehPower
   :members:
```

- TransformVehAgeGetFromIssueYear

```eval_rst
 .. autoclass:: insolver.transforms.TransformVehAgeGetFromIssueYear
   :members:
```

- TransformVehAge

```eval_rst
 .. autoclass:: insolver.transforms.TransformVehAge
   :members:
```

### Region data methods
- TransformRegionGetFromKladr

```eval_rst
 .. autoclass:: insolver.transforms.TransformRegionGetFromKladr
   :members:
```

### Sorting data methods
- TransformParamUselessGroup

```eval_rst
 .. autoclass:: insolver.transforms.TransformParamUselessGroup
   :members:
```

- TransformParamSortFreq

```eval_rst
 .. autoclass:: insolver.transforms.TransformParamSortFreq
   :members:
```

- TransformParamSortAC

```eval_rst
 .. autoclass:: insolver.transforms.TransformParamSortAC
   :members:
```

### Other data methods
- TransformToNumeric

```eval_rst
 .. autoclass:: insolver.transforms.TransformToNumeric
   :members:
```

- TransformMapValues

```eval_rst
 .. autoclass:: insolver.transforms.TransformMapValues
   :members:
```

- TransformPolynomizer

```eval_rst
 .. autoclass:: insolver.transforms.TransformPolynomizer
   :members:
```

- TransformGetDummies

```eval_rst
 .. autoclass:: insolver.transforms.TransformGetDummies
   :members:
```

- TransformCarFleetSize

```eval_rst
 .. autoclass:: insolver.transforms.TransformCarFleetSize
   :members:
```

### Fill NA and Encode methods
- AutoFillNATransforms

```eval_rst
 .. autoclass:: insolver.transforms.AutoFillNATransforms
   :members:
```

- EncoderTransforms

```eval_rst
 .. autoclass:: insolver.transforms.EncoderTransforms
   :members:
```

- OneHotEncoderTransforms

```eval_rst
 .. autoclass:: insolver.transforms.OneHotEncoderTransforms
   :members:
```

## Model Wrappers

### Base Wrapper

```eval_rst
 .. autoclass:: insolver.wrappers.InsolverBaseWrapper
   :members:
```

### Trivial Wrapper


```eval_rst
 .. autoclass:: insolver.wrappers.InsolverTrivialWrapper
   :members:
```

### Generalized Linear Model Wrapper

```eval_rst
 .. autoclass:: insolver.wrappers.InsolverGLMWrapper
   :members:
```

### Gradient Boosting Machine Wrapper

```eval_rst
 .. autoclass:: insolver.wrappers.InsolverGBMWrapper
   :members:
```

### Random Forest Wrapper

```eval_rst
 .. autoclass:: insolver.wrappers.InsolverRFWrapper
   :members:
```

## Model Tools

### Model Comparison

```eval_rst
 .. autoclass:: insolver.model_tools.ModelMetricsCompare
   :members:
```