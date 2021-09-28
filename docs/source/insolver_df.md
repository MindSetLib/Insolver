# Insolver DataFrame

Insolver provides some tools for data manipulation. In order to apply most of the transformations, we need to use `InsolverDataFrame` object, which inherits the common `pd.DataFrame`, extending it with some specific methods. 

```eval_rst
 .. autoclass:: insolver.InsolverDataFrame
   :members:
 
 ```

## Example
Creation of `InsolverDataFrame`

```python
import pandas as pd
from insolver import InsolverDataFrame

df = pd.read_csv(file_path)

InsDataFrame = InsolverDataFrame(df)
InsDataFrame.get_meta_info()
```

