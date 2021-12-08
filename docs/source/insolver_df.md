# Insolver DataFrame

```{eval-rst}
 .. autoclass:: insolver.frame.InsolverDataFrame
   :show-inheritance: 
```

Insolver provides some data manipulation tools. To apply most of the transformations, we need to use the `InsolverDataFrame` object, which inherits the `pandas.DataFrame`, extending it with specific methods. 

## Example
An example of creating `InsolverDataFrame` from pandas `DataFrame`:

```python
import pandas as pd
from insolver import InsolverDataFrame

df = pd.read_csv('file_path')

InsDataFrame = InsolverDataFrame(df)
InsDataFrame.get_meta_info()
```

