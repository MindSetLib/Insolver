# Insolver DataFrame

Insolver provides some data manipulation tools. To apply most of the transformations, we need to use the `InsolverDataFrame` object, which inherits the `pandas.DataFrame`, extending it with some specific methods. 

```eval_rst
 .. autoclass:: insolver.frame.InsolverDataFrame
   :members:
 
 ```

## Example
An example of creating `InsolverDataFrame` from pandas `DataFrame`:

```python
import pandas as pd
from insolver import InsolverDataFrame

df = pd.read_csv(file_path)

InsDataFrame = InsolverDataFrame(df)
InsDataFrame.get_meta_info()
```

