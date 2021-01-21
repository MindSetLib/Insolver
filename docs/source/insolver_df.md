# Insolver DataFrame

Create from pandas df:

```python
import pandas as pd
from insolver import InsolverDataFrame

df = pd.read_csv(file_path)

InsDataFrame = InsolverDataFrame(df)
InsDataFrame.get_meta_info()
```
