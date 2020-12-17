# Insolver DataFrame

Create from pandas df:

```shell
from insolver.InsolverDataFrame import InsolverDataFrame

df = pd.read_csv(file_path)

InsDataFrame = InsolverDataFrame(df)
InsDataFrame.get_meta_info()
```
