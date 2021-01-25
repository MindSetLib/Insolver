# Serving ML models

## Run serving service

To start serving the trained ML model you need the files with saved ML model and transformations.

The following CLI command will create the API server with the saved model:

```shell
insolver_serving -model path_to_model -transforms path_to_transforms  -service flask
```

You can choose the server based on Flask or FastApi with the option `service`:
- `-service flask`
- `-service fastapi`

The default endpoint is `http://localhost:8000/predict`, but you can change it with parameters `-ip` and `-port`.

For example:
```shell
insolver_serving -model path_to_model -transforms path_to_transforms  -service flask -ip 127.0.0.10 -port 5000
```


## Using serving service

After starting the ML serving service you can use it via REST API.


Example of the request to `http://127.0.0.10:5000/predict` endpoint:
```json
{
    "df": {
        "LicAge": {
            "162849": 57
        },
        "Gender": {
            "162849": 0
        },
        "MariStat": {
            "162849": 0
        },
        "DrivAge": {
            "162849": 31
        },
        "HasKmLimit": {
            "162849": 0
        },
        "BonusMalus": {
            "162849": 80
        },
        "RiskArea": {
            "162849": 6
        },
        "Age_m": {
            "162849": 31
        },
        "Age_f": {
            "162849": 18
        },
        "Age_m_2": {
            "162849": 961
        },
        "Age_f_2": {
            "162849": 324
        },
        "ClaimAmount": {
            "162849": 349.4447129909
        }
    }
}

```

You can also create a request from by a random sample from InsolverDataFrame:

```python
import pandas as pd

from insolver import InsolverDataFrame

df = pd.read_csv('datasets/US_Accidents_June20.csv', low_memory=False)
InsDataFrame = InsolverDataFrame(df)
request_data = InsDataFrame.sample_request(batch_size=1)
```


You can find full example with model training and serving [here](examples.md).
