# Serving several ML models and formula usage inside

## Run serving service

To start serving the trained ML model, you need the files with the saved ML model and transformations.

The following CLI command will create the API server with the saved model:

```shell
insolver_serving -service sflask -configfile settings.py -models_folder several_models/models/ -transforms_folder several_models/transforms/
```

You can choose the server based on Flask or FastApi with the option `service`:
- `-service sflask`
- `-service sfastapi`

The default endpoint is `http://localhost:8000/predict`, but you can change it with the parameters `-ip` and `-port`.

For example:
```shell
insolver_serving -service sflask -ip 127.0.0.1 -port 5040 -configfile settings.py -models_folder several_models/models/ -transforms_folder several_models/transforms/ -transforms_module path_to_transforms_module
```


## Using serving service

After starting the ML serving service, you can use it via REST API.


Example of the request to `http://127.0.0.10:5040/predict` endpoint:
```json

{"df": 
  {"Exposure":{"145813":0.617},
    "LicAge":{"145813":602},
    "RecordBeg":{"145813":"2004-05-19"},
    "RecordEnd":{"145813":"2009-05-19"},
    "Gender":{"145813":"Male"},
    "MariStat":{"145813":"Other"},
    "SocioCateg":{"145813":"CSP60"},
    "VehUsage":{"145813":"Private"},
    "DrivAge":{"145813":68},
    "HasKmLimit":{"145813":0},
    "BonusMalus":{"145813":50},
    "ClaimAmount":{"145813":5377.204531722},
    "ClaimInd":{"145813":1},
    "Dataset":{"145813":5},
    "ClaimNbResp":{"145813":1.0},
    "ClaimNbNonResp":{"145813":0.0},
    "ClaimNbParking":{"145813":1.0},
    "ClaimNbFireTheft":{"145813":0.0},
    "ClaimNbWindscreen":{"145813":1.0},
    "OutUseNb":{"145813":0.0},
    "RiskArea":{"145813":4.0}}
}

```

LightGBM Models to this example can be found [here](https://drive.google.com/file/d/1fZ-93xghPBfoxxHAGZaduAh7qVdED0Pk/view?usp=sharing).
