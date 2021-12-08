# Load test for models

## Vegeta

Useful software to make load tests is [vegeta](https://github.com/tsenart/vegeta). 

Before run vegeta test you need to build test file, for example target.lst

```
POST http://127.0.0.1:5026/predict
Content-Type: application/json
Content-Length: 47
@test.json
```

Also, we need to define json file, for example test.json:
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
   "RiskArea":{"145813":4.0}
 }
}


```


To run vegeta test you can use this code.

```bash
./vegeta attack -workers 1 -duration=5s -rate=10 -targets=target.lst -output=results-veg-httpbin-get.bin && cat results-veg-httpbin-get.bin | ./vegeta plot --title="HTTP Bin GET n rps for k seconds" > http-bin-get-nrps-kseconds.html
```
