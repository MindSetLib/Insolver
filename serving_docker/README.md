# Docker image for ML serving container

## Build a docker image
You need to set your model and transformations files as build args:
```shell
docker build -t insolver --build-arg MODEL=insolver_glm_h2o_1610467176142 --build-arg TRANSFORMS=transforms.pkl .
```

### Default version of insolver package is 0.4.10, but you can overwrite it with `INSOLVER_VER` arg:
docker build -t insolver \
    --build-arg MODEL=insolver_glm_h2o_1610467176142 \
    --build-arg TRANSFORMS=transforms.pkl \
    --build-arg INSOLVER_VER=0.4.10 .

## Run a container

```shell
docker run --name insolver -p 5000:5000 insolver
```

Default version of inference service is FastAPI, to start container with flask service:
```shell
docker run --name insolver -p 5000:5000 insolver -service flask
```

## OpenAPI and ReDoc (only for FastAPI service):
- http://127.0.0.1:5000/docs
- http://127.0.0.1:5000/redoc

### Test request:
```shell
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"df": {"ID": {"73490": "A-2707438"},
  "Source": {"73490": "Bing"},
  "TMC": {"73490": "None"},
  "Severity": {"73490": 2},
  "Start_Time": {"73490": "2017-07-07 11:12:16"},
  "End_Time": {"73490": "2017-07-07 17:12:16"},
  "Start_Lat": {"73490": 40.64359},
  "Start_Lng": {"73490": -74.65959},
  "End_Lat": {"73490": 40.64265},
  "End_Lng": {"73490": -74.63877},
  "Distance(mi)": {"73490": 1.093},
  "Description": {"73490": "At I-287/Exit 29 - Accident."},
  "Number": {"73490": "None"},
  "Street": {"73490": "I-78 E"},
  "Side": {"73490": "R"},
  "City": {"73490": "Bedminster"},
  "County": {"73490": "Somerset"},
  "State": {"73490": "NJ"},
  "Zipcode": {"73490": "07921"},
  "Country": {"73490": "US"},
  "Timezone": {"73490": "US/Eastern"},
  "Airport_Code": {"73490": "KSMQ"},
  "Weather_Timestamp": {"73490": "2017-07-07 10:53:00"},
  "Temperature(F)": {"73490": 71.1},
  "Wind_Chill(F)": {"73490": "None"},
  "Humidity(%)": {"73490": 93.0},
  "Pressure(in)": {"73490": 29.77},
  "Visibility(mi)": {"73490": 7.0},
  "Wind_Direction": {"73490": "North"},
  "Wind_Speed(mph)": {"73490": 5.8},
  "Precipitation(in)": {"73490": 0.36},
  "Weather_Condition": {"73490": "Light Rain"},
  "Amenity": {"73490": "False"},
  "Bump": {"73490": "False"},
  "Crossing": {"73490": "False"},
  "Give_Way": {"73490": "False"},
  "Junction": {"73490": "False"},
  "No_Exit": {"73490": "False"},
  "Railway": {"73490": "False"},
  "Roundabout": {"73490": "False"},
  "Station": {"73490": "False"},
  "Stop": {"73490": "False"},
  "Traffic_Calming": {"73490": "False"},
  "Traffic_Signal": {"73490": "False"},
  "Turning_Loop": {"73490": "False"},
  "Sunrise_Sunset": {"73490": "Day"},
  "Civil_Twilight": {"73490": "Day"},
  "Nautical_Twilight": {"73490": "Day"},
  "Astronomical_Twilight": {"73490": "Day"}}}' \
  http://localhost:5000/predict
```
