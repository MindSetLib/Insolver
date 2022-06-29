# Docker image for ML serving container

## Build a docker image
You need to set your model and transformations files as build args:
```shell
docker build -t insolver --build-arg MODEL=insolver_glm_h2o_1610467176142 --build-arg TRANSFORMS=transforms.pkl .
```

### Default version of insolver package is 0.4.11, but you can overwrite it with `INSOLVER_VER` arg:
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
  --data '
    "df": {
        "ID": "A-2707438",
        "Source": "Bing",
        "TMC": "None",
        "Severity": 2,
        "Start_Time": "2017-07-07 11:12:16",
        "End_Time": "2017-07-07 17:12:16",
        "Start_Lat": 40.64359,
        "Start_Lng": -74.65959,
        "End_Lat": 40.64265,
        "End_Lng": -74.63877,
        "Distance(mi)": 1.093,
        "Description": "At I-287/Exit 29 - Accident.",
        "Number": "None",
        "Street": "I-78 E",
        "Side": "R",
        "City": "Bedminster",
        "County": "Somerset",
        "State": "NJ",
        "Zipcode": "07921",
        "Country": "US",
        "Timezone": "US/Eastern",
        "Airport_Code": "KSMQ",
        "Weather_Timestamp": "2017-07-07 10:53:00",
        "Temperature(F)": 71.1,
        "Wind_Chill(F)": "None",
        "Humidity(%)": 93.0,
        "Pressure(in)": 29.77,
        "Visibility(mi)": 7.0,
        "Wind_Direction": "North",
        "Wind_Speed(mph)": 5.8,
        "Precipitation(in)": 0.36,
        "Weather_Condition": "Light Rain",
        "Amenity": "False",
        "Bump": "False",
        "Crossing": "False",
        "Give_Way": "False",
        "Junction": "False",
        "No_Exit": "False",
        "Railway": "False",
        "Roundabout": "False",
        "Station": "False",
        "Stop": "False",
        "Traffic_Calming": "False",
        "Traffic_Signal": "False",
        "Turning_Loop": "False",
        "Sunrise_Sunset": "Day",
        "Civil_Twilight": "Day",
        "Nautical_Twilight": "Day",
        "Astronomical_Twilight": "Day"
    }
}' \
  http://localhost:5000/predict
```
