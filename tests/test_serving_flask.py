from flask import json
import os

os.environ['model_path'] = 'examples/serving_example/insolver_glm_h2o_1610467176142'
os.environ['transforms_path'] = 'examples/serving_example/transforms.pkl'

from insolver.serving.flask_app import app

app.testing = True
client = app.test_client()


def test_index_page():
    response = client.get()
    assert response.status_code == 200


request_json = {
    "df": {
        "Temperature(F)": {
            "755001": 52
        },
        "Humidity(%)": {
            "755001": 44
        },
        "Pressure(in)": {
            "755001": 29.71
        },
        "Visibility(mi)": {
            "755001": 10
        },
        "Wind_Direction": {
            "755001": "NNW"
        },
        "Wind_Speed(mph)": {
            "755001": 5
        },
        "Weather_Condition": {
            "755001": "Partly Cloudy"
        },
        "Amenity": {
            "755001": 0
        },
        "Bump": {
            "755001": 0
        },
        "Crossing": {
            "755001": 0
        },
        "Give_Way": {
            "755001": 0
        },
        "Junction": {
            "755001": 0
        },
        "No_Exit": {
            "755001": 0
        },
        "Railway": {
            "755001": 0
        },
        "Roundabout": {
            "755001": 0
        },
        "Station": {
            "755001": 0
        },
        "Stop": {
            "755001": 0
        },
        "Traffic_Calming": {
            "755001": 0
        },
        "Traffic_Signal": {
            "755001": 0
        },
        "Turning_Loop": {
            "755001": 0
        },
        "Sunrise_Sunset": {
            "755001": "Day"
        },
        "Civil_Twilight": {
            "755001": "Day"
        },
        "Nautical_Twilight": {
            "755001": "Day"
        },
        "Astronomical_Twilight": {
            "755001": "Day"
        }
    }
}


def test_h2o_model():
    response = app.test_client().post(
        '/predict',
        data=json.dumps(request_json),
        content_type='application/json',
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert int(data['predicted'][0]) == 1252
