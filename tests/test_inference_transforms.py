import os
import json

os.environ['model_path'] = './dev/insolver_gbm_lightgbm_1657653374832.pickle'
os.environ['transforms_path'] = './dev/transforms'

from insolver.serving.flask_app import app

app.testing = True
client = app.test_client()


def test_index_page():
    response = client.get()
    assert response.status_code == 200


with open("./dev/test_request_frempl.json", 'r') as file:
    request_json = json.load(file)


def test_flask_transforms_inference():
    with app.test_client() as c:
        response = c.post(
            '/predict',
            data=json.dumps(request_json),
            content_type='application/json',
        )
        assert response.status_code == 200
        assert response.get_data(as_text=True) == '{"predicted":[1216.4432365935293]}\n'
