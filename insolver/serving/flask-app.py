import json
import os
import pickle

import pandas as pd
from flask import Flask, request, jsonify

from insolver.InsolverDataFrame import InsolverDataFrame
from insolver.InsolverTransforms import InsolverTransforms
from insolver.InsolverUtils import init_transforms
from insolver.InsolverWrapperGLM import InsolverGLMWrapper

model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

app = Flask(__name__)

# Load model
# model_path = 'glm/Grid_GLM_Key_Frame__upload_a685662cd198b4799aee7e181b304e66.hex_model_python_1600165671228_1_model_1'
new_iglm = InsolverGLMWrapper()
new_iglm.load_model(model_path)

# load and init transformations
# transforms_path = 'transforms.pkl'
with open(transforms_path, 'rb') as file:
    tranforms = pickle.load(file)
tranforms = init_transforms(tranforms)

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@app.route("/")
def index():
    return "API for predict service"


@app.route("/predict", methods=['POST'])
def predict():
    # Request logging
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()

    json_input = request.json
    json_str = json.dumps(json_input)
    df = pd.read_json(json_str)
    InsDataFrame = InsolverDataFrame(df)
    # Apply transformations
    InsTransforms = InsolverTransforms(InsDataFrame.get_data(), tranforms)
    InsTransforms.transform()

    # Prediction
    predict_glm = new_iglm.predict(df)

    result = {
        'predict_glm': predict_glm['predict'][0]
    }

    # Response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{current_datatime} predicted for {duration} msec: {result}\n')

    return jsonify(result)


@app.errorhandler(Exception)
def exceptions(e):
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    error_message = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                 current_datatime,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 error_message)
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)