import os

import pandas as pd
from flask import Flask, request, jsonify

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, load_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper
from insolver.serving import utils

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = Flask(__name__)

# Load model
model = utils.load_pickle_model(model_path)
if model and model.algo == 'gbm':
    model = InsolverGBMWrapper(backend=model.backend, load_path=model_path)
elif model and model.algo == 'glm':
    model = InsolverGLMWrapper(backend='sklearn', load_path=model_path)
else:
    model = InsolverGLMWrapper(backend='h2o', load_path=model_path)

# load transformations
tranforms = load_transforms(transforms_path)


@app.route("/")
def index():
    return "API for predict service"


@app.route('/favicon.ico')
def favicon():
    return ''


@app.route("/predict", methods=['POST'])
def predict():
    # Request logging
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()

    # json request
    data_dict = request.json
    df = pd.DataFrame(data_dict['df'], index=[0])
    insdataframe = InsolverDataFrame(df)
    # Apply transformations
    instransforms = InsolverTransform(insdataframe, tranforms)
    instransforms.ins_transform()

    # Prediction
    predicted = model.predict(instransforms)

    result = {'predicted': predicted.tolist()}

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
    logger.error(
        '%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
        current_datatime,
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path,
        error_message,
    )
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
