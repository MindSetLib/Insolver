import json
import os

import pandas as pd
from flask import Flask, request, jsonify
from sympy import sympify

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, load_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper
from insolver.serving import utils
from insolver.configs.settings import FORMULA, FORMULA_CALCULATION, N_CORES, VARIABLES_LIST

import re
import glob

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

from multiprocessing import Pool


# add new features
if os.environ['models_folder'] is not None:
    models_folder = os.environ['models_folder']
if os.environ['transforms_folder'] is not None:
    transforms_folder = os.environ['transforms_folder']
if os.environ['config_file'] is not None:
    config_file = os.environ['config_file']

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


path_models = models_folder
path_transforms = transforms_folder

models = []
transforms = []

# var 1 to upload models
models = [f for f in glob.glob(path_models + '/*')]
models.sort()

# var 2 to upload models
transforms = [f for f in glob.glob(path_transforms + '/*')]
transforms.sort()

# print('models:', models)
# print('transforms:', transforms)

dict_variables = {}
if FORMULA_CALCULATION:
    dict_variables = {i: 1 for i in VARIABLES_LIST}

# list of key objects  of models
mlist = []
tlist = []
vlist = []
itlist = []

# Load models once
for i, model_path in enumerate(models):
    # Load model
    # print(i, model_path, models[i], transforms[i])

    model = utils.load_pickle_model(models[i])
    if model and model.algo == 'gbm':
        model = InsolverGBMWrapper(backend=model.backend, load_path=models[i])
    elif model and model.algo == 'glm':
        model = InsolverGLMWrapper(backend='sklearn', load_path=models[i])
    else:
        model = InsolverGLMWrapper(backend='h2o', load_path=models[i])

    mlist.append(model)

    # load transformations
    transformations = load_transforms(transforms[i])

    tlist.append(transformations)

    regex = re.split(r'[\s*+()/_\s]\s*', models[i])
    current_variable_model = list(filter(None, regex))[-2]

    regex = re.split(r'[\s*+()/_\s]\s*', transforms[i])
    current_variable_transform = list(filter(None, regex))[-2]

    vlist.append(current_variable_model)
    # print(vlist)


# --------------------
def pool_inference(pack):
    i = pack[1]
    # print('index', i)
    a_json = json.loads(pack[0])
    df = pd.DataFrame.from_dict(a_json, orient='index').T
    # df = pd.read_json(pack[0])
    insdataframe = InsolverDataFrame(df)
    instransforms = InsolverTransform(insdataframe, tlist[i])
    instransforms.ins_transform()
    predicted = mlist[i].predict(instransforms)
    return [i, predicted[0]]


app = Flask(__name__)
app.debug = True


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

    json_input = request.json
    json_str = json.dumps(json_input['df'])

    # print(json_input)

    pack = list([(json_str, i) for i in range(0, len(mlist))])

    with Pool(N_CORES) as p:
        result_pool = p.map(pool_inference, pack)

    for i, vari in enumerate(vlist):
        dict_variables[vari] = result_pool[i][1]

    # Response logging

    formula_sympy = sympify(FORMULA)
    result = float(formula_sympy.subs(dict_variables).evalf())

    # print(result)

    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{current_datatime} predicted for {duration} msec: {result}\n')

    result = {'result': result, 'duration': duration}

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
    app.run()  # threaded=True, port = 5000
