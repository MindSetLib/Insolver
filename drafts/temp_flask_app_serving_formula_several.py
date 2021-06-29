import glob
import json
import os
import pickle
from sympy import sympify
import re

from configs import settings
from configs.settings import *

import glob
import json

import pandas as pd
from flask import Flask, request, jsonify

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper
from insolver.serving import utils

#               /home/frank/PycharmProjects/Insolver/drafts/several_models/models/cf1_model
model_path = '/home/frank/PycharmProjects/Insolver/drafts/several_models/models/cf1_model' #os.environ['model_path']
transforms_path = '/home/frank/PycharmProjects/Insolver/drafts/several_models/transforms/cf1_model.pkl' #os.environ['transforms_path']

# add new features
models_folder = '/home/frank/PycharmProjects/Insolver/drafts/several_models/models' #os.environ['models_folder']
transforms_folder = '/home/frank/PycharmProjects/Insolver/drafts/several_models/transforms' #os.environ['transforms_folder']
config_file = '/home/frank/PycharmProjects/Insolver/configs/settings.py' #drafts/several_models/transforms#os.environ['config_file']

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = Flask(__name__)

print(FORMULA_CALCULATION)
print(FORMULA)
print(VARIABLES_LIST)



def load_pickle_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except pickle.UnpicklingError:
        return
    return model


path_models = models_folder
path_transforms = transforms_folder

models = []
transforms = []

#var 1 to upload models
models = [f for f in glob.glob(path_models+'/*')]
models.sort()

#var 2 to upload models
transforms = [f for f in glob.glob(path_transforms+'/*')]
transforms.sort()

print('models:', models)
print('transforms:', transforms)


dict_variables = {}
if FORMULA_CALCULATION:
    dict_variables = {i: 1 for i in VARIABLES_LIST}


mlist=[]
tlist=[]
vlist=[]
itlist=[]

for i, model_path in enumerate(models):
    # Load model
    print(i, model_path, models[i], transforms[i])



    model = load_pickle_model(models[i])
    if model and model.algo == 'gbm':
        model = InsolverGBMWrapper(backend=model.backend, load_path=models[i])
    elif model and model.algo == 'glm':
        model = InsolverGLMWrapper(backend='sklearn', load_path=models[i])
    else:
        model = InsolverGLMWrapper(backend='h2o', load_path=models[i])

    mlist.append(model)

    # load and init transformations
    with open(transforms[i], 'rb') as file:
        transformations = pickle.load(file)
    transformations = init_transforms(transformations, inference=True)

    tlist.append(transformations)


    regex = re.split(r'[\s*+()/_\s]\s*', models[i])
    current_variable_model = list(filter(None, regex))[-2]

    regex = re.split(r'[\s*+()/_\s]\s*', transforms[i])
    current_variable_transform = list(filter(None, regex))[-2]

    vlist.append(current_variable_model)
    print(vlist)
#--------------------



from datetime import datetime
import time





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
    df = pd.read_json(json_str)
    InsDataFrame = InsolverDataFrame(df)


    result=1

    # Response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    print(duration)
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



# start_time = datetime.now()
#     print(start_time)
#
#     json_input = request.json
#     json_str = json.dumps(json_input['df'])
#     # df = pd.read_json(json_str)
#     # InsDataFrame = InsolverDataFrame(df)
#     print(json_str)
#
#
#
#
#     for i, vari in enumerate(vlist):
#         # print(i)
#
#         df = pd.read_json(json_str)
#
#
#         # print('InsDataFrame', InsDataFrame.T)
#
#         # Apply transformations
#
#         # itlist.append(InsolverTransform(InsDataFrame, tlist[i]))
#         # itlist[i].ins_transform()
#         InsDataFrame = InsolverDataFrame(df)
#         InsTransforms = InsolverTransform(InsDataFrame, tlist[i])
#         InsTransforms.ins_transform()
#         print(InsTransforms)
#
#         # print('InsTransforms', InsTransforms.T)
#         #
#         # # Prediction
#         predicted = mlist[i].predict(InsTransforms)
#         del InsTransforms
#
#         #
#         # result = {
#         #     'predicted': predicted.tolist()
#         # }
#
#         # print(i, models[i], result, vari)
#
#         dict_variables[vari] = predicted[0]
#
#         # print(dict_variables)
#
#     formula_sympy = sympify(FORMULA)
#     result = float(formula_sympy.subs(dict_variables).evalf())
#     print(datetime.now() - start_time)
#
#     print(formula_sympy)
#     print(result)