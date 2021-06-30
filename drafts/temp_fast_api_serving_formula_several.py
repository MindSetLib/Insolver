import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insolver import InsolverDataFrame
from insolver.serving import utils
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper


#-----

import json
import pickle

import pandas as pd
from flask import Flask, request, jsonify
from sympy import sympify

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper
from insolver.serving import utils

from configs import settings
from configs.settings import *
import re
import glob

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time
from datetime import datetime

import multiprocessing as mp
from multiprocessing import Pool

import uvicorn


# /home/frank/PycharmProjects/Insolver/drafts/several_models/models/cf1_model
model_path = '/home/frank/PycharmProjects/Insolver/drafts/several_models/models/cf1_model'  # os.environ['model_path']
transforms_path = '/home/frank/PycharmProjects/Insolver/drafts/several_models/transforms/cf1_model.pkl'  # os.environ['transforms_path']

# add new features
models_folder = '/home/frank/PycharmProjects/Insolver/drafts/several_models/models'  # os.environ['models_folder']
transforms_folder = '/home/frank/PycharmProjects/Insolver/drafts/several_models/transforms'  # os.environ['transforms_folder']
config_file = '/home/frank/PycharmProjects/Insolver/configs/settings.py'  # drafts/several_models/transforms#os.environ['config_file']


# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)



print(FORMULA_CALCULATION)
print(FORMULA)
print(VARIABLES_LIST)
print(N_CORES)


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

# var 1 to upload models
models = [f for f in glob.glob(path_models + '/*')]
models.sort()

# var 2 to upload models
transforms = [f for f in glob.glob(path_transforms + '/*')]
transforms.sort()

print('models:', models)
print('transforms:', transforms)

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


# --------------------


# def f(pack):
#     i = pack[1]
#     print('index', i)
#     df = pack[0]
#     InsDataFrame = InsolverDataFrame(df)
#     InsTransforms = InsolverTransform(InsDataFrame, tlist[i])
#     InsTransforms.ins_transform()
#     predicted = mlist[i].predict(InsTransforms)
#     return [i, predicted[0]]

def f(pack):
    i = pack[1]
    print('index', i)
    df = pack[0]
    InsDataFrame = InsolverDataFrame(df)
    InsTransforms = InsolverTransform(InsDataFrame, tlist[i])
    InsTransforms.ins_transform()
    #predicted = mlist[i].predict(InsTransforms)
    return i #[i, predicted[0]]


# def run_map():
#     with Pool(5) as p:
#         print(p.map(f, [1, 2, 3]))



app = FastAPI()



class Data(BaseModel):
    df: dict


@app.get("/")
def index():
    return "API for predict service"


@app.post("/predict")
async def predict(data: Data):

    #
    #
    # current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    # ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    # logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()
    #
    # #json_input = request.json
    # #json_str = json.dumps(json_input['df'])
    #
    # # Extract data in correct order
    data_dict = data.dict()
    df = pd.DataFrame(data_dict['df'])
    # #
    # #
    #print(len(mlist))
    pack = list(zip([df for i in range(0, len(mlist))], [i for i in range(0, len(mlist))]))
    # #
    # with Pool(1) as p:
    #     result_pool = p.map(f, pack)

    for i in range(0, len(mlist)):
        #i = pack[i][1]
        # print('index', i)

        df = pd.DataFrame(data_dict['df'])

        # df = pack[i][0]
        InsDataFrame = InsolverDataFrame(df)
        InsTransforms = InsolverTransform(InsDataFrame, tlist[i])
        InsTransforms.ins_transform()
        predicted = mlist[i].predict(InsTransforms)
        # print(predicted)
    # #
    # #
    # for i, vari in enumerate(vlist):
    #     dict_variables[vari] = result_pool[i][1]
    # #
    # #
    # formula_sympy = sympify(FORMULA)
    # result = float(formula_sympy.subs(dict_variables).evalf())
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)

    print(duration)

    return duration


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6000, log_level="info")
