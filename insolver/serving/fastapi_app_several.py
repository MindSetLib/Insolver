import os

from fastapi import FastAPI
from pydantic import BaseModel


#-----


import pickle

import pandas as pd
from fastapi.encoders import jsonable_encoder
from sympy import sympify

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper
from insolver.configs.settings import *

import re
import glob

# For logging
import logging
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



# print(FORMULA_CALCULATION)
# print(FORMULA)
# print(VARIABLES_LIST)
# print(N_CORES)


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
    # print(vlist)


# --------------------



def pool_inference(pack):
    i = pack[1]
    df = pack[0]
    # print('index', i)
    InsDataFrame = InsolverDataFrame(df)
    InsTransforms = InsolverTransform(InsDataFrame, tlist[i])
    InsTransforms.ins_transform()
    predicted = mlist[i].predict(InsTransforms)
    return [i, predicted[0]]



app = FastAPI()



class Data(BaseModel):
    df: dict


@app.get("/")
def index():
    return "API for predict service"


@app.post("/predict")
def predict(data: Data):


    start_prediction = time()

    data_dict = data.dict()
    df = pd.DataFrame(data_dict['df'])

    pack = list(zip([df for i in range(0, len(mlist))], [i for i in range(0, len(mlist))]))

    with Pool(N_CORES) as p:
        result_pool = p.map(pool_inference, pack)


    for i, vari in enumerate(vlist):
        dict_variables[vari] = result_pool[i][1]

    formula_sympy = sympify(FORMULA)

    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)

    result = {
        'result': float(formula_sympy.subs(dict_variables).evalf()),
        'duration': duration
    }


    return jsonable_encoder(result)


