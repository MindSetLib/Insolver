import os
import pickle
import zipfile
import json

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper

model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']

if not os.path.exists('models_serv'):
    os.makedirs('models_serv')

with zipfile.ZipFile(model_path, 'r') as zip_model:
    zip_model.extractall('models_serv')

model_bin_path = os.path.join (os.path.abspath(os.getcwd()), os.path.join('models_serv', os.path.basename(model_path)[:-4]))
params_path = os.path.join('models_serv', os.path.basename(model_path)[:-4] + '.json')
with open(params_path, 'r') as file:
    params = json.load(file)

# with zipfile.ZipFile(model_path, 'r') as zip_archive:
#     params = zip_archive.read(os.path.basename(model_path)[:-4] + '.json')
#     model = zip_archive.read(os.path.basename(model_path)[:-4])
#
# params = json.loads(params)

if params['algo'] == 'gbm':
    model = InsolverGBMWrapper(backend=params['backend'], load_path=model_bin_path)
elif params['algo'] == 'glm':
    model = InsolverGLMWrapper(backend=params['backend'], load_path=model_bin_path)
else:
    raise RuntimeError("Unknown model")


# load and init transformations
with open(transforms_path, 'rb') as file:
    tranforms = pickle.load(file)
tranforms = init_transforms(tranforms, inference=True)


app = FastAPI()


class Data(BaseModel):
    df: dict


@app.get("/")
def index():
    return "API for predict service"


@app.post("/predict")
async def predict(data: Data):
    # Extract data in correct order
    data_dict = data.dict()
    df = pd.DataFrame(data_dict['df'])

    InsDataFrame = InsolverDataFrame(df)
    # Apply transformations
    InsTransforms = InsolverTransform(InsDataFrame, tranforms)
    InsTransforms.ins_transform()

    # Prediction
    predicted = model.predict(df)

    result = {
        'predicted': str(predicted)
    }
    return result
