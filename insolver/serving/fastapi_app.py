import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insolver import InsolverDataFrame
from insolver.serving import utils
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper

model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']

# Load model
model = utils.load_pickle_model(model_path)
if model and model.algo == 'gbm':
    model = InsolverGBMWrapper(backend=model.backend, load_path=model_path)
elif model and model.algo == 'glm':
    model = InsolverGLMWrapper(backend='sklearn', load_path=model_path)
else:
    model = InsolverGLMWrapper(backend='h2o', load_path=model_path)

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
        'predicted': predicted.tolist()
    }
    return result
