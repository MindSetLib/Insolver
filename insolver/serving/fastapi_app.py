import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper

model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']

# Load model
new_iglm = InsolverGLMWrapper(backend='h2o', load_path=model_path)

# load and init transformations
with open(transforms_path, 'rb') as file:
    tranforms = pickle.load(file)
tranforms = init_transforms(tranforms)


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
    predict_glm = new_iglm.predict(df)

    result = {
        'predict_glm': str(predict_glm)
    }
    return result
