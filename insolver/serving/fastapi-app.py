import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from insolver.InsolverDataFrame import InsolverDataFrame
from insolver.InsolverTransforms import InsolverTransforms
from insolver.InsolverUtils import init_transforms
from insolver.InsolverWrapperGLM import InsolverGLMWrapper

model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']

# Load model
new_iglm = InsolverGLMWrapper()
new_iglm.load_model(model_path)

# load and init transformations
with open(transforms_path, 'rb') as file:
    tranforms = pickle.load(file)
tranforms = init_transforms(tranforms)


app = FastAPI()


class Data(BaseModel):
    LicAge: dict
    Gender: dict
    MariStat: dict
    DrivAge: dict
    HasKmLimit: dict
    BonusMalus: dict
    RiskArea: dict
    Age_m: dict
    Age_f: dict
    Age_m_2: dict
    Age_f_2: dict
    ClaimAmount: dict


@app.get("/")
def index():
    return "API for predict service"


@app.post("/predict")
def predict(data: Data):
    # Extract data in correct order
    data_dict = data.dict()
    df = pd.DataFrame(data_dict)

    InsDataFrame = InsolverDataFrame(df)
    # Apply transformations
    InsTransforms = InsolverTransforms(InsDataFrame.get_data(), tranforms)
    InsTransforms.transform()

    # Prediction
    predict_glm = new_iglm.predict(df)

    result = {
        'predict_glm': predict_glm
    }
    return result
