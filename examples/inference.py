import pickle

import pandas as pd

from insolver import InsolverDataFrame
from insolver.InsolverTransforms import InsolverTransforms
from insolver.InsolverUtils import init_transforms
from insolver.wrappers import InsolverGLMWrapper

# load data
df = pd.read_json('request_example.json')
InsDataFrame = InsolverDataFrame(df)

# load and init transformations
with open('transforms.pkl', 'rb') as file:
    transforms = pickle.load(file)

transforms = init_transforms(transforms)

# Apply transformations
InsTransforms = InsolverTransforms(InsDataFrame.get_data(), transforms)
InsTransforms.transform()

# Load saved model
new_iglm = InsolverGLMWrapper(backend='h2o', load_path='./insolver_glm_h2o_1605026853331')

predict_glm = new_iglm.predict(df)
print(predict_glm)
