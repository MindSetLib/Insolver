import pickle

import pandas as pd

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper

# load data
df = pd.read_json('request_example.json')
InsDataFrame = InsolverDataFrame(df)

# load and init transformations
with open('transforms.pickle', 'rb') as file:
    transforms = pickle.load(file)

transforms = init_transforms(transforms, module_path='./user_transforms.py', inference=True)

# Apply transformations
InsTransforms = InsolverTransform(InsDataFrame, transforms)
InsTransforms.ins_transform()

# Load saved model
new_iglm = InsolverGLMWrapper(backend='h2o', load_path='insolver_glm_model.h2o')

predict_glm = new_iglm.predict(df)
print(predict_glm)
