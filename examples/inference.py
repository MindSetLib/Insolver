import json
import pandas as pd

from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform, load_transforms
from insolver.wrappers import InsolverGLMWrapper

# load data
with open('request_example.json', 'r') as file:
    data_dict = json.load(file)
df = pd.DataFrame(data_dict['df'], index=[0])
InsDataFrame = InsolverDataFrame(df)

# load transformations
transforms = load_transforms('transforms.pickle')

# Apply transformations
InsTransforms = InsolverTransform(InsDataFrame, transforms)
InsTransforms.ins_transform()

# Load saved model
new_iglm = InsolverGLMWrapper(backend='h2o', load_path='insolver_glm_model.h2o')

predict_glm = new_iglm.predict(df)
print(predict_glm)
