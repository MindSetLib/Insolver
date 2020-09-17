import pickle

import pandas as pd

from insolver.InsolverDataFrame import InsolverDataFrame
from insolver.InsolverTransforms import InsolverTransforms
from insolver.InsolverUtils import init_transforms
from insolver.InsolverWrapperGLM import InsolverGLMWrapper

# load data
df = pd.read_json('request_example.json')
InsDataFrame = InsolverDataFrame(df)

# load and init transformations
with open('transforms.pkl', 'rb') as file:
    tranforms = pickle.load(file)

tranforms = init_transforms(tranforms)

# Apply transformations
InsTransforms = InsolverTransforms(InsDataFrame.get_data(), tranforms)
InsTransforms.transform()

# Load saved model
new_iglm = InsolverGLMWrapper()
new_iglm.load_model('glm/Grid_GLM_Key_Frame__upload_8c2f611973a7d4dfc5f929bb500e4281.hex_model_python_1600181729291_1_model_1')

predict_glm = new_iglm.predict(df)
print(predict_glm['predict'][0])
