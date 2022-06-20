import os
import pickle

from insolver.transforms import init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper
from insolver.serving import utils

from django.conf import settings


model_path = os.environ['model_path']
transforms_path = os.environ['transforms_path']
module_path = os.environ['module_path']


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
tranforms = init_transforms(tranforms, module_path=module_path, inference=True)
