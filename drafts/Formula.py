import itertools
from sympy import sympify

from configs import settings
from configs.settings import *

import glob
import json
import pickle
from insolver import InsolverDataFrame

from pydantic import BaseModel

from insolver import InsolverDataFrame
from insolver.serving import utils
from insolver.transforms import InsolverTransform, init_transforms
from insolver.wrappers import InsolverGLMWrapper, InsolverGBMWrapper

import pandas as pd
import re



path_models = 'several_models/models'
path_transforms = 'several_models/transforms'



models = []
transforms = []

#var 1 to upload models
models = [f for f in glob.glob(path_models+'/*')]
models.sort()

#var 2 to upload models
transforms = [f for f in glob.glob(path_transforms+'/*')]
transforms.sort()

print('models:', models)
print('transforms:', transforms)



#get variables from settings file
print(dir(settings))
print([item for item in dir(settings) if not item.startswith("__")])


print(FORMULA_CALCULATION)
print(FORMULA)
print(VARIABLES_LIST)



def load_pickle_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except pickle.UnpicklingError:
        return
    return model

request_data = {'df': {'ID': {'713016': 'A-713038'},
  'Source': {'713016': 'MapQuest-Bing'},
  'TMC': {'713016': 201.0},
  'Severity': {'713016': 2},
  'Start_Time': {'713016': '2020-02-13 09:13:53'},
  'End_Time': {'713016': '2020-02-13 10:42:18'},
  'Start_Lat': {'713016': 45.037285},
  'Start_Lng': {'713016': -93.017372},
  'End_Lat': {'713016': None},
  'End_Lng': {'713016': None},
  'Distance(mi)': {'713016': 0.0},
  'Description': {'713016': 'Accident on I-694 Eastbound at Exit 50 CR-65 White Bear Ave.'},
  'Number': {'713016': None},
  'Street': {'713016': 'I-694 W'},
  'Side': {'713016': 'R'},
  'City': {'713016': 'Saint Paul'},
  'County': {'713016': 'Ramsey'},
  'State': {'713016': 'MN'},
  'Zipcode': {'713016': '55110'},
  'Country': {'713016': 'US'},
  'Timezone': {'713016': 'US/Central'},
  'Airport_Code': {'713016': 'K21D'},
  'Weather_Timestamp': {'713016': '2020-02-13 09:15:00'},
  'Temperature(F)': {'713016': -9.0},
  'Wind_Chill(F)': {'713016': -26.0},
  'Humidity(%)': {'713016': 59.0},
  'Pressure(in)': {'713016': 29.37},
  'Visibility(mi)': {'713016': 10.0},
  'Wind_Direction': {'713016': 'NNW'},
  'Wind_Speed(mph)': {'713016': 8.0},
  'Precipitation(in)': {'713016': 0.0},
  'Weather_Condition': {'713016': 'Fair'},
  'Amenity': {'713016': False},
  'Bump': {'713016': False},
  'Crossing': {'713016': False},
  'Give_Way': {'713016': False},
  'Junction': {'713016': False},
  'No_Exit': {'713016': False},
  'Railway': {'713016': False},
  'Roundabout': {'713016': False},
  'Station': {'713016': False},
  'Stop': {'713016': False},
  'Traffic_Calming': {'713016': False},
  'Traffic_Signal': {'713016': False},
  'Turning_Loop': {'713016': False},
  'Sunrise_Sunset': {'713016': 'Day'},
  'Civil_Twilight': {'713016': 'Day'},
  'Nautical_Twilight': {'713016': 'Day'},
  'Astronomical_Twilight': {'713016': 'Day'}}}


# if FORMULA_CALCULATION:
#     dict_variables = {i: 1 for i in VARIABLES_LIST}
#
#     for key, value in dict_variables.items():
#         print(key, value)
#
#     formula_sympy = sympify(FORMULA)
#     result = float(formula_sympy.subs(dict_variables).evalf())
#
#     for i, j in itertools.product(range(1), range(1)):
#         dict_variables['cf_0_200'] = i
#         dict_variables['ac_0_200'] = j
#         result = float(formula_sympy.subs(dict_variables).evalf())
#         print(result)



dict_variables = {}
if FORMULA_CALCULATION:
    dict_variables = {i: 1 for i in VARIABLES_LIST}


mlist=[]
tlist=[]
vlist=[]

for i, model_path in enumerate(models):
    # Load model
    print(i, model_path, models[i], transforms[i])



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
        tranformations = pickle.load(file)
    tranformations = init_transforms(tranformations, inference=True)

    tlist.append(tranformations)


    regex = re.split(r'[\s*+()/_\s]\s*', models[i])
    current_variable_model = list(filter(None, regex))[-2]

    regex = re.split(r'[\s*+()/_\s]\s*', transforms[i])
    current_variable_transform = list(filter(None, regex))[-2]

    vlist.append(current_variable_model)
#--------------------

for i, vari in enumerate(vlist):
    json_input = request_data
    json_str = json.dumps(json_input['df'])
    df = pd.read_json(json_str)
    InsDataFrame = InsolverDataFrame(df)
    # Apply transformations
    InsTransforms = InsolverTransform(InsDataFrame, tlist[i])
    InsTransforms.ins_transform()
    #
    # # Prediction
    predicted = mlist[i].predict(InsTransforms)

    #
    result = {
        'predicted': predicted.tolist()
    }

    print(i, models[i], result, vari)

    dict_variables[vari] = predicted[0]

    print(dict_variables)


formula_sympy = sympify(FORMULA)
print(formula_sympy)
result = float(formula_sympy.subs(dict_variables).evalf())

print(result)







if FORMULA_CALCULATION:
    dict_variables = {i: 1 for i in VARIABLES_LIST}

    for key, value in dict_variables.items():
        print(key, value)

    formula_sympy = sympify(FORMULA)
    result = float(formula_sympy.subs(dict_variables).evalf())



 #   for i, j in itertools.product(range(1), range(1)):
 #       dict_variables['cf_0_200'] = i
 #       dict_variables['ac_0_200'] = j
 #       result = float(formula_sympy.subs(dict_variables).evalf())
 #       print(result)








# example_request = {'df': 
# {"Exposure":{"145813":0.617},"LicAge":{"145813":602},"RecordBeg":{"145813":"2004-05-19"},"RecordEnd":{"145813":null},"Gender":{"145813":"Male"},"MariStat":{"145813":"Other"},"SocioCateg":{"145813":"CSP60"},"VehUsage":{"145813":"Private"},"DrivAge":{"145813":68},"HasKmLimit":{"145813":0},"BonusMalus":{"145813":50},"ClaimAmount":{"145813":5377.204531722},"ClaimInd":{"145813":1},"Dataset":{"145813":5},"ClaimNbResp":{"145813":1.0},"ClaimNbNonResp":{"145813":0.0},"ClaimNbParking":{"145813":1.0},"ClaimNbFireTheft":{"145813":0.0},"ClaimNbWindscreen":{"145813":1.0},"OutUseNb":{"145813":0.0},"RiskArea":{"145813":4.0}}
# }