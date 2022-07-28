import pandas as pd

from insolver import InsolverDataFrame
from insolver.transforms import (
    TransformExp,
    InsolverTransform,
    TransformAge,
    TransformMapValues,
    TransformPolynomizer,
    TransformAgeGender,
)
from insolver.wrappers import InsolverGLMWrapper
from insolver.model_tools import download_dataset

download_dataset('freMPL-R')
df = pd.read_csv('./datasets/freMPL-R.csv', low_memory=False)
df = df[df.Dataset.isin([5, 6, 7, 8, 9])]
df.dropna(how='all', axis=1, inplace=True)
df = df[df.ClaimAmount > 0]

InsDataFrame = InsolverDataFrame(df.copy())

InsTransforms = InsolverTransform(
    InsDataFrame,
    [
        TransformAge('DrivAge', 18, 75),
        TransformExp('LicAge', 57),
        TransformMapValues('Gender', {'Male': 0, 'Female': 1}),
        TransformMapValues('MariStat', {'Other': 0, 'Alone': 1}),
        TransformAgeGender('DrivAge', 'Gender', 'Age_m', 'Age_f', age_default=18, gender_male=0, gender_female=1),
        TransformPolynomizer('Age_m'),
        TransformPolynomizer('Age_f'),
    ],
)

InsTransforms.ins_transform()
InsTransforms.save('transforms.pickle')

train, valid, test = InsTransforms.split_frame(val_size=0.15, test_size=0.15, random_state=0, shuffle=True)
features = [
    'LicAge',
    'Gender',
    'MariStat',
    'DrivAge',
    'HasKmLimit',
    'BonusMalus',
    'RiskArea',
    'Age_m',
    'Age_f',
    'Age_m_2',
    'Age_f_2',
]
target = 'ClaimAmount'
x_train, x_valid, x_test = train[features], valid[features], test[features]
y_train, y_valid, y_test = train[target], valid[target], test[target]

params = {'lambda': [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0], 'alpha': [i * 0.1 for i in range(0, 11)]}

df.loc[test.index.tolist()].sample(1).to_json('request_example.json', orient='records')

iglm = InsolverGLMWrapper(backend='h2o', family='gamma', link='log')
iglm.optimize_hyperparam(params, x_train, y_train, X_valid=x_valid, y_valid=y_valid)

predict_glm = iglm.predict(x_test)
iglm.save_model(name='insolver_glm_model')
print(predict_glm)
