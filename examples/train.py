import pandas as pd

from insolver.InsolverDataFrame import InsolverDataFrame
from insolver.InsolverTransforms import (
    TransformExp,
    InsolverTransforms,
    TransformAge,
    TransformMapValues,
    TransformPolynomizer,
    TransformAgeGender,
    EncoderTransforms,
    AutoFillNATransforms,
    OneHotEncoderTransforms,
)
from insolver.InsolverWrapperGLM import InsolverGLMWrapper

df = pd.read_csv('freMPL-R.csv', low_memory=False)
# df = df[df.Dataset.isin([5, 6, 7, 8, 9])]
# df.dropna(how='all', axis=1, inplace=True)
df = df[df.ClaimAmount > 0]

InsTransforms = InsolverTransforms(df, [
    AutoFillNATransforms(),
    # TransformAge('DrivAge', 18, 75),
    # TransformExp('LicAge', 57),
    # TransformMapValues('Gender', {'Male': 0, 'Female': 1}),
    # TransformMapValues('MariStat', {'Other': 0, 'Alone': 1}),
    # EncoderTransforms(['Gender', 'MariStat']),
    OneHotEncoderTransforms(['MariStat', 'Gender']),
    # TransformAgeGender('DrivAge', 'Gender', 'Age_m', 'Age_f', age_default=18, gender_male=0, gender_female=1),
    # TransformPolynomizer('Age_m'),
    # TransformPolynomizer('Age_f'),
])

InsTransforms.transform()
InsTransforms.save('transforms.pkl')
InsTransforms.save_json('transforms.json')

train, valid, test = InsTransforms.split_frame(val_size=0.15, test_size=0.15, random_state=0, shuffle=True)

iglm = InsolverGLMWrapper()

params = {'lambda': [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0],
          'alpha': [i * 0.1 for i in range(0, 11)]}
features = ['LicAge',
            # 'Gender',
            # 'MariStat',
            'DrivAge', 'HasKmLimit', 'BonusMalus', 'RiskArea',
            'Age_m', 'Age_f', 'Age_m_2', 'Age_f_2']
target = 'ClaimAmount'

test.sample(1).to_json('request_example.json')

iglm.model_init(train, valid, family='gamma', link='log')
iglm.grid_search_cv(features, target, params, search_criteria={'strategy': "Cartesian"})
predict_glm = iglm.predict(test)
iglm.save_model('glm')
print(predict_glm)
