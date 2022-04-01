# Interpretation
Several motivations exist for having an interpretability layer for ML models, including: a) being able to justify predictions, b) being able to diagnose vulnerabilities of models, to be able to c) improve them, and c) to understand better our reality, as when algorithms discover relevant patterns or relations, we can build upon them to generate new knowledge. ([source](https://arxiv.org/pdf/2104.04144.pdf))

The interpretation module provides various tools for understanding your models. These tools are Model-Agnostic, meaning they do not depend on the specific model used. 
Here is a list of methods available in this module:
- Partial Dependence Plot (PDP), 
- Individual Condition Expectation (ICE), 
- Accumulated Local Effects (ALE),
- Diverse Counterfactual Explanations (DiCE),
- Local Interpretable Model-Agnostic Explanations (LIME).

The interpretation module consists of three classes: `ExplanationPlot`, `DiCEExplanation`, `LimeExplanation`.

## PDP, ICE and ALE
`ExplanationPlot` is a class for creating a plot for interpretation. Partial Dependence Plot (PDP), Individual Condition Expectation (ICE), Accumulated Local Effects (ALE) are supported.
You can select the method by changing the `method` parameter: [`pdp`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay), [`ice`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay) or [`ale`](https://docs.seldon.io/projects/alibi/en/latest/methods/ALE.html).

You need x, a fitted model and a list of features (features to interpret) to create a plot for interpretation.
```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from insolver.interpretation import ExplanationPlot, DiCEExplanation, LimeExplanation

dataset = pd.read_csv(...)
x, y = dataset.drop(['y'], axis=1), dataset['y']
clr = GradientBoostingRegressor().fit(x, y)
features = x.columns
```

When all the data and model are set up, you just need to create a class instance and call the `plot()` method.
```python
ep = ExplanationPlot(method='pdp', x=x, estimator=clr, features=features)
ep.plot()
```
You can change plot by setting different parameters in the `plot()` method.
```python
ep = ExplanationPlot(method = 'ice', x=x, estimator=clr, features=features)
ep.plot(ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.2},
        pd_line_kw={"color": "tab:orange", "linestyle": "--"})
```

## DiCE
`DiCEExplanation` is a class for creating Diverse Counterfactual Explanations (DiCE). It uses [dice_ml](https://github.com/interpretml/DiCE) package. It supports only sklearn models.

You need x and y OR dataset and outcome name, a fitted model to create an explanation.
```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

dataset = pd.read_csv(...)
x, y = dataset.drop(['y'], axis=1), dataset['y']
clr = GradientBoostingRegressor().fit(x, y)
features = x.columns
```
Create a class instance and call the `show_explanation()` method with the data instance you need to explain. `model_type` patameter can be `regressor` or `classifier`. 
```python
de = DiCEExplanation(estimator=clr, model_type='regressor', x=x, y=y, continuous_features=features)
de.show_explanation(instance=x[0:1], total_CFs=3, desired_range=[50, 55])
```
You can also change the `method` parameter. Values `genetic`, `random` and `kdtree` are supported.
`show_explanation()` has different parameters to specify an explanation: `features_to_vary`, `permitted_range`, `result_type`, `return_json`
```python
de = DiCEExplanation(estimator=clr, model_type='regressor', x=dataset, #set x as dataset and outcome_name
                     continuous_features=features, 
                     outcome_name=['y'])

de.show_explanation(instance=x[0:1], total_CFs=3, desired_range=[50, 55], 
                    features_to_vary=['X1', 'X2', 'X3'],
                    permitted_range={'X3': [378, 400]})
```

## LIME
`LimeExplanation` is a class for creating Local Interpretable Model-Agnostic Explanations (LIME). Uses [lime](https://github.com/marcotcr/lime) package.

You need x and a fitted model to create an interpretation.
```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

dataset = pd.read_csv(...)
x, y = dataset.drop(['y'], axis=1), dataset['y']
clr = GradientBoostingRegressor().fit(x, y)
features = x.columns
```

Create a class instance and call the `show_explanation()` method with the data instance you need to explain. `mode` patameter can be `regression` or `classification`. Categorical features are set in the `categorical_features` parameter.
```python
le = LimeExplanation(estimator=clr, x=x, mode='regression', features_names=features)
le.show_explanation(instance=x.iloc[1])
```
You can change the `result_type` parameter, values `html`, `map`, `list`, `file`, `show_in_notebook` are supported.
```python
le.show_explanation(instance=x.iloc[1], result_type='html')
```