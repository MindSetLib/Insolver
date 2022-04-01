[![PyPI](https://img.shields.io/pypi/v/insolver?style=flat)](https://pypi.org/project/insolver/)
[![Documentation Status](https://readthedocs.org/projects/insolver/badge/?version=latest)](https://insolver.readthedocs.io/en/latest/?badge=latest)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/MindSetLib/Insolver/Python%20application?logo=github&label=tests)](https://github.com/MindSetLib/Insolver/actions)
[![Downloads](https://pepy.tech/badge/insolver/week)](https://pepy.tech/project/insolver)

# Insolver

Insolver is a low-code machine learning library, originally created for the insurance industry, but can be used in any other. You can find a more detailed overview [here](https://insolver.readthedocs.io/en/latest/source/overview.html).

## Installation:

```shell
pip install insolver
```

### Post-install:

To fix the display of plotly figs in jyputerlab, install:
```shell
jupyter labextension install jupyterlab-plotly
```

### Insolver is already installed in the easy access cloud via the GitHub login. Try https://mset.space with a familiar notebook-style environment.

## Examples:

- [Regression Example - FreeMLP](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the `freMPL-R` dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.
- [Regression Example - US Accidents](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the [`US Traffic Accident`](https://smoosavi.org/datasets/us_accidents) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.
- [Gradient Boosting Example - Lending Club](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Gradient%20Boosting%20Example%20-%20Lending%20Club.ipynb)
This tutorial demonstrates how to create **classification models** for the [`Lending Club`](https://www.kaggle.com/wordsforthewise/lending-club) dataset using the **Gradient Boosting libraries** and the `InsolverGBMWrapper` class.
- [Binary Classification Example - Rain in Australia Prediction](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Binary%20Classification%20Example%20-%20Rain%20in%20Australia%20Prediction.ipynb)
This tutorial demonstrates how to create **classification models** for the [`weatherAUS`](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

## Documentation:

Available [here](https://insolver.readthedocs.io/)

## Supported libraries:

Libs:
- sklearn
- H2O

Boosting models:
- XGBoost
- LightGBM
- CatBoost

Model interpretation:
- shap plots

Serving (REST-API):
- flask
- fastapi


### Run tests:
```shell
python -m pytest
```

tests with coverage:
```shell
python -m pytest --cov=insolver; coverage html; xdg-open htmlcov/index.html
```


## Contributing to Insolver:

Please, feel free to open an issue or/and suggest PR, if you find any bugs or any enhancements.

## Demo
### Example of creating models using the Insolver
![](InsolverDemo.gif)

### Example of a model production service
![](InsolverImplementation.gif)

### Example of an elyra pipeline built with the Insolver inside
![](InsolverElyraPipeline.gif)

### Contacts
frank@mind-set.ru
+79263790123
