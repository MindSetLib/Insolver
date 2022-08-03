# Insolver
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/insolver)
[![PyPI](https://img.shields.io/pypi/v/insolver?style=flat)](https://pypi.org/project/insolver/)
[![Documentation Status](https://readthedocs.org/projects/insolver/badge/?version=latest)](https://insolver.readthedocs.io/en/latest/?badge=latest)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/MindSetLib/Insolver/Insolver%20testing?logo=github&label=tests)](https://github.com/MindSetLib/Insolver/actions)
[![Coverage](https://codecov.io/github/MindSetLib/Insolver/coverage.svg?branch=master)](https://codecov.io/github/MindSetLib/Insolver)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/insolver/week)](https://pepy.tech/project/insolver)
<!--- [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/MindSetLib/Insolver/Python%20application?logo=github&label=tests)](https://github.com/MindSetLib/Insolver/actions) --->

Insolver is a low-code machine learning library, originally created for the insurance industry, but can be used in any other. You can find a more detailed overview [here](https://insolver.readthedocs.io/en/latest/source/overview.html).

## Installation:

Insolver can be installed via pip from PyPI. There are several installation options available:

| Description                                | Command                       |
|--------------------------------------------|-------------------------------|
| Regular installation                       | `pip install insolver`        |
| Installation with all heavy requirements   | `pip install insolver[full]`  |
| Installation with development requirements | `pip install insolver[dev]`   |


### Insolver is already installed in the easy access cloud via the GitHub login. Try https://mset.space with a familiar notebook-style environment.

## Examples:

- [Binary Classification Example - Rain in Australia Prediction](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Binary%20Classification%20Example%20-%20Rain%20in%20Australia%20Prediction.ipynb)
This tutorial demonstrates how to create **classification models** for the [`weatherAUS`](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Data Preprocessing Example I - New York City Airbnb](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Data%20Preprocessing%20Example%20I%20-%20New%20York%20City%20Airbnb.ipynb)
This tutorial demonstrates how to use the [`feature_engineering`](https://github.com/MindSetLib/Insolver/tree/master/insolver/feature_engineering) module and all the **main features of each class**. For this, the [`AB_NYC_2019`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset is used.

- [Data Preprocessing Example II - New York City Airbnb](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Data%20Preprocessing%20Example%20II%20-%20New%20York%20City%20Airbnb.ipynb)
This tutorial also demonstrates how to use the [`feature_engineering`](https://github.com/MindSetLib/Insolver/tree/master/insolver/feature_engineering) module, but it covers the **automated data preprossesing** class and all of its features. For this, the [`AB_NYC_2019`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset is used.

- [Gradient Boosting Example - Lending Club](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Gradient%20Boosting%20Example%20-%20Lending%20Club.ipynb)
This tutorial demonstrates how to create **classification models** for the [`Lending Club`](https://www.kaggle.com/wordsforthewise/lending-club) dataset using the **Gradient Boosting libraries** and the `InsolverGBMWrapper` class.

- [Transforms Inference Example](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Transforms%20Inference%20Example.ipynb)
This tutorial demonstrates how to load `InsolverTransform` transforms from a file using the `load_transforms` function.

- [InsolverDataFrame and InsolverTransform Example](https://github.com/MindSetLib/Insolver/blob/master/tutorials/InsolverDataFrame%20and%20InsolverTransform%20Example.ipynb)
This tutorial demonstrates main features of the `InsolverDataFrame` class and the `InsolverTransform` class.

- [Regression Example - FreeMLP](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the `freMPL-R` dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Regression Example - US Accidents](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the [`US Traffic Accident`](https://smoosavi.org/datasets/us_accidents) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Report Example](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Report%20Example.ipynb)
This tutorial demonstrates how to create a **HTML report** with different models using the `Report` class.

## Documentation:

Available [here](https://insolver.readthedocs.io/)

## Supported libraries:

| GLM                 | Boosting models                           | Serving (REST-API)                 | Model interpretation |
|---------------------|-------------------------------------------|------------------------------------|----------------------|
| - sklearn<br/>- h2o | - XGBoost<br/> - LightGBM<br/> - CatBoost | - Flask<br/>- FastAPI<br/>- Django | - shap plots         |

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
![](https://github.com/MindSetLib/Insolver/releases/download/v0.4.6/InsolverDemo.gif)

### Example of a model production service
![](https://github.com/MindSetLib/Insolver/releases/download/v0.4.6/InsolverImplementation.gif)

### Example of an elyra pipeline built with the Insolver inside
![](https://github.com/MindSetLib/Insolver/releases/download/v0.4.6/InsolverElyraPipeline.gif)

## Contacts
frank@mind-set.ru
+79263790123
