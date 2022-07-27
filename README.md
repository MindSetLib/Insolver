![PyPI - Python Version](https://img.shields.io/pypi/pyversions/insolver)
[![PyPI](https://img.shields.io/pypi/v/insolver?style=flat)](https://pypi.org/project/insolver/)
[![Documentation Status](https://readthedocs.org/projects/insolver/badge/?version=latest)](https://insolver.readthedocs.io/en/latest/?badge=latest)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/MindSetLib/Insolver/Python%20application?logo=github&label=tests)](https://github.com/MindSetLib/Insolver/actions)
[![Downloads](https://pepy.tech/badge/insolver/week)](https://pepy.tech/project/insolver)
<!-- [![GitHub Workflow Status](https://img.shields.io/github/workflow/status/MindSetLib/Insolver/Insolver%20testing?logo=github&label=tests)](https://github.com/MindSetLib/Insolver/actions) -->
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

# Insolver

Insolver is a low-code machine learning library, originally created for the insurance industry, but can be used in any other. You can find a more detailed overview [here](https://insolver.readthedocs.io/en/latest/source/overview.html).

## Installation:

- Insolver can be installed by running `pip install insolver`.
- If you want to install insolver with all heavy requirements, install with `pip install insolver[full]`.
- If you want to install insolver with development requirements, install with `pip install insolver[dev]`.


### Insolver is already installed in the easy access cloud via the GitHub login. Try https://mset.space with a familiar notebook-style environment.

## Examples:

- [Binary Classification Example - Rain in Australia Prediction](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Binary%20Classification%20Example%20-%20Rain%20in%20Australia%20Prediction.ipynb)
This tutorial demonstrates how to create **classification models** for the [`weatherAUS`](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Data Preprocessing Example I - New York City Airbnb](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Data%20Preprocessing%20Example%20I%20-%20New%20York%20City%20Airbnb.ipynb)
This tutorial demonstrates how to use the [`feature_engineering`](https://github.com/MindSetLib/Insolver/tree/fixed_docs/insolver/feature_engineering) module and all the **main features of each class**. For this, the [`AB_NYC_2019`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset is used.

- [Data Preprocessing Example II - New York City Airbnb](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Data%20Preprocessing%20Example%20II%20-%20New%20York%20City%20Airbnb.ipynb)
This tutorial also demonstrates how to use the [`feature_engineering`](https://github.com/MindSetLib/Insolver/tree/fixed_docs/insolver/feature_engineering) module, but it covers the **automated data preprossesing** class and all of its features. For this, the [`AB_NYC_2019`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset is used.

- [Gradient Boosting Example - Lending Club](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Gradient%20Boosting%20Example%20-%20Lending%20Club.ipynb)
This tutorial demonstrates how to create **classification models** for the [`Lending Club`](https://www.kaggle.com/wordsforthewise/lending-club) dataset using the **Gradient Boosting libraries** and the `InsolverGBMWrapper` class.

- [Inference Example](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Inference%20Example.ipynb)
This tutorial demonstrates how to load `InsolverTransform` transforms from a file and initialize them using the `init_transforms` function.

- [InsolverDataFrame and InsolverTransform Example](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/InsolverDataFrame%20and%20InsolverTransform%20Example.ipynb)
This tutorial demonstrates main features of the `InsolverDataFrame` class and the `InsolverTransform` class.

- [Regression Example - FreeMLP](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the `freMPL-R` dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Regression Example - US Accidents](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the [`US Traffic Accident`](https://smoosavi.org/datasets/us_accidents) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Report Example](https://github.com/MindSetLib/Insolver/blob/fixed_docs/tutorials/Report%20Example.ipynb)
This tutorial demonstrates how to create a **HTML report** with different models using the `Report` class.

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
![](https://github.com/MindSetLib/Insolver/releases/download/v0.4.6/InsolverDemo.gif)

### Example of a model production service
![](https://github.com/MindSetLib/Insolver/releases/download/v0.4.6/InsolverImplementation.gif)

### Example of an elyra pipeline built with the Insolver inside
![](https://github.com/MindSetLib/Insolver/releases/download/v0.4.6/InsolverElyraPipeline.gif)

### Contacts
frank@mind-set.ru
+79263790123
