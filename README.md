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

- [Private motor French insurer dataset](https://github.com/MindSetLib/Insolver/blob/master/examples/Insolver_FreMPL.ipynb)
- [US  traffic accident dataset](https://github.com/MindSetLib/Insolver/blob/master/examples/Insolver_US_Accidents.ipynb)
- [Landing club dataset](https://github.com/MindSetLib/Insolver/blob/master/examples/Insolver_LendingClub.ipynb)

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
