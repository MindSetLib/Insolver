[![PyPI](https://img.shields.io/pypi/v/insolver?style=flat)](https://pypi.org/project/insolver/)
[![Documentation Status](https://readthedocs.org/projects/insolver/badge/?version=latest)](https://insolver.readthedocs.io/en/latest/?badge=latest)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/MindSetLib/Insolver/Python%20application?logo=github&label=tests)](https://github.com/MindSetLib/Insolver/actions)

# Insolver

Insolver is low-code machine learning library, initially created for the insurance industry, but can be used in any other. A more detailed overview you can find [here](https://insolver.readthedocs.io/en/latest/source/overview.html).

## Installation:

```shell
pip install insolver
```
### Install with addons:

```shell
pip install insolver[db-connects]
```

### Post-install:

To fix displaying plotly figs in jyputerlab install:
```shell
jupyter labextension install jupyterlab-plotly
```

In case of problem with `pyodbc` you may need to install:
```shell
sudo apt install unixodbc-dev
```


## Examples:

- [Private motor French insurer dataset](https://github.com/MindSetLib/Insolver/blob/master/examples/Insolver_FreMPL.ipynb)
- [US  traffic accident dataset](https://github.com/MindSetLib/Insolver/blob/master/examples/Insolver_US_Accidents.ipynb)

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
