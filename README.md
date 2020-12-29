# Insolver

Mindset insurance scoring - product repository

## Installation:

```shell
pip install insolver
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


## Quickstart:

```python
# short example
```

## Examples:

- [private motor French insurer dataset](https://github.com/MindSetLib/MS-InsuranceScoring/blob/master/examples/Insolver_FreMPL.ipynb)
- [US  traffic accident dataset](https://github.com/MindSetLib/MS-InsuranceScoring/blob/master/examples/Insolver_US_Accidents.ipynb)

## Documentation:

will be here soon

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
