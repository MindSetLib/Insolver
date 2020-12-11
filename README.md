# Insolver

Mindset insurance scoring - product repository

## Installing from git:
```shell
pip install "git+ssh://git@github.com/MindSetLib/MS-InsuranceScoring.git"
```

## Examples:

- [private motor French insurer dataset](https://github.com/MindSetLib/MS-InsuranceScoring/blob/master/examples/Insolver_FreMPL.ipynb)
- [US  traffic accident dataset](https://github.com/MindSetLib/MS-InsuranceScoring/blob/master/examples/Insolver_US_Accidents.ipynb)

## Documentation:

will be here soon


### Run tests:
```shell
python -m pytest
```

tests with coverage:
```shell
python -m pytest --cov=insolver; coverage html; xdg-open htmlcov/index.html
```

### Caveats:

To fix displaying plotly figs in jyputerlab install:
```shell
jupyter labextension install jupyterlab-plotly
```

In case of problem with `pyodbc` you may need to install:
```shell
sudo apt install unixodbc-dev
```
