# InsuranceScoring
Mindset insurance scoring - product repository

## Установка пакета `insolver`
#### Запускается из папки с гит репозиторием
- обычная установка
```shell script
pip install .
```
- установка в режиме редактирования (можно изменять код самого пакета без его переустановки)
```shell script
pip install -e .
```
#### Установка сразу из ветки репозиторий (**вариант для лекций**)
```shell script
pip install "git+ssh://git@github.com/MindSetLib/MS-InsuranceScoring.git@InsolverPackage#egg=insolver"
```

## Запуск тестов
```shell script
python -m pytest
```

### Проверка покрытия тестами
```shell script
 python -m pytest --cov=insolver; coverage html; xdg-open htmlcov/index.html
```

#### При проблемах с `pyodbc` на `ubuntu-20.04` установить:
```shell script
sudo apt install unixodbc-dev
```

### Автоматическое создание документации:
```shell script
cd docs/
sphinx-apidoc -f -o ../docs/source/ ../insolver
make html
```

**[Ссылка на документацию](docs/_build/html/index.html)**

## Запуск сервиса с моделью (пример)
```shell script
insolver_serving -model glm/Grid_GLM_Key_Frame__upload_a685662cd198b4799aee7e181b304e66.hex_model_python_1600165671228_1_model_1 -transforms transforms.pkl  -service flask
```

---
## Установка insolver в JupyterHub на наш сервер
```
(base) andrey@mindset1:~$ python -m venv new_env
(base) andrey@mindset1:~$ source ./new_env/bin/activate
(new_env) (base) andrey@mindset1:~$ pip install ipykernel
(new_env) (base) andrey@mindset1:~$ sudo apt install unixodbc-dev
(new_env) (base) andrey@mindset1:~$ python -m ipykernel install --user --name new_env --display-name "new_env"
(new_env) (base) andrey@mindset1:~$ pip install "git+ssh://git@github.com/MindSetLib/MS-InsuranceScoring.git@InsolverPackage#egg=insolver"
```


## Если не отображаются рисунки (plotly) в jupyterlab (например, в shap_explain), поставить расширение:
```
jupyter labextension install jupyterlab-plotly
```
