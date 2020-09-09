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
