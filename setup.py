from setuptools import setup


setup(name='insolver',
      version='0.1',
      description='Mindset insurance scoring ',
      url='https://github.com/MindSetLib/MS-InsuranceScoring',
      author='Mindset',
      author_email='request@mind-set.ru',
      license='MIT',
      packages=['insolver'],
      entry_points={
            "console_scripts": [
                  "insolver_serving = insolver.serving.run_service:run"
            ]
      },
      install_requires=[
            'wheel',
            'numpy',
            'pandas',
            'xgboost',
            'lightgbm',
            'catboost',
            'hyperopt',
            'sklearn',
            'pyodbc',
            'requests',
            'requests_cache',
            'plotly',
            'seaborn',
            'shap',
            'geocoder',
            'h2o',
            'flask',
            'fastapi',
            'uvicorn',
            'pydantic',
            'gunicorn',
            'uvicorn'
      ],
      zip_safe=False)
