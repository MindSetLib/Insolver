from setuptools import setup


setup(name='insolver',
      version='0.3.dev3',
      description='Mindset insurance scoring ',
      url='https://github.com/MindSetLib/MS-InsuranceScoring',
      author='Mindset',
      author_email='request@mind-set.ru',
      license='MIT',
      packages=['insolver', 'insolver.serving'],
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
            'scikit-learn',
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
      ],
      zip_safe=False)
