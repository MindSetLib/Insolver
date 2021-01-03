from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='insolver',
      version='0.4.3.dev4',
      description='Mindset insurance scoring',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/MindSetLib/MS-InsuranceScoring',
      keywords='ML insurance scoring',
      author='Mindset',
      author_email='request@mind-set.ru',
      license='MIT',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'insolver_serving = insolver.serving.run_service:run'
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
            'pdpbox',
            'pyodbc',
            'kaleido',
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
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
      ],
      python_requires='>=3.6',
      )
