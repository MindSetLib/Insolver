from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(name='insolver',
      version='0.4.9',
      description='Insolver is low-code machine learning library, initially created for the insurance industry, '
                  'but can be used in any other.\n You can find a detailed overview at '
                  'https://insolver.readthedocs.io/en/latest/source/overview.html.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/MindSetLib/Insolver',
      keywords=['insurance', 'machine learning'],
      author='Mindset',
      author_email='request@mind-set.ru',
      license='MIT',
      packages=find_packages(),
      entry_points={
            'console_scripts': [
                  'insolver_serving = insolver.serving.run_service:run'
            ]
      },
      install_requires=required,
      extras_require={
            'db-connects': ['pyodbc']
      },
      zip_safe=False,
      classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
      ],
      python_requires='>=3.7',
      )
