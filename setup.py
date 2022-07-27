from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-extra.txt") as f:
    extras = f.read().splitlines()


if __name__ == "__main__":
    setup(packages=find_packages(),
          entry_points={
              'console_scripts': [
                  'insolver_serving = insolver.serving.run_service:run'
              ]
          },
          install_requires=required,
          extras_require={'full': extras},
          zip_safe=False,
          python_requires='>=3.7',
          include_package_data=True
          )
