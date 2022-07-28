from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-dev.txt") as f:
    req_dev = f.read().splitlines()

with open("./docs/requirements-docs.txt") as f:
    req_dev += f.read().splitlines()

with open("requirements-extra.txt") as f:
    extras = f.read().splitlines()


if __name__ == "__main__":
    setup(
        packages=find_packages(),
        entry_points={'console_scripts': ['insolver_serving = insolver.serving.run_service:run']},
        install_requires=required,
        extras_require={'full': extras, 'dev': req_dev},
        zip_safe=False,
        python_requires='>=3.7',
        include_package_data=True,
    )
