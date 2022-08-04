import subprocess

from typing import Union

from ...utils import warn_insolver


class InsolverRequirementsWarning(Warning):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


def get_requirements() -> str:
    insolver_line = subprocess.run("pip show insolver", shell=True, capture_output=True, encoding='utf8').stdout
    env = subprocess.run("pip freeze", shell=True, capture_output=True, encoding='utf8').stdout

    insolver_dict = {key: val for key, val in (x.split(': ') for x in insolver_line[:-1].split('\n'))}
    insolver_version = f"insolver=={insolver_dict['Version']}\n"
    requires = [f'{req}==' for req in insolver_dict['Requires'].split(', ')]
    env_req = '\n'.join([req for req in env.split('\n')[:-1] if any((x in req for x in requires))])

    return f"{insolver_version}{env_req}"


def check_requirements(requirements: Union[bytes, str]) -> None:
    if isinstance(requirements, bytes):
        requirements = requirements.decode('utf-8')
    required = {key: val for key, val in [req.split('==') for req in requirements.split('\n')]}

    env = subprocess.run("pip freeze", shell=True, capture_output=True, encoding='utf8').stdout
    env_req = {key: val for key, val in [req.split('==') for req in env.split('\n')[:-1] if len(req.split('==')) == 2]}

    potential_missing = set(required.keys()).difference(set(env_req.keys()))
    missing_packages = list()

    for pack in potential_missing:
        check = subprocess.run(f"pip show {pack}", shell=True, capture_output=True, encoding='utf8').stdout
        if check == '':
            missing_packages.append(f'{pack}: missing package')
        else:
            check_dict = {key: val for key, val in (x.split(': ') for x in check[:-1].split('\n'))}
            env_req.update({pack: check_dict['Version']})

    env_req = {
        key: val for key, val in env_req.items() if key in set(required.keys()).intersection(set(env_req.keys()))
    }

    non_equal_packages = [
        f'{key}: required {required[key]} got {env_req[key]}'
        for key in env_req
        if key in required and env_req[key] != required[key]
    ]

    problem_packages = missing_packages + non_equal_packages

    if problem_packages != '':
        for warning in problem_packages:
            warn_insolver(warning, InsolverRequirementsWarning)
