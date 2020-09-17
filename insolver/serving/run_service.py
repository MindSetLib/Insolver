import argparse
import os
from subprocess import Popen, PIPE

import uvicorn


def exec_cmd():
    cmd = 'gunicorn flask-app:app --pythonpath /home/andrey/PycharmProjects/MS-InsuranceScoring/insolver/serving/'
    out, err = Popen(f'{cmd}', shell=True, stdout=PIPE).communicate()
    print(str(out, 'utf-8'))


def run():
    parser = argparse.ArgumentParser(description='ML API service')
    parser.add_argument('-model', action='store')
    parser.add_argument('-transforms', action='store')
    parser.add_argument('-service', action='store')

    args = parser.parse_args()

    os.environ['model_path'] = args.model
    os.environ['transforms_path'] = args.transforms

    if args.service == 'flask':
        exec_cmd()
    elif args.service == 'fastapi':
        uvicorn.run("fastapi-app:app", host="127.0.0.1", port=8000, log_level="info")
    else:
        print('wrong service, try "-service flask" or "-service fastapi"')
