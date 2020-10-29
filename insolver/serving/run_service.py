import argparse
import os
from subprocess import Popen, PIPE


def exec_cmd(cmd):
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
        cmd = 'gunicorn -b 0.0.0.0:8000 insolver.serving.flask_app:app'
        exec_cmd(cmd)
    elif args.service == 'fastapi':
        cmd = 'gunicorn -b 0.0.0.0:8000 insolver.serving.fastapi_app:app -k uvicorn.workers.UvicornWorker'
        exec_cmd(cmd)
    else:
        print('wrong service, try "-service flask" or "-service fastapi"')
