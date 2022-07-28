import argparse
import os
from subprocess import Popen, PIPE
from pathlib import Path


SERVING_DIR = Path(__file__).resolve().parent


def exec_cmd(cmd):
    out, err = Popen(f'{cmd}', shell=True, stdout=PIPE).communicate()
    print(str(out, 'utf-8'))


def run():
    parser = argparse.ArgumentParser(description='ML API service')
    parser.add_argument('-model', action='store', required=False)
    parser.add_argument('-transforms', action='store', required=False)
    parser.add_argument('-service', action='store', required=False)

    # add new parameter config file and models and transforms
    parser.add_argument('-configfile', action='store', required=False)
    parser.add_argument('-transforms_folder', action='store', required=False)
    parser.add_argument('-models_folder', action='store', required=False)

    # gunicorn config
    parser.add_argument('-ip', action='store', default='0.0.0.0', required=True)
    parser.add_argument('-port', action='store', default=8000, type=int, required=True)

    args = parser.parse_args()

    if args.model is not None:
        print(args.model)
        os.environ['model_path'] = args.model
    else:
        os.environ['model_path'] = ''

    if args.transforms is not None:
        print(args.transforms)
        os.environ['transforms_path'] = args.transforms
    else:
        os.environ['transforms_path'] = ''

    # add new config file and models
    if args.configfile is not None:
        print(args.configfile)
        os.environ['config_file'] = args.configfile
    else:
        os.environ['config_file'] = ''

    if args.transforms_folder is not None:
        print(args.transforms_folder)
        os.environ['transforms_folder'] = args.transforms_folder
    else:
        os.environ['transforms_folder'] = ''

    if args.models_folder is not None:
        print(args.models_folder)
        os.environ['models_folder'] = args.models_folder
    else:
        os.environ['models_folder'] = ''

    if args.service == 'flask':
        cmd = f'gunicorn -b {args.ip}:{args.port} insolver.serving.flask_app:app'
        exec_cmd(cmd)
    elif args.service == 'fastapi':
        cmd = f'gunicorn -b {args.ip}:{args.port} insolver.serving.fastapi_app:app -k uvicorn.workers.UvicornWorker'
        exec_cmd(cmd)
    if args.service == 'sflask':
        cmd = f'gunicorn --worker-class gthread -b {args.ip}:{args.port} insolver.serving.flask_app_several:app'
        exec_cmd(cmd)
    elif args.service == 'sfastapi':
        cmd = (
            f'gunicorn -b {args.ip}:{args.port}'
            'insolver.serving.fastapi_app_several:app -k uvicorn.workers.UvicornWorker'
        )
        exec_cmd(cmd)
    if args.service == 'django':
        django_dir = os.path.join(SERVING_DIR, 'django_insolver')
        cmd = f'gunicorn -b {args.ip}:{args.port} --chdir {django_dir} django_insolver.wsgi:application'
        exec_cmd(cmd)
    else:
        print('wrong service, try "-service flask" or "-service fastapi"')
