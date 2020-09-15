import argparse
import os
from pprint import pprint
from subprocess import Popen, PIPE


def exec_cmd():
    cmd = 'gunicorn flask-app:app --pythonpath /home/andrey/PycharmProjects/MS-InsuranceScoring/insolver/serving/'
    out, err = Popen(f'{cmd}', shell=True, stdout=PIPE).communicate()
    print(str(out, 'utf-8'))


parser = argparse.ArgumentParser(description='ML API service')
parser.add_argument('-model', action="store")
parser.add_argument('-transforms', action="store")
args = parser.parse_args()

os.environ['model_path'] = args.model
os.environ['transforms_path'] = args.transforms
pprint(os.environ)

exec_cmd()
