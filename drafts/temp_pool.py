from multiprocessing import Pool
from flask import Flask
app = Flask(__name__)


def f(x):
    return x * x


if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))



def f(x):
    return x * x


def run_map():
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))


@app.route('/')
def index():
    run_map()




    return 'Index'


if __name__ == '__main__':
    app.run(debug=True)