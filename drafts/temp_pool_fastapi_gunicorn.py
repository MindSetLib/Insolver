from multiprocessing import Pool

from fastapi import FastAPI


app = FastAPI()


def f(x):
    return x * x


def run_map():
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))


@app.post("/")
async def predict():
    run_map()
    return 'Index'