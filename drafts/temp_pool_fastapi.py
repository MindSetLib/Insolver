from multiprocessing import Pool

from fastapi import FastAPI
from flask import Flask

import uvicorn

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


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=6000, log_level="info")