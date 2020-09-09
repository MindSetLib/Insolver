from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def index():
    return "API for predict service"

# example
# @app.post("/predict", response_model=StockOut, status_code=200)
# def get_prediction(payload: StockIn):
#     ticker = payload.ticker
#
#     prediction_list = predict(ticker)
#
#     if not prediction_list:
#         raise HTTPException(status_code=400, detail="Model not found.")
#
#     response_object = {"ticker": ticker, "forecast": convert(prediction_list)}
#     return response_object


# запуск сервера
# uvicorn fastapi-app:app
