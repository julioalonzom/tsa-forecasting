"""Functions for forecasting values based on the estimated model."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast(data, model):
    train_size = int(len(data) * 0.7)
    test_data = data[train_size : len(data)]

    forecasts = model.forecast(len(test_data))
    forecasts = forecasts.to_frame(name="PRED")
    forecasts.index.name = "DATE"
    return forecasts


def metrics(data, hw_forecasts, arima_forecasts):
    train_size = int(len(data) * 0.7)
    test_data = data[train_size : len(data)]

    mae_holt = mean_absolute_error(test_data["UNRATE"], hw_forecasts)
    mse_holt = mean_squared_error(test_data["UNRATE"], hw_forecasts)
    rmse_holt = mean_squared_error(test_data["UNRATE"], hw_forecasts, squared=False)
    mape_holt = (
        np.mean(
            np.abs((test_data["UNRATE"] - hw_forecasts) / (test_data["UNRATE"] + 1e-6)),
        )
        * 100
    )

    mae_arima = mean_absolute_error(test_data["UNRATE"], arima_forecasts)
    mse_arima = mean_squared_error(test_data["UNRATE"], arima_forecasts)
    rmse_arima = mean_squared_error(test_data["UNRATE"], arima_forecasts, squared=False)
    mape_arima = (
        np.mean(
            np.abs(
                (test_data["UNRATE"] - arima_forecasts) / (test_data["UNRATE"] + 1e-6),
            ),
        )
        * 100
    )

    df = {
        "Model": ["Holt-Winters", "ARIMA"],
        "MAE": [mae_holt, mae_arima],
        "MSE": [mse_holt, mse_arima],
        "RMSE": [rmse_holt, rmse_arima],
        "MAPE": [mape_holt, mape_arima],
    }

    df = pd.DataFrame(df)

    return df
