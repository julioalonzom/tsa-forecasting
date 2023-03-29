"""Functions for fitting ..."""

from statsmodels.iolib.smpickle import load_pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_model(data, model_type):
    """Fits a time series model to the provided data using the specified model type.

    Args:
        data (array-like): The time series data to fit the model to.
        model_type (str): The type of model to fit. Currently supported model types are 'hw' (Holt-Winters exponential smoothing) and 'arima' (AutoRegressive Integrated Moving Average).

    Returns:
        A fitted time series model object of the specified type.

    """
    train_size = int(len(data) * 0.7)
    train_data = data[0:train_size]

    if model_type == "hw":
        fit = ExponentialSmoothing(
            train_data,
            trend="add",
            seasonal="add",
            seasonal_periods=52,
            damped_trend=True,
        ).fit()

    elif model_type == "arima":
        fit = ARIMA(train_data, order=(2, 0, 0)).fit()

    else:
        message = "Only 'hw' and 'arima' model_type is supported right now."
        raise ValueError(message)

    return fit


def load_model(path):
    """Load statsmodels model.

    Args:
        path (str or pathlib.Path): Path to model file.

    Returns:
        statsmodels.base.model.Results: The stored model.

    """
    return load_pickle(path)
