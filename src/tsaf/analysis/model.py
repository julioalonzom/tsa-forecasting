"""Functions for fitting ..."""

from statsmodels.iolib.smpickle import load_pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_model(data, model_type):  # later include another model type
    """
    Args:
        data: Either unemployment rates or GDP. add later
        model_type: Either Holt-Winter expontential smoothing ('hw') or ARIMA ('arima').
    """
    train_size = int(len(data) * 0.7)
    train_data = data[0:train_size]

    if model_type == "hw":
        fit = ExponentialSmoothing(
            train_data,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
        ).fit()

    elif model_type == "arima":
        fit = ARIMA(train_data, order=(1, 1, 1)).fit()

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
