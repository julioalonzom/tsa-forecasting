"""Functions for fitting the regression model."""

import statsmodels.tsa.holtwinters as sm
from statsmodels.iolib.smpickle import load_pickle


def fit_model(data):  # later include another model type
    """
    Args:
        data: Either unemployment rates or GDP. add later
        model_type: Either Holt-Winter expontential smoothing ('hw') or Centering Moving Average ('cma').
    """
    train_size = int(len(data) * 0.7)
    train_data = data[0:train_size]

    model = sm.ExponentialSmoothing(
        train_data, trend="add", seasonal="add", seasonal_periods=12,
    )
    fit = model.fit()

    return fit


def load_model(path):
    """Load statsmodels model.

    Args:
        path (str or pathlib.Path): Path to model file.

    Returns:
        statsmodels.base.model.Results: The stored model.

    """
    return load_pickle(path)
