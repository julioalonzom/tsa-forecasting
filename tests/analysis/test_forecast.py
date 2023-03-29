import pytest
import numpy as np
import pandas as pd
from tsaf.analysis.model import fit_model
from tsaf.analysis.forecast import forecast

@pytest.fixture
def mock_data():
    np.random.seed(123)
    trend = np.arange(0, 500)
    seasonal = np.sin(np.linspace(0, 2 * np.pi, 500))
    noise = np.random.normal(0, 1, 500)
    data = pd.Series(10 + trend + 2 * seasonal + noise)
    return data

def test_forecast_returns_dataframe(mock_data):
    model = fit_model(mock_data, model_type='hw')
    forecasts = forecast(mock_data, model)
    assert isinstance(forecasts, pd.DataFrame)

def test_forecast_returns_correct_length(mock_data):
    model = fit_model(mock_data, model_type='hw')
    forecasts = forecast(mock_data, model)
    assert len(forecasts) == len(mock_data) - int(len(mock_data) * 0.7)

def test_forecast_returns_correct_index(mock_data):
    model = fit_model(mock_data, model_type='hw')
    forecasts = forecast(mock_data, model)
    assert isinstance(forecasts.index, pd.DatetimeIndex)