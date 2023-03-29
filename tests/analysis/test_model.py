"""Tests for the model."""

import numpy as np
import pandas as pd
import pytest
from tsaf.analysis.model import fit_model


@pytest.fixture()
def mock_data():
    """Mock time series data for testing.

    Returns:
        array-like: A mock time series data array of length 500.

    """
    np.random.seed(123)
    trend = np.arange(0, 500)
    seasonal = np.sin(np.linspace(0, 2 * np.pi, 500))
    noise = np.random.normal(0, 1, 500)
    data = pd.Series(10 + trend + 2 * seasonal + noise)
    return data


def test_fit_model_returns_model_object(mock_data):
    """Test that the fit_model function returns a model object."""
    model = fit_model(mock_data, "hw")
    assert model is not None


def test_fit_model_returns_arima_model_object(mock_data):
    """Test that the fit_model function returns an ARIMA model object."""
    model = fit_model(mock_data, "arima")
    assert model is not None


def test_fit_model_raises_value_error_with_invalid_model_type(mock_data):
    """Test that the fit_model function raises a ValueError with an invalid model
    type."""
    with pytest.raises(ValueError):
        fit_model(mock_data, "invalid_model_type")
