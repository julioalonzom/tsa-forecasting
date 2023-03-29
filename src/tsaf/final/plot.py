"""Functions plotting results."""

import plotly.express as px


def plot_forecasts(data, forecasts):
    """Plots the forecasted values alongside the actual training and test data.

    Args:
        data: A pandas DataFrame containing the time series data.
        forecasts: A pandas DataFrame containing the forecasted values.

    Returns:
        A plotly figure object showing the training and test data, and the forecasted values.

    """
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    test_data = data[train_size : len(data)]

    fig = px.line()
    fig.add_scatter(x=train_data.index, y=train_data["UNRATE"], name="Training data")
    fig.add_scatter(x=test_data.index, y=test_data["UNRATE"], name="Test data")
    fig.add_scatter(x=forecasts.index, y=forecasts["PRED"], name="Forecasts")

    return fig


def plot_metrics(measures):
    """Plots the performance metrics of different models.

    Args:
        measures: A pandas DataFrame containing the performance metrics of different models.

    Returns:
        A plotly figure object showing the performance metrics of different models.

    """
    df_melt = measures.melt(id_vars=["Model"], var_name="Metric", value_name="Value")
    fig = px.bar(df_melt, x="Metric", y="Value", color="Model", barmode="group")

    return fig
