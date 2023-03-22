"""Functions plotting results."""

import plotly.express as px


def plot_forecasts(data, forecasts):
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    test_data = data[train_size : len(data)]

    fig = px.line()
    fig.add_scatter(x=train_data.index, y=train_data["UNRATE"], name="Training data")
    fig.add_scatter(x=test_data.index, y=test_data["UNRATE"], name="Test data")
    fig.add_scatter(x=forecasts.index, y=forecasts["PRED"], name="Forecasts")

    return fig


def plot_metrics(measures):
    df_melt = measures.melt(id_vars=["Model"], var_name="Metric", value_name="Value")
    fig = px.bar(df_melt, x="Metric", y="Value", color="Model", barmode="group")

    return fig
