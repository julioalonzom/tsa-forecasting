"""Functions plotting results."""

import plotly.express as px


def plot_predictions(data, predictions):
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    test_data = data[train_size : len(data)]

    fig = px.line()
    fig.add_scatter(x=train_data.index, y=train_data["UNRATE"], name="Training data")
    fig.add_scatter(x=test_data.index, y=test_data["UNRATE"], name="Test data")
    fig.add_scatter(x=predictions.index, y=predictions["PRED"], name="Predictions")

    return fig
