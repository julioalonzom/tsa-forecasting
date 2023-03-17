"""Functions plotting results."""

import plotly.express as px


def plot_predictions(data, predictions):
    train_size = int(len(data) * 0.7)
    test_data = data[train_size : len(data)]

    fig = px.line()
    fig.add_scatter(x=test_data.index, y=test_data.values, name="Actual")
    fig.add_scatter(x=predictions.index, y=predictions.values, name="Predicted")

    return fig
