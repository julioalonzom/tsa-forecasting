"""Functions for predicting outcomes based on the estimated model."""


def predict(data, model):
    train_size = int(len(data) * 0.7)
    test_data = data[train_size : len(data)]

    predictions = model.predict(start=test_data.index[0], end=test_data.index[-1])
    predictions = predictions.to_frame(name="PRED")
    predictions.index.name = "DATE"
    return predictions


# add some forecast functions here

""" def forecast(model, steps)
    forecast = model.forecast(steps)
    return forecast """
