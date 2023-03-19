"""Tasks running the core analyses."""

import pandas as pd
import pytask

from tsaf.analysis.model import fit_model, load_model
from tsaf.analysis.predict import predict
from tsaf.config import BLD

for model_type in ["hw", "arima"]:

    kwargs = {
        "model_type": model_type,
        "produces": BLD / "python" / "models" / f"{model_type}_model.pickle",
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["model.py", "predict.py"],
            "data": BLD / "python" / "data" / "data_clean.csv",
        },
    )
    @pytask.mark.task(id=model_type, kwargs=kwargs)
    def task_fit_model(depends_on, produces, model_type):
        """."""
        data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
        model = fit_model(data, model_type)
        model.save(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "data_clean.csv",
        "hw_model": BLD / "python" / "models" / "hw_model.pickle",
        "arima_model": BLD / "python" / "models" / "arima_model.pickle",
    },
)
@pytask.mark.produces(
    {
        "hw": BLD / "python" / "predictions" / "hw_predictions.csv",
        "arima": BLD / "python" / "predictions" / "arima_predictions.csv",
    },
)
def task_predict(depends_on, produces):
    """Predict based on the model estimates."""
    data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
    for model_type, model_file in [
        ("hw", depends_on["hw_model"]),
        ("arima", depends_on["arima_model"]),
    ]:
        model = load_model(model_file)
        predicted = predict(data, model)
        predicted.to_csv(produces[model_type], header=True)
