"""Tasks running the core analyses."""

import pandas as pd
import pytask

from tsaf.analysis.forecast import forecast, metrics
from tsaf.analysis.model import fit_model, load_model
from tsaf.config import BLD, SRC

for model_type in ["hw", "arima"]:

    kwargs = {
        "model_type": model_type,
        "produces": BLD / "python" / "models" / f"{model_type}_model.pickle",
    }

    @pytask.mark.depends_on(
        {
            "scripts": ["model.py", "forecast.py"],
            "data": SRC / "data" / "data.csv",
        },
    )
    @pytask.mark.task(id=model_type, kwargs=kwargs)
    def task_fit_model(depends_on, produces, model_type):
        """Fits a specified model on the cleaned data and saves the model as a pickle
        file.
        """
        data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
        model = fit_model(data, model_type)
        model.save(produces)


@pytask.mark.depends_on(
    {
        "data": SRC / "data" / "data.csv",
        "hw_model": BLD / "python" / "models" / "hw_model.pickle",
        "arima_model": BLD / "python" / "models" / "arima_model.pickle",
    },
)
@pytask.mark.produces(
    {
        "hw": BLD / "python" / "forecasts" / "hw_forecasts.csv",
        "arima": BLD / "python" / "forecasts" / "arima_forecasts.csv",
    },
)
def task_forecast(depends_on, produces):
    """Forecasts values based on the model estimates for both Holt-Winters and ARIMA
    models.
    """
    data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
    for model_type, model_file in [
        ("hw", depends_on["hw_model"]),
        ("arima", depends_on["arima_model"]),
    ]:
        model = load_model(model_file)
        forecasts = forecast(data, model)
        forecasts.to_csv(produces[model_type], header=True)


@pytask.mark.depends_on(
    {
        "data": SRC / "data" / "data.csv",
        "hw_forecasts": BLD / "python" / "forecasts" / "hw_forecasts.csv",
        "arima_forecasts": BLD / "python" / "forecasts" / "arima_forecasts.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "forecasts" / "metrics.csv")
def task_metrics(depends_on, produces):
    """Calculates and saves performance metrics (RMSE, MAE, MASE) for the forecasts
    generated by both Holt-Winters and ARIMA models.
    """
    data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
    hw_forecasts = pd.read_csv(
        depends_on["hw_forecasts"],
        index_col=0,
        parse_dates=True,
    )
    arima_forecasts = pd.read_csv(
        depends_on["arima_forecasts"],
        index_col=0,
        parse_dates=True,
    )

    df = metrics(data, hw_forecasts, arima_forecasts)
    df.to_csv(produces, index=False)
