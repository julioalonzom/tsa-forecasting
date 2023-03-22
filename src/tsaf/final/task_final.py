"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from tsaf.config import BLD
from tsaf.final.plot import plot_forecasts, plot_metrics

for model_type in ["hw", "arima"]:

    kwargs = {
        "produces": BLD / "python" / "figures" / f"{model_type}_forecasts.png",
        "depends_on": {
            "data": BLD / "python" / "data" / "data_clean.csv",
            "forecasts": BLD / "python" / "forecasts" / f"{model_type}_forecasts.csv",
        },
    }

    @pytask.mark.task(id=model_type, kwargs=kwargs)
    def task_plot_forecasts(depends_on, produces):
        data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
        forecasts = pd.read_csv(
            depends_on["forecasts"],
            index_col=0,
            parse_dates=True,
        )

        fig = plot_forecasts(data, forecasts)

        fig.write_image(produces)


@pytask.mark.depends_on(BLD / "python" / "forecasts" / "metrics.csv")
@pytask.mark.produces(BLD / "python" / "figures" / "metrics.png")
def task_plot_metrics(depends_on, produces):
    measures = pd.read_csv(depends_on, index_col=0, parse_dates=True)

    fig = plot_metrics(measures)

    fig.write_image(produces)


""" @pytask.mark.depends_on( BLD / "python" / "") """

""" for group in GROUPS:

    kwargs = {
        "group": group,
        "depends_on": {"predictions": BLD / "python" / "predictions" / f"{group}.csv"},
        "produces": BLD / "python" / "figures" / f"smoking_by_{group}.png",
    }

    @pytask.mark.depends_on(
        {
            "data_info": SRC / "data_management" / "data_info.yaml",
            "data": BLD / "python" / "data" / "data_clean.csv",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_plot_results_by_age_python(depends_on, group, produces):
        """ """
        data_info = read_yaml(depends_on["data_info"])
        data = pd.read_csv(depends_on["data"])
        predictions = pd.read_csv(depends_on["predictions"])
        fig = plot_regression_by_age(data, data_info, predictions, group)
        fig.write_image(produces)


@pytask.mark.depends_on(BLD / "python" / "models" / "model.pickle")
@pytask.mark.produces(BLD / "python" / "tables" / "estimation_results.tex")
def task_create_results_table_python(depends_on, produces):
    """ """
    model = load_model(depends_on)
    table = model.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table) """
