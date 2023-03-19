"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from tsaf.config import BLD
from tsaf.final.plot import plot_predictions

for model_type in ["hw", "arima"]:

    kwargs = {
        "produces": BLD / "python" / "figures" / f"{model_type}_predictions.png",
        "depends_on": {
            "data": BLD / "python" / "data" / "data_clean.csv",
            "predictions": BLD
            / "python"
            / "predictions"
            / f"{model_type}_predictions.csv",
        },
    }

    @pytask.mark.task(id=model_type, kwargs=kwargs)
    def task_plot_predictions(depends_on, produces):
        data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
        predictions = pd.read_csv(
            depends_on["predictions"], index_col=0, parse_dates=True,
        )

        fig = plot_predictions(data, predictions)

        fig.write_image(produces)


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
