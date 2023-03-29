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
        """Plot the forecasts for a given model type and save the resulting figure to a
        file.
        """
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
    """Plot the evaluation metrics for each model and save the resulting figure to a
    file.
    """
    measures = pd.read_csv(depends_on)

    fig = plot_metrics(measures)

    fig.write_image(produces)


@pytask.mark.depends_on(BLD / "python" / "forecasts" / "metrics.csv")
@pytask.mark.produces(BLD / "python" / "tables" / "metrics.tex")
def task_csv_to_latex_table(depends_on, produces):
    """Convert the evaluation metrics in CSV format to a LaTeX table and save it to a
    file.
    """
    df = pd.read_csv(depends_on)

    table = df.to_latex(index=False, caption="Values of the different measures.")

    with open(produces, "w") as f:
        f.write(table)
