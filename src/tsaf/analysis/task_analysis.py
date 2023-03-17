"""Tasks running the core analyses."""

import pandas as pd
import pytask

from tsaf.analysis.model import fit_model, load_model
from tsaf.analysis.predict import predict
from tsaf.config import BLD


@pytask.mark.depends_on(
    {
        "scripts": ["model.py", "predict.py"],
        "data": BLD / "python" / "data" / "data_clean.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
def task_fit_model(depends_on, produces):
    """."""
    data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
    model = fit_model(data)
    model.save(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "data_clean.csv",
        "model": BLD / "python" / "models" / "model.pickle",
    },
)
@pytask.mark.produces(BLD / "python" / "predictions" / "predictions.csv")
def task_predict(depends_on, produces):
    """Predict based on the model estimates."""
    model = load_model(depends_on["model"])
    data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
    predicted = predict(data, model)
    predicted.to_csv(produces)
