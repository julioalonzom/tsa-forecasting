"""Tasks for managing the data."""

import pandas as pd
import pytask

from tsaf.config import BLD, SRC
from tsaf.data_management import clean_data


@pytask.mark.depends_on(
    {
        "scripts": ["clean_data.py"],
        "data": SRC / "data" / "data.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
def task_clean_data(depends_on, produces):
    """Clean the data."""
    data = pd.read_csv(depends_on["data"], index_col=0, parse_dates=True)
    data = clean_data(data)
    data.to_csv(produces)
