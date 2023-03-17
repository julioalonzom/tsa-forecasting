"""Function(s) for cleaning the data set(s)."""



def clean_data(data):
    """Clean data set.

    Args:
        data (pandas.DataFrame): the data set.

    Returns:

    """
    data["UNRATE"] = data["UNRATE"].fillna(method="ffill")
    return data
