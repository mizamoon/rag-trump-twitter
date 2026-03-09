import pandas as pd


def load_tweets(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)

    elif path.endswith(".json"):
        df = pd.read_json(path)

    else:
        raise ValueError("Unsupported file format")

    return df