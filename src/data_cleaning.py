import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath, encoding="latin-1")

def basic_clean(df):
    df = df.drop_duplicates()

    drop_cols = [
        "seller", "offerType", "nrOfPictures", "lastSeen"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

def filter_invalid_prices(df):
    df = df[(df["price"] > 500) & (df["price"] < 350000)]
    return df

def filter_invalid_mileage(df):
    if "odometer" in df.columns:
        df = df[df["odometer"] > 0]
    return df