def frequency_encode(df, col):
    freq = df[col].value_counts(normalize=True)
    return df[col].map(freq)

def preprocess_features(df):
    df = df.copy()

    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col] = frequency_encode(df, col)

    # Fill remaining NaNs
    df = df.fillna(df.median(numeric_only=True))

    return df