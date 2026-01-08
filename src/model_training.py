import joblib
import numpy as np


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_cleaning import (
    load_data,
    basic_clean,
    filter_invalid_prices,
    filter_invalid_mileage,
)

DATA_PATH = "data/raw/autos.csv"
MODEL_PATH = "models/car_price_model.pkl"


def train_model():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = basic_clean(df)
    df = filter_invalid_prices(df)
    df = filter_invalid_mileage(df)

    # -------------------------------------------------
    # 🔧 STANDARDIZE COLUMN NAMES (CRITICAL FIX)
    # -------------------------------------------------
    rename_map = {
        "yearOfRegistration": "registration_year",
        "powerPS": "power_ps",
        "kilometer": "odometer",
    }
    df = df.rename(columns=rename_map)

    print("Sampling data...")
    df = df.sample(n=100000, random_state=42)
    # -----------------------------
    CURRENT_YEAR = 2025

    df["car_age"] = CURRENT_YEAR - df["registration_year"]
    df["car_age"] = df["car_age"].clip(lower=0)

    df["km_per_year"] = df["odometer"] / (df["car_age"] + 1)

    # Drop high-cardinality column
    if "model" in df.columns:
        df = df.drop(columns=["model"])

    print("Preparing features and target...")
    X = df.drop("price", axis=1)

    # -----------------------------
    # LOG TRANSFORM TARGET
    # -----------------------------
    y = np.log1p(df["price"])

    # -----------------------------
    # Target encoding for BRAND
    # -----------------------------
    brand_target_mean = (
    X[["brand"]]
    .join(y)
    .groupby("brand")[y.name]
    .mean()
    )

    global_price_mean = y.mean()

    # Replace brand with target-encoded values
    X["brand"] = X["brand"].map(brand_target_mean).fillna(global_price_mean)

    # -----------------------------
    # FREQUENCY ENCODING
    # -----------------------------
    print("Applying frequency encoding...")
    freq_maps = {}
    X_encoded = X.copy()

    for col in X.columns:
        if X[col].dtype == "object" and col != "brand":
            freq = X[col].value_counts(normalize=True)
            freq_maps[col] = freq
            X_encoded[col] = X[col].map(freq)

    # -----------------------------
    # SCALE NUMERIC FEATURES
    # -----------------------------
    numeric_cols = ["power_ps", "odometer", "car_age", "km_per_year"]

    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    # Fill remaining NaNs
    X_encoded = X_encoded.fillna(0)

    X = X_encoded

    # -----------------------------
    # SAVE FEATURE SCHEMA
    # -----------------------------
    feature_columns = list(X.columns)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = HistGradientBoostingRegressor(
        max_depth=10, learning_rate=0.08, max_iter=300, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))

    # -----------------------------
    # Save uncertainty (confidence)
    # -----------------------------
    errors = np.abs(np.expm1(y_test) - np.expm1(preds))
    error_std = errors.std()

    joblib.dump(error_std, "models/error_std.pkl")

    print(f"MAE (EUR): {mae:.2f}")
    print("Saving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(freq_maps, "models/frequency_maps.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(brand_target_mean, "models/brand_target_mean.pkl")
    joblib.dump(global_price_mean, "models/global_price_mean.pkl")

    print("✅ Model, encoders, scaler, and schema saved successfully.")


if __name__ == "__main__":
    train_model()
