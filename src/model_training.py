import joblib
import numpy as np
from pathlib import Path
import sys

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

# Use Path for cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "autos.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_model():
    print("=" * 60)
    print("🚗 TRAINING USED CAR PRICE MODEL")
    print("=" * 60)
    
    # Check if data file exists
    if not DATA_PATH.exists():
        print(f"\n❌ ERROR: Data file not found at {DATA_PATH}")
        print("\nPlease ensure autos.csv is in data/raw/ directory")
        return False
    
    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = load_data(str(DATA_PATH))
    print(f"   Loaded {len(df):,} records")

    print("\n🧹 Cleaning data...")
    df = basic_clean(df)
    df = filter_invalid_prices(df)
    df = filter_invalid_mileage(df)
    print(f"   After cleaning: {len(df):,} records")

    # Standardize column names
    rename_map = {
        "yearOfRegistration": "registration_year",
        "powerPS": "power_ps",
        "kilometer": "odometer",
    }
    df = df.rename(columns=rename_map)

    print("\n📊 Sampling data...")
    sample_size = min(100000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"   Sample size: {sample_size:,}")

    # Feature engineering
    print("\n🔧 Engineering features...")
    CURRENT_YEAR = 2025
    df["car_age"] = (CURRENT_YEAR - df["registration_year"]).clip(lower=0)
    df["km_per_year"] = df["odometer"] / (df["car_age"] + 1)

    # Drop high-cardinality column
    if "model" in df.columns:
        df = df.drop(columns=["model"])

    print("   Preparing features and target...")
    X = df.drop("price", axis=1)

    # Log transform target
    y = np.log1p(df["price"])

    # Target encoding for brand
    brand_target_mean = (
        X[["brand"]]
        .join(y)
        .groupby("brand")[y.name]
        .mean()
    )

    global_price_mean = y.mean()

    # Replace brand with target-encoded values
    X["brand"] = X["brand"].map(brand_target_mean).fillna(global_price_mean)

    # Frequency encoding
    print("   Applying frequency encoding...")
    freq_maps = {}
    X_encoded = X.copy()

    for col in X.columns:
        if X[col].dtype == "object" and col != "brand":
            freq = X[col].value_counts(normalize=True)
            freq_maps[col] = freq
            X_encoded[col] = X[col].map(freq).fillna(0)

    # Scale numeric features
    print("   Scaling numeric features...")
    numeric_cols = ["power_ps", "odometer", "car_age", "km_per_year"]
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    # Fill remaining NaNs
    X_encoded = X_encoded.fillna(0)
    X = X_encoded

    # Save feature schema
    feature_columns = list(X.columns)

    print("\n📈 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🤖 Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_depth=10, 
        learning_rate=0.08, 
        max_iter=300, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("\n📊 Evaluating model...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_mae = mean_absolute_error(np.expm1(y_train), np.expm1(train_preds))
    test_mae = mean_absolute_error(np.expm1(y_test), np.expm1(test_preds))

    print(f"   Training MAE: €{train_mae:,.2f}")
    print(f"   Testing MAE:  €{test_mae:,.2f}")

    # Calculate confidence interval
    errors = np.abs(np.expm1(y_test) - np.expm1(test_preds))
    error_std = errors.std()
    print(f"   Model confidence: ±€{int(error_std):,}")

    # Save artifacts
    print("\n💾 Saving model artifacts...")
    joblib.dump(model, MODEL_DIR / "car_price_model.pkl")
    joblib.dump(freq_maps, MODEL_DIR / "frequency_maps.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(brand_target_mean, MODEL_DIR / "brand_target_mean.pkl")
    joblib.dump(global_price_mean, MODEL_DIR / "global_price_mean.pkl")
    joblib.dump(error_std, MODEL_DIR / "error_std.pkl")

    print(f"   ✅ All files saved to {MODEL_DIR}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTo run the Streamlit app, execute:")
    print("   streamlit run app/streamlit_app.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)