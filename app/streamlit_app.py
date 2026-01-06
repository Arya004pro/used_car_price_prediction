import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("models/car_price_model.pkl")
freq_maps = joblib.load("models/frequency_maps.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Predictor")
st.caption("Realistic pricing • Scalable ML • Indian-friendly UI")

st.divider()

# -----------------------------
# Indian currency formatter
# -----------------------------
def format_inr(amount):
    s = str(int(amount))
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ",".join(parts) + "," + last3

# -----------------------------
# UI label → dataset value maps
# -----------------------------
fuel_map = {
    "Petrol": "benzin",
    "Diesel": "diesel",
    "CNG / LPG": "lpg",
    "Electric": "elektro",
    "Hybrid": "hybrid"
}

gearbox_map = {
    "Manual": "manuell",
    "Automatic": "automatik"
}

vehicle_type_map = {
    "Sedan": "limousine",
    "SUV": "suv",
    "Hatchback": "kleinwagen",
    "Station Wagon": "kombi",
    "Coupe": "coupe",
    "Convertible": "cabrio",
    "Other": "andere"
}

# -----------------------------
# UI Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        sorted(freq_maps["brand"].index)
    )

    vehicle_type_label = st.selectbox(
        "Vehicle Type",
        list(vehicle_type_map.keys())
    )

    fuel_label = st.selectbox(
        "Fuel Type",
        list(fuel_map.keys())
    )

    gearbox_label = st.selectbox(
        "Gearbox",
        list(gearbox_map.keys())
    )

with col2:
    registration_year = st.slider(
        "Registration Year",
        min_value=1990,
        max_value=2025,
        value=2016
    )

    power_ps = st.slider(
        "Engine Power (PS)  — approx. Horsepower",
        min_value=50,
        max_value=500,
        value=120,
        step=10,
        help="PS (Pferdestärke) is a European unit similar to horsepower"
    )

    odometer = st.slider(
        "Mileage (km)",
        min_value=0,
        max_value=300000,
        value=80000,
        step=5000
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Price", use_container_width=True):

    # Build raw input
    input_df = pd.DataFrame([{
        "brand": brand,
        "vehicleType": vehicle_type_map[vehicle_type_label],
        "fuelType": fuel_map[fuel_label],
        "gearbox": gearbox_map[gearbox_label],
        "registration_year": registration_year,
        "power_ps": power_ps,
        "odometer": odometer
    }])

    # -----------------------------
    # Apply frequency encoding
    # -----------------------------
    for col, fmap in freq_maps.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(fmap).fillna(0)

    # -----------------------------
    # Scale numeric features
    # -----------------------------
    numeric_cols = ["registration_year", "power_ps", "odometer"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # -----------------------------
    # Add missing columns
    # -----------------------------
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[feature_columns]

    # -----------------------------
    # Predict & reverse log
    # -----------------------------
    log_price = model.predict(input_df)[0]
    price_eur = np.expm1(log_price)
    price_inr = int(price_eur * 90)

    st.success(
        f"💰 Estimated Price: € {int(price_eur):,}  (≈ ₹ {format_inr(price_inr)})"
    )
