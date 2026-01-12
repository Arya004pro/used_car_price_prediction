import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging

# -------------------------------------------------
# Page Configuration (MUST BE FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Resolve base directory
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# -------------------------------------------------
# Load model artifacts with caching
# -------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts with caching."""
    try:
        return {
            "model": joblib.load(MODEL_DIR / "car_price_model.pkl"),
            "freq_maps": joblib.load(MODEL_DIR / "frequency_maps.pkl"),
            "feature_columns": joblib.load(MODEL_DIR / "feature_columns.pkl"),
            "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
            "error_std": joblib.load(MODEL_DIR / "error_std.pkl"),
            "brand_target_mean": joblib.load(MODEL_DIR / "brand_target_mean.pkl"),
            "global_price_mean": joblib.load(MODEL_DIR / "global_price_mean.pkl"),
        }
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found!\n\n"
            "Please train the model first:\n"
            "```\ncd src\npython model_training.py\n```"
        )
        st.stop()


# Load artifacts
artifacts = load_model_artifacts()
model = artifacts["model"]
freq_maps = artifacts["freq_maps"]
feature_columns = artifacts["feature_columns"]
scaler = artifacts["scaler"]
error_std = artifacts["error_std"]
brand_target_mean = artifacts["brand_target_mean"]
global_price_mean = artifacts["global_price_mean"]

# -------------------------------------------------
# Custom CSS for better styling
# -------------------------------------------------
st.markdown(
    """
<style>
    /* Main containers */
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    /* Insight boxes with proper text color */
    .insight-success {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
        font-weight: 500;
    }
    
    .insight-warning {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
        font-weight: 500;
    }
    
    .insight-danger {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
        font-weight: 500;
    }
    
    .insight-info {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
        font-weight: 500;
    }
    
    /* Price display */
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Currency formatters
# -------------------------------------------------
def format_inr(amount):
    """Format amount in Indian Rupee style."""
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


def format_currency(amount, currency="EUR"):
    """Format amount with currency symbol."""
    if currency == "INR":
        return f"₹{format_inr(amount)}"
    return f"€{int(amount):,}"


# -------------------------------------------------
# UI label → dataset value maps (with emojis)
# -------------------------------------------------
fuel_map = {
    "⛽ Petrol": "benzin",
    "🛢️ Diesel": "diesel",
    "💨 CNG / LPG": "lpg",
    "⚡ Electric": "elektro",
    "🔋 Hybrid": "hybrid",
}

gearbox_map = {
    "🔧 Manual": "manuell",
    "🅰️ Automatic": "automatik",
}

vehicle_type_map = {
    "🚗 Sedan": "limousine",
    "🚙 SUV": "suv",
    "🚐 Hatchback": "kleinwagen",
    "🚃 Station Wagon": "kombi",
    "🏎️ Coupe": "coupe",
    "🛻 Convertible": "cabrio",
    "📦 Other": "andere",
}

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    '<h1 style="text-align: center; color: #1f77b4;">🚗 Used Car Price Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown("*Powered by Machine Learning • Accuracy V3 • Brand-aware pricing*")
st.divider()

# -------------------------------------------------
# Layout: Sidebar + Main
# -------------------------------------------------
with st.sidebar:
    st.header("📝 Car Details")

    brand = st.selectbox(
        "🏷️ Brand",
        sorted(brand_target_mean.index),
    )

    vehicle_type_label = st.selectbox(
        "🚘 Vehicle Type",
        list(vehicle_type_map.keys()),
    )

    fuel_label = st.selectbox(
        "⛽ Fuel Type",
        list(fuel_map.keys()),
    )

    gearbox_label = st.selectbox(
        "⚙️ Gearbox",
        list(gearbox_map.keys()),
    )

    st.divider()

    registration_year = st.slider(
        "📅 Year",
        min_value=1990,
        max_value=2025,
        value=2016,
    )

    power_ps = st.slider(
        "🐎 Power (PS)",
        min_value=50,
        max_value=500,
        value=120,
        step=10,
    )

    odometer = st.slider(
        "📏 Mileage (km)",
        min_value=0,
        max_value=300000,
        value=80000,
        step=5000,
    )

    st.divider()

    currency_option = st.radio(
        "💱 Currency",
        ["EUR only", "INR only", "Both"],
        index=2,
        horizontal=True,
    )

# -------------------------------------------------
# Main Content
# -------------------------------------------------
st.subheader("🔮 Price Prediction")

# ACTION BUTTON AT TOP - Easy to access
predict_clicked = st.button(
    "🔮 PREDICT PRICE",
    use_container_width=True,
    type="primary",
    key="predict_button",
)

if predict_clicked:
    # Calculate features
    CURRENT_YEAR = 2025
    car_age = max(0, CURRENT_YEAR - registration_year)
    km_per_year = odometer / (car_age + 1)

    # Brand encoding
    brand_encoded = brand_target_mean.get(brand, global_price_mean)

    # Build input
    input_df = pd.DataFrame(
        [
            {
                "brand": brand_encoded,
                "vehicleType": vehicle_type_map[vehicle_type_label],
                "fuelType": fuel_map[fuel_label],
                "gearbox": gearbox_map[gearbox_label],
                "power_ps": power_ps,
                "odometer": odometer,
                "car_age": car_age,
                "km_per_year": km_per_year,
            }
        ]
    )

    # Frequency encoding
    for col, fmap in freq_maps.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(fmap).fillna(0)

    # Scale numeric
    numeric_cols = ["power_ps", "odometer", "car_age", "km_per_year"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Schema safety
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Predict
    log_price = model.predict(input_df)[0]
    price_eur = np.expm1(log_price)
    price_inr = int(price_eur * 90)

    # Confidence range
    MAX_DOWNSIDE_PCT = 0.30
    low_eur = max(price_eur * (1 - MAX_DOWNSIDE_PCT), price_eur - error_std)
    high_eur = price_eur + error_std
    low_inr = int(low_eur * 90)
    high_inr = int(high_eur * 90)

    # Display Results
    st.markdown("### 💰 Estimated Price")

    if currency_option == "EUR only":
        st.success(f"**{format_currency(price_eur, 'EUR')}**")
    elif currency_option == "INR only":
        st.success(f"**{format_currency(price_inr, 'INR')}**")
    else:
        st.success(
            f"**{format_currency(price_eur, 'EUR')}** (≈ {format_currency(price_inr, 'INR')})"
        )

    # Price Range - Use columns for compact display
    col_low, col_high = st.columns(2)
    with col_low:
        if currency_option == "INR only":
            st.info(f"**Min:** {format_currency(low_inr, 'INR')}")
        else:
            st.info(f"**Min:** {format_currency(low_eur, 'EUR')}")

    with col_high:
        if currency_option == "INR only":
            st.info(f"**Max:** {format_currency(high_inr, 'INR')}")
        else:
            st.info(f"**Max:** {format_currency(high_eur, 'EUR')}")

        # Metrics
        st.markdown("### 📈 Car Analysis")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("🕐 Age", f"{car_age}y")

        with m2:
            st.metric("📊 Avg km/y", f"{km_per_year:,.0f}")

        with m3:
            if km_per_year < 10000:
                usage = "🟢 Low"
            elif km_per_year < 20000:
                usage = "🟡 Normal"
            else:
                usage = "🔴 High"
            st.metric("Usage", usage)

        # Use tabs to organize insights and depreciation
        tab1, tab2 = st.tabs(["💡 Insights", "📉 Depreciation"])

        with tab1:
            st.markdown("**Market Insights & Recommendations**")

            # Age insights
            if car_age <= 3:
                st.markdown(
                    '<div class="insight-success">✅ <b>Recently registered</b> — Maintains good value and warranty coverage likely</div>',
                    unsafe_allow_html=True,
                )
            elif car_age <= 7:
                st.markdown(
                    '<div class="insight-warning">⚠️ <b>Mid-age vehicle</b> — In typical depreciation phase, consider maintenance costs</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="insight-danger">📉 <b>Older vehicle</b> — Significant depreciation expected, higher maintenance risk</div>',
                    unsafe_allow_html=True,
                )

            # Mileage insights
            if km_per_year > 20000:
                st.markdown(
                    '<div class="insight-danger">🚨 <b>High mileage usage</b> — More than 20k km/year, may require major maintenance soon</div>',
                    unsafe_allow_html=True,
                )
            elif km_per_year < 8000:
                st.markdown(
                    '<div class="insight-success">✅ <b>Low mileage</b> — Less than 8k km/year, excellent condition preservation</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="insight-info">ℹ️ <b>Average usage</b> — Normal wear and tear expected for the car age</div>',
                    unsafe_allow_html=True,
                )

            # Brand insights
            if brand in [
                "bmw",
                "audi",
                "mercedes_benz",
                "porsche",
                "jaguar",
                "tesla",
                "lexus",
            ]:
                st.markdown(
                    '<div class="insight-info">🏆 <b>Premium brand</b> — Better resale value retention compared to mainstream brands</div>',
                    unsafe_allow_html=True,
                )

            # Power insights
            if power_ps > 200:
                st.markdown(
                    '<div class="insight-info">⚡ <b>High performance</b> — Appeals to enthusiasts, may have specialized maintenance needs</div>',
                    unsafe_allow_html=True,
                )
            elif power_ps < 100:
                st.markdown(
                    '<div class="insight-success">🌱 <b>Fuel efficient</b> — Lower power consumption, economical for daily use</div>',
                    unsafe_allow_html=True,
                )

        with tab2:
            st.markdown("**Value Projection & Depreciation**")

            if car_age <= 3:
                dep_rate = (
                    15 if brand in ["bmw", "audi", "mercedes_benz", "porsche"] else 20
                )
                remaining_value_5yr = price_eur * ((1 - dep_rate / 100) ** 5)
                st.info(
                    f"**Annual depreciation:** {dep_rate}%\n\n"
                    f"**Projected value in 5 years:** {format_currency(remaining_value_5yr, currency_option.split()[0])}"
                )
            elif car_age <= 7:
                dep_rate = (
                    10 if brand in ["bmw", "audi", "mercedes_benz", "porsche"] else 12
                )
                remaining_value_5yr = price_eur * ((1 - dep_rate / 100) ** 3)
                st.info(
                    f"**Annual depreciation:** {dep_rate}%\n\n"
                    f"**Projected value in 3 years:** {format_currency(remaining_value_5yr, currency_option.split()[0])}"
                )
            else:
                dep_rate = (
                    5 if brand in ["bmw", "audi", "mercedes_benz", "porsche"] else 8
                )
                st.warning(
                    f"**Annual depreciation:** {dep_rate}%\n\n"
                    f"⚠️ Older vehicles depreciate slowly but face higher maintenance costs"
                )

        # Summary recommendation
        st.markdown("### 🎯 Quick Summary")
        summary_points = []

        price_tier = (
            "Budget-friendly"
            if price_eur < 5000
            else ("Mid-range" if price_eur < 15000 else "Premium")
        )
        summary_points.append(f"💰 **Price Tier:** {price_tier}")

        condition = (
            "Excellent"
            if car_age <= 3 and km_per_year < 10000
            else "Good"
            if car_age <= 7 and km_per_year < 15000
            else "Fair"
        )
        summary_points.append(f"🔧 **Expected Condition:** {condition}")

        reliability = (
            "High"
            if brand in ["bmw", "audi", "mercedes_benz", "toyota", "honda"]
            else "Medium"
        )
        summary_points.append(f"🛡️ **Estimated Reliability:** {reliability}")

        cols = st.columns(len(summary_points))
        for i, point in enumerate(summary_points):
            with cols[i]:
                st.markdown(point)


# -------------------------------------------------
# Sidebar Info Section
# -------------------------------------------------
with st.sidebar:
    st.divider()
    st.markdown("### ℹ️ About")

    with st.expander("🤖 Model Info", expanded=False):
        st.markdown(
            """
        **Algorithm:**
        - HistGradientBoostingRegressor
        - Trained on 100k+ cars
        - Target-encoded brands
        
        **Accuracy:**
        - ±€2-3k confidence
        - Based on test variance
        """
        )

    with st.expander("💡 Tips", expanded=True):
        st.markdown(
            """
        ✅ Use realistic mileage
        
        ✅ Match power to brand
        
        ✅ Check condition
        
        ⚠️ Prices vary by region
        """
        )
