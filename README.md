# 🚗 Used Car Price Prediction

A machine learning application that predicts used car prices using a HistGradientBoostingRegressor model trained on European car listings data.

## Features

- 🔮 Real-time price prediction with confidence intervals
- 📊 Brand-aware pricing using target encoding
- 🌐 Multi-currency display (EUR & INR)
- 🎯 Usage-sensitive predictions (km/year analysis)
- 📈 Depreciation insights and market analysis
- 🏆 Premium brand recognition

## Tech Stack

- **ML Framework**: scikit-learn (HistGradientBoostingRegressor)
- **Web App**: Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Caching**: Joblib

## Installation

```bash
# Clone the repository
git clone https://github.com/Arya004pro/used_car_price_prediction.git
cd used_car_price_prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt