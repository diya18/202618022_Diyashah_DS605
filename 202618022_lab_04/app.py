import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    return joblib.load("airbnb_price_artifacts.joblib")

artifacts = load_artifacts()

model = artifacts["model"]
scaler_mean = artifacts["scaler_mean"]
scaler_scale = artifacts["scaler_scale"]
num_cols = artifacts["num_cols"]
cat_cols = artifacts["cat_cols"]
categories = artifacts["categories"]

st.title("🗽 NYC Airbnb Nightly Price Predictor")
st.markdown("Estimate the optimal nightly price for an Airbnb listing in New York City.")

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox(
        "Borough / Area",
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room"]
    )
    latitude = st.number_input("Latitude", value=40.73061, format="%.5f")
    longitude = st.number_input("Longitude", value=-73.93524, format="%.5f")

with col2:
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=2)
    number_of_reviews = st.number_input("Total Number of Reviews", min_value=0, max_value=1200, value=15)
    reviews_per_month = st.number_input("Reviews Per Month", min_value=0.0, max_value=60.0, value=1.2, step=0.1)
    calculated_host_listings_count = st.number_input("Host Total Listings", min_value=1, max_value=400, value=1)
    availability_365 = st.slider("Availability (Days in a Year)", 0, 365, 180)

if st.button("Predict Nightly Price", type="primary"):
    # 1. Numerical array & manual scaling: (X - mean) / scale
    num_vals = np.array([
        latitude,
        longitude,
        minimum_nights,
        number_of_reviews,
        reviews_per_month,
        calculated_host_listings_count,
        availability_365
    ], dtype=float)
    
    num_scaled = (num_vals - scaler_mean) / scaler_scale

    # 2. Manual One-Hot Encoding matching trained categories
    user_cats = [neighbourhood_group, room_type]
    cat_encoded_list = []
    
    for val, cat_list in zip(user_cats, categories):
        for cat in cat_list:
            cat_encoded_list.append(1.0 if val == cat else 0.0)
            
    cat_encoded = np.array(cat_encoded_list, dtype=float)

    # 3. Concatenate and predict
    final_input = np.hstack([num_scaled, cat_encoded]).reshape(1, -1)
    predicted_price = model.predict(final_input)[0]

    st.success(f"### Estimated Price: ${predicted_price:.2f} / night")