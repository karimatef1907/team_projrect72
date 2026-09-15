import streamlit as st
import pandas as pd
import joblib

kmeans = joblib.load("kmeans_model.pkl")
preprocessor = joblib.load("cluster_preprocessor.pkl")
price_model = joblib.load("price_model.pkl")

st.title("Car Price Prediction & Segmentation")

brand = st.text_input("Brand")
model = st.text_input("Model")

model_year = st.number_input(
    "Model Year",
    min_value=1980,
    max_value=2026,
    value=2020
)

milage = st.number_input(
    "Mileage",
    min_value=0,
    value=30000
)

fuel_type = st.text_input("Fuel Type")
engine = st.text_input("Engine")
transmission = st.text_input("Transmission")
ext_col = st.text_input("Exterior Color")
int_col = st.text_input("Interior Color")
accident = st.text_input("Accident")
clean_title = st.text_input("Clean Title")

if st.button("Predict"):

    data = pd.DataFrame([{
        "brand": brand,
        "model": model,
        "model_year": model_year,
        "milage": milage,
        "fuel_type": fuel_type,
        "engine": engine,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "clean_title": clean_title
    }])

    data_transformed = preprocessor.transform(data)

    cluster = kmeans.predict(data_transformed)[0]

    predicted_price = price_model.predict(data)[0]

    segments = {
        0: "Premium / New Cars",
        1: "Budget / Older Cars",
        2: "Mid-Range Cars"
    }

    st.success(f"Predicted Segment: Cluster {cluster}")
    st.info(f"Segment Type: {segments[cluster]}")
    st.metric("Predicted Price", f"${predicted_price:,.0f}")