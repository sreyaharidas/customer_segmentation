import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("customer_clustering_model.pkl")

st.title("Customer Clustering")

# User inputs
income = st.number_input(
    "Annual Income (INR)",
    min_value=0,
    value=500000
)

spending = st.number_input(
    "Monthly Spending (INR)",
    min_value=0,
    value=10000
)

visits = st.number_input(
    "Visits Per Month",
    min_value=0,
    value=5
)

# Prediction
if st.button("Predict Cluster"):

    input_data = pd.DataFrame({
        "Annual_Income_INR": [income],
        "Monthly_Spending_INR": [spending],
        "Visits_Per_Month": [visits]
    })

    cluster = model.predict(input_data)[0]

    st.success(f"Customer belongs to Cluster {cluster}")
