# dashboard.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Insight Dashboard", layout="wide")

# STRATEGIC MOVE: Centralized API configuration
API_URL = "http://localhost:8000/predict"

def get_prediction(data):
    try:
        # We send the raw dict; the API's pipeline handles the rest
        response = requests.post(API_URL, json={"data": data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

st.title("📡 Customer Churn Intelligence")
st.markdown("---")

# Organize inputs into logical business sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("💳 Contract & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    charges = st.number_input("Monthly Charges", value=50.0)

# The Prediction Trigger
if st.button("Analyze Churn Risk", use_container_width=True):
    # Constructing the payload exactly as the model expects
    payload = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Tenure in Months": tenure,
        "Contract": contract,
        "Payment Method": method,
        "Monthly Charge": charges
    }
    
    result = get_prediction(payload)
    
    if result:
        st.markdown("---")
        risk_score = result['probability']
        
        # UI Logic: Color coding based on risk
        if risk_score > 0.7:
            st.error(f"### High Risk: {risk_score:.1%}")
            st.warning("Action Required: High-priority retention offer recommended.")
        elif risk_score > 0.4:
            st.warning(f"### Elevated Risk: {risk_score:.1%}")
        else:
            st.success(f"### Low Risk: {risk_score:.1%}")
