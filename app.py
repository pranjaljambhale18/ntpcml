import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("ntpc_model.pkl")

st.set_page_config(page_title="NTPC Predictor", layout="centered")
st.title("🔮 NTPC Power & CO₂ Predictor")
st.markdown("This app predicts power generation, CO₂ emissions, revenue, fuel cost, and profit based on your inputs.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        installed_capacity = st.number_input("Installed Capacity (MW)", value=60000)
        coal_received = st.number_input("Coal Received (MTPA)", value=18500000)
        gas_received = st.number_input("Gas Received (MMSCM)", value=3000)
        plf = st.slider("PLF (%)", 0, 100, 72)

    with col2:
        fuel_cost = st.number_input("Fuel Cost per Unit (₹/kWh)", value=3.2)
        avg_tariff = st.number_input("Average Tariff (₹/kWh)", value=4.0)
        re_share = st.slider("RE Share (%)", 0, 100, 28)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([[
        installed_capacity, coal_received, gas_received,
        plf, fuel_cost, avg_tariff, re_share
    ]], columns=[
        'Installed_Capacity_MW',
        'Coal_Received_MTPA',
        'Gas_Received_MMSCM',
        'PLF_Percentage',
        'Fuel_Cost_per_Unit',
        'Avg_Tariff (ECR)',
        'RE_Share_Percentage'
    ])

    prediction = model.predict(input_df)
    predicted_power, predicted_co2 = prediction[0]

    # Derived metrics
    revenue = predicted_power * avg_tariff * 100
    cost = predicted_power * fuel_cost * 100
    profit = revenue - cost

    st.success("✅ Prediction Complete")
    st.metric("🔋 Predicted Power", f"{predicted_power:.2f} BU")
    st.metric("🌍 Predicted CO₂ Emissions", f"{predicted_co2:,.2f} Tonnes")
    st.metric("💰 Revenue", f"₹{revenue:,.2f} Cr")
    st.metric("🔥 Fuel Cost", f"₹{cost:,.2f} Cr")
    st.metric("📈 Estimated Profit", f"₹{profit:,.2f} Cr")
