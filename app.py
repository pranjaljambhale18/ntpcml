import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

st.set_page_config(page_title="NTPC Output Prediction App", layout="wide")
st.title("NTPC Plant Output Predictor")
st.write("This app predicts Power, CO2 emissions, Efficiency, and Estimated Profit based on input parameters.")

# Load the trained model
model_path = "ntpc_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload or place the file correctly.")
    st.stop()

model = joblib.load(model_path)

# Initialize prediction history
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("input_form"):
    st.subheader("Enter Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        plf = st.slider("PLF (Plant Load Factor) %", min_value=0, max_value=100, value=75)
        fuel_cost = st.number_input("Fuel Cost (INR/ton)", min_value=1000, value=2500)
        fuel_availability = st.slider("Fuel Availability (%)", min_value=0, max_value=100, value=80)
        load = st.number_input("Load (MW)", min_value=100, value=500)

    with col2:
        heat_rate = st.number_input("Heat Rate (kCal/kWh)", min_value=1000, value=2400)
        ambient_temp = st.slider("Ambient Temperature (°C)", min_value=10, max_value=50, value=30)
        o_m_cost = st.number_input("O&M Cost (INR/unit)", min_value=1, value=3)
        tariff = st.number_input("Tariff (INR/unit)", min_value=1, value=5)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[plf, fuel_cost, fuel_availability, load, heat_rate, ambient_temp, o_m_cost, tariff]])
    prediction = model.predict(input_data)[0]
    predicted_power, predicted_co2, predicted_efficiency, estimated_profit = prediction

    record = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PLF": plf,
        "Fuel Cost": fuel_cost,
        "Fuel Availability": fuel_availability,
        "Load": load,
        "Heat Rate": heat_rate,
        "Ambient Temp": ambient_temp,
        "O&M Cost": o_m_cost,
        "Tariff": tariff,
        "Predicted Power": round(predicted_power, 2),
        "Predicted CO2": round(predicted_co2, 2),
        "Efficiency": round(predicted_efficiency, 2),
        "Estimated Profit": round(estimated_profit, 2)
    }
    st.session_state.history.append(record)

    st.subheader("Prediction Results")
    st.write(f"**Predicted Power (MW):** {round(predicted_power, 2)}")
    st.write(f"**Predicted CO2 Emissions (tons):** {round(predicted_co2, 2)}")
    st.write(f"**Efficiency (%):** {round(predicted_efficiency, 2)}")
    st.write(f"**Estimated Profit (INR):** {round(estimated_profit, 2)}")

    # Suggestions based on thresholds
    st.subheader("Suggestions")
    suggestions = []

    if plf < 50:
        suggestions.append("PLF is very low. Increase utilization to boost power output and reduce per-unit cost.")
    elif 50 <= plf < 70:
        suggestions.append("PLF is below optimal range. Aim for above 70% to increase efficiency and revenue.")
    elif plf > 90:
        suggestions.append("Excellent PLF. Ensure maintenance practices continue supporting high utilization.")

    if fuel_cost > 3500:
        suggestions.append("Fuel cost is high. Explore cheaper alternatives or improve fuel efficiency.")
    elif fuel_cost < 2000:
        suggestions.append("Low fuel costs. Consider long-term contracts to sustain this benefit.")

    if predicted_power < 200:
        suggestions.append("Predicted power is quite low. Consider increasing PLF or fuel availability.")

    if predicted_co2 > 1000000:
        suggestions.append("CO2 emissions are high. Consider switching to cleaner fuel or improving combustion efficiency.")

    if estimated_profit < 1000:
        suggestions.append("Profit is very low. Review O&M costs, fuel costs and tariff strategy.")

    if heat_rate > 2600:
        suggestions.append("High heat rate detected. Improve boiler and turbine performance.")

    if o_m_cost > 5:
        suggestions.append("High O&M cost. Audit maintenance processes to reduce cost.")

    if ambient_temp > 40:
        suggestions.append("High ambient temperature. May affect turbine performance. Consider cooling strategies.")

    if predicted_efficiency < 25:
        suggestions.append("Efficiency is below average. Consider tuning operations and reducing heat losses.")

    if not suggestions:
        suggestions.append("All parameters are within optimal range. Keep monitoring and maintaining the performance.")

    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")

    # Show and export predictions
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download All Predictions as CSV", data=csv, file_name="ntpc_predictions.csv", mime="text/csv")

    st.subheader("All Predictions Made in This Session")
    st.dataframe(df, use_container_width=True)
